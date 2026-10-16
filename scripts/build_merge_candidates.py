"""
Builds the data the merge tool needs: ranked merge candidates for one session.

WHAT A MERGE DECISION ACTUALLY NEEDS, AND WHERE EACH PIECE COMES FROM
---------------------------------------------------------------------
Two clusters are the same neuron if (a) they are in the same place, (b) they look the
same, and (c) they never fire too close together in time. (c) is the one that settles
it, and it is the one Phy's similarity score cannot see:

  same place   peak-channel distance in MICRONS from channel_positions.npy -- not
               channel index, which is meaningless on a staggered Neuropixels map.
  same shape   mean waveform read from the RAW probe1.dat on the union of both units'
               nearby channels, so the two are compared on the same channels in the
               same microvolts. Kilosort's similar_templates.npy is carried along for
               reference, but it compares TEMPLATES -- two different neurons with
               similar-shaped waveforms score high, which is exactly the error mode
               this tool exists to avoid.
  never both   the cross-correlogram. One neuron cannot fire twice within its
               refractory period, so if the two clusters are one neuron the CCG has a
               DIP at zero lag. If they are two neurons it does not. Everything else
               is circumstantial; this is evidence.

Plus the case the CCG cannot catch on its own: a DRIFT PAIR, where one unit fades out
as the other fades in. They then have almost no overlapping spikes, so the CCG is
empty and uninformative -- but their firing-rate timelines are complementary, which is
visible immediately and is why the session timeline is in here too.

OUTPUT: outputs/merge_candidates_<session>.json -- consumed by the merge tool artifact.
Pair-level metrics, per-unit waveforms/ACGs/timelines, and the session meta needed to
turn a decision into unit_merges.json.

Usage:
  python build_merge_candidates.py <session_id> [--rec_root F:/Gil/Shamir]
                                   [--max_dist_um 60] [--max_pairs 150]
                                   [--n_waveform_spikes 150]

Needs numpy 2 for nothing in particular, but does need the review-data JSON built
earlier (outputs/pipeline_review_data_<session>.json) and the session timeline
(outputs/session_timeline_<session>.json).
"""
import argparse
import base64
import json
import os
import sys
import time

import numpy as np

FS = 30000.0
OUT_DIR = r"D:\Gil\spike_sorting_agent\outputs"

# CCG/ACG geometry. 0.5 ms bins to 30 ms: fine enough to resolve a refractory dip,
# coarse enough that a pair with a few thousand spikes still has usable counts.
CCG_BIN_MS = 0.5
CCG_MAX_LAG_MS = 30.0
# The refractory window a real neuron cannot violate. 1.5 ms is conservative for
# striatal units (MSNs sit around 2 ms); anything inside it is contamination.
REFRACTORY_MS = 1.5
# Waveform window around each spike, in samples at 30 kHz: -0.67 ms to +1.33 ms.
WF_PRE, WF_POST = 20, 40
# How many channels around a unit's peak to read. Enough to see the footprint fall
# off on both sides without making the payload large.
N_NEAR_CHANNELS = 10


def log(msg):
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


T0 = time.time()


# ----------------------------------------------------------------- correlograms
def correlogram(t1_s, t2_s, bin_ms=CCG_BIN_MS, max_lag_ms=CCG_MAX_LAG_MS, auto=False):
    """Counts of (t2 - t1) differences per lag bin, in spikes.

    Implemented by searchsorted rather than a full outer difference: a pair of units
    here can have 400k spikes each, and we only ever care about differences inside
    +/-30 ms.
    """
    max_lag = max_lag_ms / 1000.0
    edges = np.arange(-max_lag, max_lag + bin_ms / 1000.0 / 2, bin_ms / 1000.0)
    hist = np.zeros(len(edges) - 1, dtype=np.int64)
    if len(t1_s) == 0 or len(t2_s) == 0:
        return hist, edges
    lo = np.searchsorted(t2_s, t1_s - max_lag, side="left")
    hi = np.searchsorted(t2_s, t1_s + max_lag, side="right")
    n_per = hi - lo
    # Vectorised ragged gather: every t2 spike inside each t1 spike's window, with no
    # Python-level loop over spikes. A unit here can have 400k spikes and this runs
    # 150 times, so the loop version is the difference between seconds and an hour.
    total = int(n_per.sum())
    if total == 0:
        return hist, edges
    src = np.repeat(np.arange(len(t1_s)), n_per)
    starts = np.repeat(lo, n_per)
    offsets = np.arange(total) - np.repeat(np.cumsum(n_per) - n_per, n_per)
    d = t2_s[starts + offsets] - t1_s[src]
    if auto:
        d = d[d != 0.0]              # drop each spike against itself
    hist += np.histogram(d, bins=edges)[0]
    return hist, edges


def ccg_metrics(hist, edges):
    """Turn a CCG into the numbers a merge decision rests on.

    `dip_ratio` is the heart of it: counts inside the refractory window divided by
    what the correlogram's own flanks predict for a window that size. One neuron
    cannot fire inside its own refractory period, so:

        ~0.0 - 0.3   a real dip. Consistent with ONE neuron.
        ~1.0         flat. Two independent neurons -- do not merge.
        >1.5         a peak at zero lag, which means synchrony, not identity
                     (common input, or the same spike detected twice on two channels).

    Reported with the flank rate it was measured against, because a ratio computed
    from nearly no spikes is not a measurement.
    """
    centers = (edges[:-1] + edges[1:]) / 2 * 1000.0
    refr = np.abs(centers) <= REFRACTORY_MS
    flank = np.abs(centers) >= 10.0
    n_refr = int(hist[refr].sum())
    flank_per_bin = float(hist[flank].mean()) if flank.any() else 0.0
    expected = flank_per_bin * int(refr.sum())
    return {
        "n_refractory": n_refr,
        "n_total": int(hist.sum()),
        "flank_per_bin": round(flank_per_bin, 3),
        "dip_ratio": (round(n_refr / expected, 4) if expected > 0 else None),
        "expected_refractory": round(expected, 2),
        "peak_lag_ms": round(float(centers[int(np.argmax(hist))]), 2),
    }


def refractory_violations(t_s, refractory_ms=REFRACTORY_MS):
    """Fraction of consecutive ISIs shorter than the refractory period.

    Computed for each unit alone AND for the combined train, because that difference
    is what a merge costs: merging two clean units into something with 4% violations
    means they were not one neuron, whatever the waveforms look like.
    """
    if len(t_s) < 2:
        return {"n": int(len(t_s)), "violation_pct": None}
    isi = np.diff(np.sort(t_s))
    return {"n": int(len(t_s)),
            "violation_pct": round(100.0 * float((isi < refractory_ms / 1000.0).mean()), 3)}


# ----------------------------------------------------------------- raw waveforms
def mean_waveforms(memmap, n_channels, spike_rows, channels, n_spikes):
    """Mean raw waveform on `channels`, from up to `n_spikes` spikes spread evenly.

    Spikes are sampled across the WHOLE session rather than from the start: a drift
    pair's waveform changes over time, and a mean built from the first 150 spikes
    would describe a stretch the unit may not even be present for.
    """
    if len(spike_rows) == 0:
        return np.zeros((WF_PRE + WF_POST, len(channels)), dtype=np.float32), 0
    idx = np.linspace(0, len(spike_rows) - 1, min(n_spikes, len(spike_rows))).astype(int)
    rows = spike_rows[idx]
    acc = np.zeros((WF_PRE + WF_POST, len(channels)), dtype=np.float64)
    used = 0
    n_samples = memmap.shape[0]
    for r in rows:
        a, b = int(r) - WF_PRE, int(r) + WF_POST
        if a < 0 or b > n_samples:
            continue
        acc += memmap[a:b, channels].astype(np.float64)
        used += 1
    if used == 0:
        return np.zeros((WF_PRE + WF_POST, len(channels)), dtype=np.float32), 0
    wf = acc / used
    wf -= wf[:5].mean(axis=0)        # baseline the pre-spike samples to zero
    return wf.astype(np.float32), used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    ap.add_argument("--rec_root", default="F:/Gil/Shamir")
    ap.add_argument("--ks_subdir", default="kilosort4")
    ap.add_argument("--max_dist_um", type=float, default=60.0,
                    help="peak channels further apart than this are not candidates; "
                         "a neuron's footprint does not jump across the probe")
    ap.add_argument("--max_pairs", type=int, default=150)
    ap.add_argument("--n_waveform_spikes", type=int, default=150)
    ap.add_argument("--uv_per_bit", type=float, default=0.195,
                    help="Neuropixels 1.0 at gain 500: 0.195 uV per LSB")
    args = ap.parse_args()

    sid = args.session_id
    ks = os.path.join(args.rec_root, sid + ".rec", sid + ".kilosort", args.ks_subdir)
    review_path = os.path.join(OUT_DIR, "pipeline_review_data_%s.json" % sid)
    timeline_path = os.path.join(OUT_DIR, "session_timeline_%s.json" % sid)
    for p in (ks, review_path, timeline_path):
        if not os.path.exists(p):
            raise SystemExit("missing input: %s" % p)

    # ------------------------------------------------------------------ inputs
    log("loading clustering from %s" % ks)
    spike_rows = np.load(os.path.join(ks, "spike_times.npy")).ravel()
    clusters = np.load(os.path.join(ks, "spike_clusters.npy")).ravel()
    chan_pos = np.load(os.path.join(ks, "channel_positions.npy"))
    sim = np.load(os.path.join(ks, "similar_templates.npy"))
    log("%d spikes, %d clusters, similar_templates %s"
        % (len(spike_rows), len(np.unique(clusters)), sim.shape))

    review = {u["unit_id"]: u for u in json.load(open(review_path))}
    timeline = json.load(open(timeline_path))
    log("review data: %d units; timeline: %d bins of %gs"
        % (len(review), timeline["n_bins"], timeline["bin_s"]))

    # Spike times in SECONDS on the probe1.dat row clock. Lags and ISIs only ever
    # involve differences, so the 87.5 s offset to absolute Trodes time is irrelevant
    # here -- and deliberately not applied, to keep this independent of the sync.
    t_all = spike_rows.astype(np.float64) / FS
    order = np.argsort(t_all)
    t_all, clusters_sorted, rows_sorted = t_all[order], clusters[order], spike_rows[order]

    # Per-unit spike times, excluding units the classification already called noise:
    # merging noise into anything is not a decision worth surfacing.
    unit_ids = sorted(u for u in review if not review[u].get("is_noise"))
    log("%d non-noise units of %d" % (len(unit_ids), len(review)))
    by_unit, by_unit_rows = {}, {}
    for uid in unit_ids:
        m = clusters_sorted == uid
        by_unit[uid] = t_all[m]
        by_unit_rows[uid] = rows_sorted[m]

    peak_ch = {uid: int(review[uid]["peak_channel"]) for uid in unit_ids}
    pos = {uid: chan_pos[peak_ch[uid]] for uid in unit_ids}

    # --------------------------------------------------- candidate pair shortlist
    # Geometry first, because it is the one criterion that is never wrong: a neuron
    # has one position, so two clusters far apart on the probe are two neurons, no
    # matter how similar their waveforms look.
    log("pairing units within %.0f um" % args.max_dist_um)
    cands = []
    for i, a in enumerate(unit_ids):
        for b in unit_ids[i + 1:]:
            d = float(np.hypot(*(pos[a] - pos[b])))
            if d > args.max_dist_um:
                continue
            s = float(sim[a, b]) if a < sim.shape[0] and b < sim.shape[1] else 0.0
            cands.append((a, b, d, s))
    log("%d pairs within range" % len(cands))

    # Rank by template similarity before the expensive work: the CCG is the decider,
    # but it costs real time per pair, and similarity is a cheap way to decide which
    # pairs are worth computing it for. (NOT a way to decide the merge -- see the
    # module docstring.)
    cands.sort(key=lambda c: -c[3])
    cands = cands[:args.max_pairs]
    log("keeping top %d by template similarity for full analysis" % len(cands))

    # ------------------------------------------------------- per-unit raw waveforms
    involved = sorted({u for c in cands for u in c[:2]})
    log("%d units involved; reading raw waveforms" % len(involved))
    dat = os.path.join(args.rec_root, sid + ".rec", sid + ".kilosort",
                       sid + ".probe1.dat")
    n_channels = chan_pos.shape[0]
    mm = np.memmap(dat, dtype=np.int16, mode="r")
    mm = mm.reshape(-1, n_channels)
    log("probe1.dat memmapped: %s samples x %d channels" % (mm.shape[0], n_channels))

    # Channels near each unit's peak, by actual distance on the probe.
    near = {}
    for uid in involved:
        d = np.hypot(chan_pos[:, 0] - pos[uid][0], chan_pos[:, 1] - pos[uid][1])
        near[uid] = np.argsort(d)[:N_NEAR_CHANNELS]

    waveforms = {}
    for n, uid in enumerate(involved):
        wf, used = mean_waveforms(mm, n_channels, by_unit_rows[uid], near[uid],
                                  args.n_waveform_spikes)
        waveforms[uid] = {
            "channels": near[uid].tolist(),
            "channel_um": [[float(chan_pos[c][0]), float(chan_pos[c][1])]
                           for c in near[uid]],
            # Stored in MICROVOLTS, rounded to 0.1: the whole point is comparing two
            # units on one absolute scale, so a normalised shape would hide the very
            # thing you need to see.
            "uv": np.round(wf * args.uv_per_bit, 1).T.tolist(),
            "n_spikes_averaged": used,
        }
        if (n + 1) % 10 == 0:
            log("  waveforms %d/%d" % (n + 1, len(involved)))

    # ------------------------------------------------------------ per-unit payload
    counts_raw = timeline["counts"]

    def decode_timeline(uid):
        b = counts_raw.get(str(uid)) or counts_raw.get(uid)
        if b is None:
            return None
        return np.frombuffer(base64.b64decode(b), dtype=np.uint16).tolist()

    units_out = {}
    for uid in involved:
        r = review[uid]
        units_out[str(uid)] = {
            "unit_id": uid,
            "label": r.get("label"),
            "is_mua": bool(r.get("is_mua")),
            "n_spikes": int(r.get("n_spikes") or len(by_unit[uid])),
            "mean_rate_hz": round(float(r.get("mean_rate_hz") or 0.0), 3),
            "peak_channel": peak_ch[uid],
            "trodes_channel": r.get("trodes_channel"),
            "depth_um": float(pos[uid][1]),
            "x_um": float(pos[uid][0]),
            "violation_ratio": r.get("violation_ratio"),
            "sep_vs_noise": r.get("sep_vs_noise"),
            "sep_vs_other": r.get("sep_vs_other"),
            "acg_hist": r.get("acg_hist"),
            "acg_bin_ms": r.get("acg_bin_ms"),
            "acg_max_lag_ms": r.get("acg_max_lag_ms"),
            "rate_timeline": decode_timeline(uid),
            "waveform": waveforms[uid],
            "refractory": refractory_violations(by_unit[uid]),
        }

    # ------------------------------------------------------------- per-pair metrics
    log("computing CCGs and merge metrics for %d pairs" % len(cands))
    pairs_out = []
    for n, (a, b, dist, s) in enumerate(cands):
        ta, tb = by_unit[a], by_unit[b]
        hist, edges = correlogram(ta, tb)
        cm = ccg_metrics(hist, edges)

        combined = np.sort(np.concatenate([ta, tb]))
        merged_refr = refractory_violations(combined)
        merged_acg, _ = correlogram(combined, combined, auto=True)

        # Waveform agreement on the channels the two units SHARE, in microvolts.
        # Correlation says "same shape", the amplitude ratio says "same size" -- a
        # pair can match on one and fail the other, and that is informative.
        wa, wb = waveforms[a], waveforms[b]
        shared = [c for c in wa["channels"] if c in wb["channels"]]
        if shared:
            ia = [wa["channels"].index(c) for c in shared]
            ib = [wb["channels"].index(c) for c in shared]
            va = np.array(wa["uv"], dtype=np.float64)[ia].ravel()
            vb = np.array(wb["uv"], dtype=np.float64)[ib].ravel()
            wf_corr = (float(np.corrcoef(va, vb)[0, 1])
                       if va.std() > 0 and vb.std() > 0 else None)
            amp_a = float(np.abs(np.array(wa["uv"])).max())
            amp_b = float(np.abs(np.array(wb["uv"])).max())
            amp_ratio = round(min(amp_a, amp_b) / max(amp_a, amp_b), 4) if max(amp_a, amp_b) else None
        else:
            wf_corr, amp_ratio = None, None

        # Drift signature: do the two units' rate timelines trade places? A strongly
        # negative correlation means one is present when the other is not, which is
        # the one case where an uninformative CCG is expected rather than damning.
        ra, rb = units_out[str(a)]["rate_timeline"], units_out[str(b)]["rate_timeline"]
        if ra and rb:
            va_, vb_ = np.array(ra, dtype=float), np.array(rb, dtype=float)
            rate_corr = (round(float(np.corrcoef(va_, vb_)[0, 1]), 4)
                         if va_.std() > 0 and vb_.std() > 0 else None)
            active_a = va_ > 0.2 * va_.max() if va_.max() else np.zeros(len(va_), bool)
            active_b = vb_ > 0.2 * vb_.max() if vb_.max() else np.zeros(len(vb_), bool)
            both = int((active_a & active_b).sum())
            either = int((active_a | active_b).sum())
            overlap = round(both / either, 4) if either else None
        else:
            rate_corr, overlap = None, None

        pairs_out.append({
            "pair_id": "%d_%d" % (a, b),
            "unit_a": a, "unit_b": b,
            "distance_um": round(dist, 1),
            "template_similarity": round(s, 4),
            "ccg_hist": hist.tolist(),
            "ccg_bin_ms": CCG_BIN_MS,
            "ccg_max_lag_ms": CCG_MAX_LAG_MS,
            "ccg": cm,
            "merged_acg_hist": merged_acg.tolist(),
            "merged_refractory": merged_refr,
            "refractory_a": units_out[str(a)]["refractory"],
            "refractory_b": units_out[str(b)]["refractory"],
            "waveform_corr": (round(wf_corr, 4) if wf_corr is not None else None),
            "amplitude_ratio": amp_ratio,
            "rate_corr": rate_corr,
            "active_overlap": overlap,
        })
        if (n + 1) % 25 == 0:
            log("  pairs %d/%d" % (n + 1, len(cands)))

    # Order the list the way a human should work through it: the CCG dip first,
    # because that is the evidence. Pairs with too few coincident spikes to measure
    # a dip sink to the bottom -- not because they are bad, but because the decision
    # for them has to be made on the timelines instead.
    def rank(p):
        dip = p["ccg"]["dip_ratio"]
        measurable = p["ccg"]["flank_per_bin"] >= 0.5
        if dip is not None and measurable:
            return (0, dip)
        return (1, -(p["template_similarity"] or 0))

    pairs_out.sort(key=rank)

    doc = {
        "format": "merge_candidates/1",
        "session_id": sid,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "built_by": "scripts/build_merge_candidates.py",
        "kilosort_dir": ks,
        "fs_hz": FS,
        "uv_per_bit": args.uv_per_bit,
        "refractory_ms": REFRACTORY_MS,
        "max_dist_um": args.max_dist_um,
        "n_units_total": len(review),
        "n_units_non_noise": len(unit_ids),
        "n_pairs_in_range": len(cands),
        "timeline": {"bin_s": timeline["bin_s"], "n_bins": timeline["n_bins"],
                     "t0_s": timeline["ephys_t0_s"], "t1_s": timeline["ephys_t1_s"]},
        # Everything needed to turn a decision into unit_merges.json, so the tool can
        # emit a manifest that the pipeline will accept without a second lookup.
        "clustering": {
            "spike_clusters_sha1": None,   # filled below
            "n_spikes": int(len(clusters)),
            "n_clusters": int(len(np.unique(clusters))),
        },
        "units": units_out,
        "pairs": pairs_out,
    }

    import hashlib
    with open(os.path.join(ks, "spike_clusters.npy"), "rb") as f:
        doc["clustering"]["spike_clusters_sha1"] = hashlib.sha1(f.read()).hexdigest()

    out = os.path.join(OUT_DIR, "merge_candidates_%s.json" % sid)
    with open(out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    log("wrote %s (%.1f MB)" % (out, os.path.getsize(out) / 1e6))

    # A short printed summary, so a run is self-checking rather than something you
    # have to open the JSON to believe.
    log("top candidates by CCG dip:")
    for p in pairs_out[:12]:
        log("  %-10s d=%5.1fum sim=%.3f dip=%-8s flank=%-6.2f wf_r=%-7s "
            "rate_r=%-7s merged_viol=%s%%"
            % (p["pair_id"], p["distance_um"], p["template_similarity"],
               p["ccg"]["dip_ratio"], p["ccg"]["flank_per_bin"],
               p["waveform_corr"], p["rate_corr"],
               p["merged_refractory"]["violation_pct"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

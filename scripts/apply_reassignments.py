# -*- coding: utf-8 -*-
"""Apply the hand-drawn regions to the WHOLE session, and measure whether that is safe.

    python scripts/apply_reassignments.py 20260925_103539 \
        --rec-root Z:/Gil/Shamir_1 \
        --dumps outputs/review_verdicts/20260925_103539/raw/reassignments

WHAT GIL DECIDED, AND WHY IT NEEDS THIS SCRIPT
----------------------------------------------
The review tool draws ten 120 s windows -- 20 minutes of a session that runs nearly
two hours. A region lassoed there is a decision about a BOUNDARY IN FEATURE SPACE, not
about those 1200 seconds. Gil chose (2026-11-19) that a region applies to the whole
session, with a drift check, rather than only inside the windows it was drawn on:

  * applying inside the windows only would clean a unit for 17 % of the recording and
    leave it uncleaned for the rest, so the unit changes identity partway through --
    which breaks every analysis that compares early trials to late ones, and that is
    most of them;
  * applying session-wide assumes the cloud sits where it sat when he drew it, and
    clouds do move as the probe settles. So the assumption is MEASURED here rather
    than hoped for, and the measurement is shown to him before anything is written.

WHY THIS IS REPRODUCIBLE AT ALL
-------------------------------
Because build_cluster_windows.py stored the PCA basis in its own file instead of
refitting per window. pc1 means the same thing in minute 3 and minute 115 only if it
is the same axis, with the same sign convention and the same mean subtracted. That
file (cluster_basis_<sid>.json) is what makes a boundary drawn in one window
meaningful in another, and it is the reason this step can exist.

WHAT IT COSTS
-------------
probe1.dat is sample-major: pulling one channel's span costs the same read as pulling
all 384, so the price is the SPAN, and the span here is the whole recording -- 255 GB
for 20260925. One pass, every channel that carries a region, roughly an hour. That is
the honest price of a session-wide decision and it is paid once per session.

WHAT IT WRITES -- AND WHAT IT DELIBERATELY DOES NOT
---------------------------------------------------
Nothing is written into the sort. No spike_clusters.npy is rewritten, exactly as with
merges: the result is a MANIFEST that a loader applies, so the Kilosort output on Z
stays the thing Kilosort produced and every decision stays reversible.

    outputs/reassign_scan_<sid>/chan<NNN>.npz   every detected event on that channel:
                                                sample time, the six features, and the
                                                sorted unit it matched (-1 = none)
    outputs/reassignment_manifest_<sid>.npz     per record, the KILOSORT SPIKE INDICES
                                                to relabel, and the sample times of
                                                events that matched no spike at all
    outputs/reassignment_manifest_<sid>.json    what each record does, its counts, and
                                                its drift measurement

THE PART THAT IS NOT A RELABEL
------------------------------
A region whose sources include `null` is asking for events that matched NO sorted
spike -- the unassigned pile -- to join a unit. Those are not relabels: Kilosort never
detected them, so there is no spike to relabel, and honouring them means INSERTING
spikes (new times, new cluster ids) rather than moving existing ones. Nine of
20260925's eighteen regions are of this kind. They are counted, measured and written
to the manifest as `insert`, kept separate from `relabel`, and no loader applies them
until that is a decision taken on purpose.
"""
from __future__ import print_function

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from striatal_agent.config import AgentConfig          # noqa: E402
from striatal_agent.session import discover_session    # noqa: E402
from striatal_agent.raw_io import RawReader, highpass_filter  # noqa: E402
from striatal_agent.paths import out_path              # noqa: E402

FS = 30000
NT, NT0MIN = 60, 20
DEAD_SAMPLES = int(0.001 * FS)
MATCH_WINDOW = int(0.0005 * FS)
NOISE = -1
SLICE_SEC = 10
PAD = 200
TEO_SCALE = 1000.0
SCALED = ("teo_max", "psi_trough", "teo2")
FIELDS = ("pc1", "pc2", "ptp", "teo_max", "psi_trough", "teo2")

T0 = time.time()


def log(msg):
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


# ----------------------------------------------------------------- detection, as built
def detect_trace(trace, start_abs):
    """Identical to build_cluster_windows.detect_trace -- the same events, or none of
    this is comparable with what Gil drew on."""
    mad = np.median(np.abs(trace - np.median(trace))) * 1.4826
    if not np.isfinite(mad) or mad <= 0:
        return None
    below = trace < -4.5 * mad
    crossings = np.where(below & ~np.roll(below, 1))[0]
    keep, last = [], -DEAD_SAMPLES
    for c in crossings:
        if c - last >= DEAD_SAMPLES:
            keep.append(c)
            last = c
    crossings = np.array(keep, dtype=np.int64)
    crossings = crossings[(crossings > NT0MIN + 2)
                          & (crossings < len(trace) - (NT - NT0MIN) - 2)]
    if len(crossings) == 0:
        return None
    snips = np.array([trace[c - NT0MIN:c - NT0MIN + NT] for c in crossings],
                     dtype=np.float32)
    return crossings + start_abs, snips


def shape_features(snips):
    ptp = snips.max(axis=1) - snips.min(axis=1)
    teo_max = (snips[:, 1:-1] ** 2 - snips[:, :-2] * snips[:, 2:]).max(axis=1)
    trough = np.clip(np.argmin(snips, axis=1), 1, NT - 2)
    rows = np.arange(len(snips))
    psi_trough = (snips[rows, trough] ** 2
                  - snips[rows, trough - 1] * snips[rows, trough + 1])
    teo2 = (snips[:, 2:-2] ** 2 - snips[:, :-4] * snips[:, 4:]).max(axis=1)
    return ptp, teo_max, psi_trough, teo2


def match_units(times, spike_times, spike_clusters):
    """Which sorted unit each detection is, within half a millisecond, or NOISE.

    Vectorised: the loop in build_cluster_windows costs a Python iteration per event,
    which is fine for 40 000 events a window and is not fine for the nine million this
    pass produces. Same answer -- the nearest spike within the match window.
    """
    who = np.full(len(times), NOISE, dtype=np.int64)
    if not len(spike_times):
        return who
    pos = np.searchsorted(spike_times, times)
    best = np.full(len(times), MATCH_WINDOW + 1, dtype=np.int64)
    for off in (-2, -1, 0, 1):
        j = np.clip(pos + off, 0, len(spike_times) - 1)
        d = np.abs(spike_times[j] - times)
        take = (d <= MATCH_WINDOW) & (d < best)
        who[take] = spike_clusters[j][take]
        best[take] = d[take]
    return who


def scan(reader, cfg, channels, basis, spike_times, spike_clusters,
         start, stop, out_dir, flush_every=120):
    """One sequential pass over [start, stop), detecting on every channel that matters.

    Writes one npz per channel. The read is the cost, and the read is per SPAN, so
    every channel rides along on the same bytes.
    """
    acc = dict((ch, {"t": [], "u": [],
                     "f": dict((k, []) for k in FIELDS)}) for ch in channels)
    step = SLICE_SEC * FS
    s, n_done, last_log = start, 0, time.time()
    chan_idx = np.array(channels)
    while s < stop:
        s1 = min(stop, s + step)
        a, b = max(0, s - PAD), min(reader.n_samples, s1 + PAD)
        blk = np.asarray(reader._arr[a:b, chan_idx])       # only the columns we need
        off = s - a
        for k, ch in enumerate(channels):
            filt = highpass_filter(blk[:, k].astype(np.float32), FS, cfg.highpass_hz,
                                   axis=0)
            got = detect_trace(filt[off:off + (s1 - s)], s)
            if got is None:
                continue
            times, snips = got
            centred = snips - basis[ch][1]
            acc[ch]["t"].append(times)
            acc[ch]["u"].append(match_units(times, spike_times, spike_clusters))
            acc[ch]["f"]["pc1"].append(centred @ basis[ch][0][0])
            acc[ch]["f"]["pc2"].append(centred @ basis[ch][0][1])
            ptp, teo_max, psi_trough, teo2 = shape_features(snips)
            acc[ch]["f"]["ptp"].append(ptp)
            acc[ch]["f"]["teo_max"].append(teo_max)
            acc[ch]["f"]["psi_trough"].append(psi_trough)
            acc[ch]["f"]["teo2"].append(teo2)
        del blk
        s = s1
        n_done += 1
        if time.time() - last_log > flush_every:
            frac = (s - start) / float(stop - start)
            el = time.time() - T0
            log("  %.1f%% of the session -- %.0f min elapsed, ~%.0f min left"
                % (100 * frac, el / 60.0, (el / max(frac, 1e-6) - el) / 60.0))
            last_log = time.time()

    for ch in channels:
        if not acc[ch]["t"]:
            continue
        d = {"time": np.concatenate(acc[ch]["t"]).astype(np.int64),
             "unit": np.concatenate(acc[ch]["u"]).astype(np.int32)}
        for k in FIELDS:
            v = np.concatenate(acc[ch]["f"][k]).astype(np.float32)
            d[k] = v / TEO_SCALE if k in SCALED else v
        np.savez_compressed(os.path.join(out_dir, "chan%03d.npz" % ch), **d)
        log("  channel %3d: %d events -> chan%03d.npz" % (ch, len(d["time"]), ch))


# ------------------------------------------------------------------ polygon membership
def inside(xs, ys, poly):
    """Even-odd ray casting, vectorised over points. The page's own test, so a dot it
    showed as selected is a spike this calls selected."""
    poly = np.asarray(poly, dtype=np.float64)
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    hit = np.zeros(len(xs), dtype=bool)
    for k in range(len(poly)):
        a, b_, c, d = x1[k], y1[k], x2[k], y2[k]
        straddles = (d > ys) != (b_ > ys)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (c - a) * (ys - b_) / np.where(d == b_, np.nan, d - b_) + a
        hit ^= straddles & (xs < xint)
    return hit


def load_records(dump_dir):
    recs = []
    for f in sorted(glob.glob(os.path.join(dump_dir, "*.json"))):
        d = json.load(open(f))
        d["_doc_id"] = os.path.splitext(os.path.basename(f))[0]
        recs.append(d)
    recs.sort(key=lambda r: r.get("ts", 0))
    return recs


def sources_of(rec):
    """Source owners as the page means them: a unit id, or None for the unassigned."""
    if rec.get("source_units"):
        return list(rec["source_units"])
    return [rec.get("source_unit")]


def owner_keys(matched, units_here):
    """The page's three categories, which are NOT the same as the match result.

    build_cluster_windows calls a detection "other" only when it matched a unit that
    PEAKS ON THIS CHANNEL. A detection matching a unit 500 um away -- which happens
    constantly, since a distant spike still crosses threshold here -- is drawn in the
    grey "unassigned" pile along with the detections that matched nothing at all.

    So the grey cloud is two different things, and the distinction decides what a
    region sourced from `null` actually does: taking a grey dot that matched nothing
    inserts a spike, while taking a grey dot that matched an off-channel unit TAKES
    THAT SPIKE AWAY FROM ANOTHER UNIT. Returning the key the page used, plus the real
    match, is what lets the two be counted separately.
    """
    here = np.isin(matched, list(units_here)) if units_here \
        else np.zeros(len(matched), dtype=bool)
    key = np.where(here, matched, NOISE)
    return key.astype(np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id")
    ap.add_argument("--rec-root", default=r"Z:\Gil\Shamir_1")
    ap.add_argument("--dumps", default=None,
                    help="the reassignments dump directory (default: the in-repo copy)")
    ap.add_argument("--minutes", type=float, default=0.0,
                    help="scan only the first N minutes -- for proving the code, never "
                         "for a decision. 0 means the whole session.")
    ap.add_argument("--skip-scan", action="store_true",
                    help="reuse an existing scan and only redo the measurement")
    args = ap.parse_args()

    sid = args.session_id
    dumps = args.dumps or out_path(os.path.join(
        "review_verdicts", sid, "raw", "reassignments"))
    scan_dir = out_path("reassign_scan_%s" % sid)
    if not os.path.isdir(scan_dir):
        os.makedirs(scan_dir)

    recs = load_records(dumps)
    if not recs:
        raise SystemExit("no reassignment records in %s" % dumps)
    log("%d region(s) drawn" % len(recs))

    windows = json.load(open(out_path("cluster_windows_%s.json" % sid)))
    basis_doc = json.load(open(out_path("cluster_basis_%s.json" % sid)))
    peak_of = dict((int(u), int(d["peak_channel"]))
                   for u, d in windows["units"].items())
    units_on = {}
    for uid, pch in peak_of.items():
        units_on.setdefault(pch, set()).add(uid)
    basis = {}
    for ch, d in basis_doc["channels"].items():
        basis[int(ch)] = (np.array(d["vt"], dtype=np.float32),
                          np.array(d["mean_snippet"], dtype=np.float32))

    # Every record is drawn on ONE unit's panel, so it lives on that unit's channel.
    for r in recs:
        su = r.get("scatter_unit", r.get("source_unit"))
        r["_channel"] = peak_of[int(su)]
        r["_scatter_unit"] = int(su)
    channels = sorted({r["_channel"] for r in recs})
    log("regions sit on %d channel(s): %s" % (len(channels), channels))

    cfg = AgentConfig()
    paths = discover_session(os.path.join(args.rec_root, sid + ".rec"))
    ks = paths.kilosort_original_dir
    spike_times = np.load(os.path.join(ks, "spike_times.npy")).ravel().astype(np.int64)
    spike_clusters = np.load(os.path.join(ks, "spike_clusters.npy")).ravel()
    order = np.argsort(spike_times, kind="stable")
    spike_times, spike_clusters = spike_times[order], spike_clusters[order]
    # The manifest has to name spikes in the FILE's order, not this sorted order, or a
    # loader applying it would relabel the wrong spikes.
    orig_index = order.astype(np.int64)

    reader = RawReader(paths.probe_dat, cfg.n_chan)
    stop = reader.n_samples
    if args.minutes:
        stop = min(stop, int(args.minutes * 60 * FS))
        log("TRIAL RUN: first %.1f min only -- not a basis for any decision"
            % args.minutes)
    log("probe1.dat %.1f min, scanning to %.1f min"
        % (reader.n_samples / FS / 60.0, stop / FS / 60.0))

    if not args.skip_scan:
        scan(reader, cfg, channels, basis, spike_times, spike_clusters, 0, stop,
             scan_dir)

    # ------------------------------------------------------------------ the measurement
    bin_s = 120.0
    n_bins = int(np.ceil(stop / FS / bin_s))
    drawn = set()
    for w in windows["windows"]:
        drawn.add(int(w["t0_s"] // bin_s))

    per_chan = {}
    for ch in channels:
        p = os.path.join(scan_dir, "chan%03d.npz" % ch)
        if os.path.exists(p):
            per_chan[ch] = dict(np.load(p))

    out_arrays, out_json = {}, []
    for r in recs:
        ch = r["_channel"]
        if ch not in per_chan:
            log("record %s: nothing detected on channel %d" % (r["_doc_id"], ch))
            continue
        ev = per_chan[ch]
        xf, yf = r["x_field"], r["y_field"]
        poly = r["polygon"]
        srcs = sources_of(r)
        want_unassigned = any(s is None for s in srcs)
        want_units = set(int(s) for s in srcs if s is not None)

        okey = owner_keys(ev["unit"], units_on.get(ch, set()))
        owner_ok = np.isin(okey, list(want_units)) if want_units \
            else np.zeros(len(okey), dtype=bool)
        if want_unassigned:
            owner_ok |= (okey == NOISE)
        in_poly = inside(ev[xf], ev[yf], poly)
        sel = owner_ok & in_poly

        t_s = ev["time"] / float(FS)
        b = np.clip((t_s // bin_s).astype(int), 0, n_bins - 1)
        n_in = np.bincount(b[sel], minlength=n_bins)
        n_src = np.bincount(b[owner_ok], minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(n_src > 0, n_in / np.maximum(n_src, 1).astype(float), np.nan)

        drawn_bins = np.array(sorted(drawn))
        drawn_bins = drawn_bins[drawn_bins < n_bins]
        f_drawn = np.nanmean(frac[drawn_bins]) if len(drawn_bins) else np.nan
        f_rest = np.nanmean(np.delete(frac, drawn_bins))
        ok = np.isfinite(frac)
        if ok.sum() >= 4:
            xs = np.where(ok)[0].astype(float)
            slope = np.polyfit(xs, frac[ok], 1)[0] * n_bins   # change across the session
        else:
            slope = np.nan

        # Relabels versus inserts -- and, inside the relabels, the ones taken out of a
        # unit Gil never saw on this panel, because they were drawn in the grey pile.
        is_assigned = ev["unit"] != NOISE
        relabel_mask = sel & is_assigned
        insert_mask = sel & ~is_assigned
        hidden_mask = relabel_mask & (okey == NOISE)
        stolen = {}
        for uu, cc in zip(*np.unique(ev["unit"][hidden_mask], return_counts=True)):
            stolen[int(uu)] = int(cc)
        # Map every selected event back to the Kilosort spike it matched. match_units
        # only recorded WHICH unit; the manifest needs WHICH SPIKE, so the same nearest-
        # within-half-a-millisecond search is repeated here keeping the index.
        j = np.searchsorted(spike_times, ev["time"][relabel_mask])
        best = np.full(relabel_mask.sum(), -1, dtype=np.int64)
        bestd = np.full(relabel_mask.sum(), MATCH_WINDOW + 1, dtype=np.int64)
        tt = ev["time"][relabel_mask]
        for off in (-2, -1, 0, 1):
            jj = np.clip(j + off, 0, len(spike_times) - 1)
            d = np.abs(spike_times[jj] - tt)
            take = (d <= MATCH_WINDOW) & (d < bestd)
            best[take] = jj[take]
            bestd[take] = d[take]
        good = best >= 0
        ks_idx = orig_index[best[good]]

        key = r["_doc_id"]
        out_arrays["relabel_" + key] = ks_idx.astype(np.int64)
        out_arrays["insert_" + key] = ev["time"][insert_mask].astype(np.int64)
        out_arrays["frac_" + key] = frac.astype(np.float32)
        out_arrays["nin_" + key] = n_in.astype(np.int64)
        out_arrays["nsrc_" + key] = n_src.astype(np.int64)

        tgt = r.get("target_unit")
        out_json.append({
            "doc_id": key,
            "scatter_unit": r["_scatter_unit"],
            "channel": ch,
            "sources": srcs,
            "target_unit": tgt,
            "x_field": xf, "y_field": yf,
            "drawn_dots": r.get("n_selected_dots"),
            "page_estimate_spikes": r.get("n_estimated"),
            "relabel_spikes": int(len(ks_idx)),
            "insert_events": int(insert_mask.sum()),
            "taken_from_offchannel_units": stolen,
            "source_events_whole_session": int(owner_ok.sum()),
            "frac_inside_drawn_windows": None if not np.isfinite(f_drawn) else round(float(f_drawn), 4),
            "frac_inside_rest_of_session": None if not np.isfinite(f_rest) else round(float(f_rest), 4),
            "drift_slope_over_session": None if not np.isfinite(slope) else round(float(slope), 4),
        })
        log("%-22s %s -> %s | relabel %6d (%d off-channel), insert %6d | "
            "in-window %.3f vs rest %.3f"
            % (key, srcs, tgt, len(ks_idx), int(hidden_mask.sum()),
               int(insert_mask.sum()),
               f_drawn if np.isfinite(f_drawn) else -1,
               f_rest if np.isfinite(f_rest) else -1))

    np.savez_compressed(out_path("reassignment_manifest_%s.npz" % sid), **out_arrays)
    doc = {
        "format": "reassignment_manifest/1",
        "session_id": sid,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "decision": ("whole-session application with a drift check, chosen by Gil "
                     "2026-11-19. The windows are where the region was DRAWN; the "
                     "region itself is a boundary in feature space and applies "
                     "everywhere."),
        "applied": False,
        "bin_seconds": bin_s,
        "n_bins": n_bins,
        "drawn_bins": sorted(int(b) for b in drawn),
        "scan": {"minutes_scanned": round(stop / FS / 60.0, 1),
                 "channels": channels,
                 "spike_clusters_file": os.path.join(ks, "spike_clusters.npy")},
        "records": out_json,
    }
    with open(out_path("reassignment_manifest_%s.json" % sid), "w") as f:
        json.dump(doc, f, indent=1)
    log("wrote the manifest -- nothing in the sort has been changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

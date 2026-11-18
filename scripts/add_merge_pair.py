# -*- coding: utf-8 -*-
"""
Adds one named pair to an existing merge_candidates file, by hand.

WHY THIS EXISTS
---------------
build_merge_candidates.py shortlists on geometry: peak channels within --max_dist_um
(60 um) of each other, nothing else gets in. That gate is right as a default -- across
the five sessions judged so far, 125 accepted merges and the farthest two were 52 um
apart -- but it is a hard edge, and the pad lattice puts the next rung at 62.1 um
(16 um across, 60 um down). A pair one rung out is invisible to the panel, and there
is no way to look at it without rebuilding the whole session at a wider gate.

So: name the pair, and it is computed and inserted. Every number is produced by the
same functions the sweep uses, on the per-unit payloads already in the file, so the
record is identical to what a rebuild with a wider gate would have written. Nothing
else in the file is recomputed and no verdict is touched -- decisions live in the
page's database, keyed by pair_id, and this only ever adds a pair_id.

The added pairs are listed in the document header under "manual_pairs" with their
distance, so the file never quietly implies the 60 um gate covered them.

Usage:
  python add_merge_pair.py <session_id> --pair 422_425 [--pair 86_93 ...]
                           [--ks <kilosort4 dir>] [--out <merge_candidates json>]

Both units of a pair must already be in the file (true whenever each has at least one
in-gate neighbour of its own). A unit with no in-gate neighbour at all is absent, has
no stored waveform, and needs the raw probe1.dat -- the script says so and stops
rather than guessing.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from striatal_agent.paths import outputs_dir  # noqa: E402

OUT_DIR = outputs_dir()


def load_builder():
    """Import build_merge_candidates as a module, for its metric functions.

    The point is that a hand-added pair is measured by exactly the same code as the
    3283 that came from the sweep. Copying the formulas here would be the one way to
    make a pair that looks comparable and is not.
    """
    path = os.path.join(HERE, "build_merge_candidates.py")
    spec = importlib.util.spec_from_file_location("bmc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pair_record(bmc, doc, a, b, sim, by_unit):
    """The 16 fields of a pair record, in the builder's own terms."""
    ua, ub = doc["units"][str(a)], doc["units"][str(b)]
    ta, tb = by_unit[a], by_unit[b]

    hist, edges = bmc.correlogram(ta, tb)
    cm = bmc.ccg_metrics(hist, edges)

    combined = np.sort(np.concatenate([ta, tb]))
    merged_refr = bmc.refractory_violations(combined)
    merged_acg, _ = bmc.correlogram(combined, combined, auto=True)

    dist = float(np.hypot(ua["x_um"] - ub["x_um"], ua["depth_um"] - ub["depth_um"]))

    # Waveform agreement on the channels the two units share, in microvolts -- the
    # stored uv block is already channels-first, so these are the builder's indices.
    wa, wb = ua["waveform"], ub["waveform"]
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
        amp_ratio = (round(min(amp_a, amp_b) / max(amp_a, amp_b), 4)
                     if max(amp_a, amp_b) else None)
    else:
        wf_corr, amp_ratio = None, None

    ra, rb = ua.get("rate_timeline"), ub.get("rate_timeline")
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

    s = float(sim[a, b]) if a < sim.shape[0] and b < sim.shape[1] else 0.0
    return {
        "pair_id": "%d_%d" % (a, b),
        "unit_a": a, "unit_b": b,
        "distance_um": round(dist, 1),
        "template_similarity": round(s, 4),
        "ccg_hist": hist.tolist(),
        "ccg_bin_ms": bmc.CCG_BIN_MS,
        "ccg_max_lag_ms": bmc.CCG_MAX_LAG_MS,
        "ccg": cm,
        "merged_acg_hist": merged_acg.tolist(),
        "merged_refractory": merged_refr,
        "refractory_a": ua["refractory"],
        "refractory_b": ub["refractory"],
        "waveform_corr": (round(wf_corr, 4) if wf_corr is not None else None),
        "amplitude_ratio": amp_ratio,
        "rate_corr": rate_corr,
        "active_overlap": overlap,
        # Why this pair is in a file whose gate says 60 um.
        "added_by_hand": True,
    }


def rank(p):
    """The builder's own ordering: the measurable CCG dip first."""
    dip = p["ccg"]["dip_ratio"]
    measurable = p["ccg"]["flank_per_bin"] >= 0.5
    if dip is not None and measurable:
        return (0, dip)
    return (1, -(p["template_similarity"] or 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    ap.add_argument("--pair", action="append", required=True,
                    help="unit ids as A_B; repeat for several pairs")
    ap.add_argument("--ks", default=None,
                    help="kilosort4 dir (default: the one recorded in the file)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--note", default="asked for by hand: outside the geometry gate")
    args = ap.parse_args()

    sid = args.session_id
    out = args.out or os.path.join(OUT_DIR, "merge_candidates_%s.json" % sid)
    if not os.path.exists(out):
        print("no such file: %s" % out)
        return 2

    t0 = time.time()
    with open(out, encoding="utf-8") as f:
        doc = json.load(f)
    print("loaded %s: %d pairs, %d units, gate %.0f um"
          % (os.path.basename(out), len(doc["pairs"]), len(doc["units"]),
             doc["max_dist_um"]))

    wanted = []
    for spec in args.pair:
        try:
            a, b = (int(x) for x in spec.split("_"))
        except ValueError:
            print("cannot read pair %r -- expected A_B" % spec)
            return 2
        a, b = min(a, b), max(a, b)
        missing = [u for u in (a, b) if str(u) not in doc["units"]]
        if missing:
            print("unit(s) %s are not in this file -- they have no neighbour inside "
                  "the gate, so no waveform was stored for them. That needs a rebuild "
                  "with a wider --max_dist_um, which reads the raw probe1.dat."
                  % ", ".join(str(m) for m in missing))
            return 2
        if any(p["pair_id"] == "%d_%d" % (a, b) for p in doc["pairs"]):
            print("pair %d_%d is already in the file -- nothing to do" % (a, b))
            continue
        wanted.append((a, b))
    if not wanted:
        return 0

    ks = args.ks or doc["kilosort_dir"]
    print("reading the clustering from %s" % ks)
    spike_rows = np.load(os.path.join(ks, "spike_times.npy")).ravel()
    clusters = np.load(os.path.join(ks, "spike_clusters.npy")).ravel()
    sim = np.load(os.path.join(ks, "similar_templates.npy"))
    # Same clock the sweep used: probe1.dat rows over the sample rate, no sync offset.
    t_all = spike_rows.astype(np.float64) / doc["fs_hz"]
    order = np.argsort(t_all)
    t_all, clusters = t_all[order], clusters[order]
    print("%d spikes in %.0fs" % (len(t_all), time.time() - t0))

    sha = doc.get("clustering", {}).get("spike_clusters_sha1")
    import hashlib
    # The file's bytes, header included -- the same thing the builder stamped, so the
    # two are comparable at all.
    with open(os.path.join(ks, "spike_clusters.npy"), "rb") as f:
        got = hashlib.sha1(f.read()).hexdigest()
    if sha and got != sha:
        print("spike_clusters.npy has changed since this file was built "
              "(%s now, %s then). Refusing: the pair would not be comparable."
              % (got[:12], sha[:12]))
        return 2
    print("spike_clusters sha1 matches the file: %s" % got[:12])

    bmc = load_builder()
    need = sorted({u for p in wanted for u in p})
    by_unit = {u: t_all[clusters == u] for u in need}

    added = []
    for a, b in wanted:
        rec = pair_record(bmc, doc, a, b, sim, by_unit)
        doc["pairs"].append(rec)
        added.append(rec)
        c = rec["ccg"]
        print("  %s  %.1f um  sim %.4f  wf r %s  dip %s (%d observed vs %.1f "
              "expected from the far flanks)"
              % (rec["pair_id"], rec["distance_um"], rec["template_similarity"],
                 rec["waveform_corr"], c["dip_ratio"], c["n_refractory"],
                 c["expected_refractory"]))

    doc["pairs"].sort(key=rank)
    doc["n_pairs_analysed"] = len(doc["pairs"])
    man = doc.setdefault("manual_pairs", [])
    for rec in added:
        man.append({
            "pair_id": rec["pair_id"],
            "distance_um": rec["distance_um"],
            "beyond_gate_um": round(rec["distance_um"] - doc["max_dist_um"], 1),
            "added": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": args.note,
        })
    doc["built_by"] = doc.get("built_by", "") + " + add_merge_pair.py"

    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    os.replace(tmp, out)
    print("wrote %s: %d pairs (%d added by hand), %.1f MiB, %.0fs total"
          % (os.path.basename(out), len(doc["pairs"]), len(man),
             os.path.getsize(out) / 1048576.0, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())

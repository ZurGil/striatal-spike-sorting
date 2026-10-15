"""
Sanity check for the review tool's per-unit trial selection: does dropping the marked
trials actually remove the holes from a unit's rasters?

Builds, per unit, cue-end-aligned and reward-aligned rasters + PSTHs side by side --
ALL trials on the left (with the dropped stretch shaded, so the hole is visible) and
KEPT trials only on the right. Nothing statistical; this is a look-at-it check.

Alignments follow the conventions already used for this project (see
track_choice_response_20260916_110311.py / left_right_cue_response_session.py):
  cue end   = `wait_Sin` state START  (stimulus ends, rat is free to choose)
  reward    = `water_L` / `water_R` state START (valve opens; verified in TASK_TIMELINE.md)
Only TrialCompleted==1 trials are drawn, which already means "the rat poked a side port
at all, regardless of reward outcome".

**Rasters are in chronological trial order, deliberately.** The earlier raster scripts
sort by decision time, which scatters a dead stretch across the whole panel and hides
exactly what we are looking for here. In trial order a dead stretch is a clean band.

Reads the kept windows from outputs/trial_selections_<session_id>.json -- a plain file,
NOT the review tool. Nothing here talks to the artifact database, so this runs whether
or not the review tool was ever opened; a unit missing from that file simply keeps every
trial, which is the documented default.

Usage: python response_sanity_check_selected_trials_session.py <session_id> [unit ...]
Any env with pandas/pyarrow/matplotlib (phy2_ky is fine).
"""
import json, os, sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"D:\Gil\spike_sorting_agent\outputs"
SYNCED = {"20260916_110311": r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\synced_20261002"}

WIN = (-1.5, 3.0)
BIN_S = 0.05
BINS = np.arange(WIN[0], WIN[1] + 1e-9, BIN_S)
CENTERS = (BINS[:-1] + BINS[1:]) / 2


def load_selection(session_id, unit_id):
    """Kept windows for one unit, in absolute Trodes seconds. [] means keep everything."""
    path = os.path.join(OUT, "trial_selections_%s.json" % session_id)
    if not os.path.exists(path):
        print("  no selection file at %s -- keeping every trial" % path)
        return []
    with open(path) as f:
        doc = json.load(f)
    sel = doc.get("selections", {}).get(str(unit_id))
    return list(sel.get("ranges", [])) if sel else []


def kept_mask(trial_start_trodes, ranges):
    if not ranges:
        return np.ones(len(trial_start_trodes), dtype=bool)
    m = np.zeros(len(trial_start_trodes), dtype=bool)
    for r in ranges:
        m |= (trial_start_trodes >= r["t0"]) & (trial_start_trodes <= r["t1"])
    return m


def raster_psth(ax_r, ax_p, align_t, spikes_by_trial, order, kept, title, shade):
    """One raster + its PSTH. `order` is the trial order to draw; `kept` is a bool per
    entry of `order` used only to colour the rows when the panel shows everything."""
    counts = np.zeros(len(BINS) - 1)
    n_drawn = 0
    for row, tid in enumerate(order):
        if tid not in align_t:
            continue
        rel = spikes_by_trial.get(tid)
        if rel is None:
            continue
        rel = rel - align_t[tid]
        rel = rel[(rel >= WIN[0]) & (rel <= WIN[1])]
        if len(rel):
            ax_r.plot(rel, np.full(len(rel), row), "|", ms=1.6, mew=0.5,
                      color=("#1f6f6a" if kept[row] else "#b03030"), alpha=0.85)
            counts += np.histogram(rel, BINS)[0]
        n_drawn += 1
    if shade:
        bad = np.nonzero(~kept)[0]
        if len(bad):
            # contiguous runs of dropped trials, drawn as bands so the hole is obvious
            edges = np.nonzero(np.diff(bad) != 1)[0]
            for s, e in zip(np.r_[0, edges + 1], np.r_[edges, len(bad) - 1]):
                ax_r.axhspan(bad[s] - 0.5, bad[e] + 0.5, color="#b03030", alpha=0.10, lw=0)
    ax_r.axvline(0, color="k", lw=0.8, ls="--")
    ax_r.set_xlim(*WIN); ax_r.set_ylim(-0.5, max(len(order), 1) - 0.5)
    ax_r.set_title(title, fontsize=9)
    ax_r.set_ylabel("trial (chronological)", fontsize=8)
    ax_r.tick_params(labelsize=7, labelbottom=False)

    rate = counts / max(n_drawn, 1) / BIN_S
    ax_p.fill_between(CENTERS, rate, color="#1f6f6a", alpha=0.25)
    ax_p.plot(CENTERS, rate, color="#1f6f6a", lw=1.2)
    ax_p.axvline(0, color="k", lw=0.8, ls="--")
    ax_p.set_xlim(*WIN); ax_p.set_ylim(bottom=0)
    ax_p.set_xlabel("s from alignment", fontsize=8)
    ax_p.set_ylabel("Hz", fontsize=8)
    ax_p.tick_params(labelsize=7)
    return n_drawn


def main():
    session_id = sys.argv[1]
    units = [int(u) for u in sys.argv[2:]] or [163, 38]
    synced = SYNCED[session_id]

    trials = pd.read_parquet(os.path.join(synced, "trials.parquet"))
    se = pd.read_parquet(os.path.join(synced, "state_events.parquet"))
    print("loaded %d trials, %d state events" % (len(trials), len(se)))

    first = (se.sort_values(["trial_id", "start_time_in_trial"])
               .drop_duplicates(["trial_id", "state_name"], keep="first"))
    cue_end = first[first.state_name == "wait_Sin"].set_index("trial_id")["start_time_in_trial"]
    rew = first[first.state_name.isin(["water_L", "water_R"])]
    rew = rew.sort_values(["trial_id", "start_time_in_trial"]).drop_duplicates("trial_id", keep="first")
    reward_t = rew.set_index("trial_id")["start_time_in_trial"]
    print("alignments available: cue end on %d trials, reward on %d trials"
          % (len(cue_end), len(reward_t)))

    completed = trials[trials.TrialCompleted == 1].copy().sort_values("trial_id")
    tstart = completed.set_index("trial_id")["trial_start_trodes"]

    for uid in units:
        sp = pd.read_parquet(os.path.join(synced, "spikes.parquet"),
                             columns=["unit_id", "trial_id", "spike_time_in_trial"],
                             filters=[("unit_id", "==", uid)])
        by_trial = {t: g["spike_time_in_trial"].to_numpy()
                    for t, g in sp.groupby("trial_id")}
        ranges = load_selection(session_id, uid)
        order = completed.trial_id.to_numpy()
        kept = kept_mask(tstart.loc[order].to_numpy(), ranges)
        print("\nunit %d: %d spikes in %d completed trials | kept windows %s -> %d/%d trials kept"
              % (uid, len(sp), len(order), ranges or "none (all trials)", kept.sum(), len(order)))

        fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 3, 1], "hspace": 0.22,
                                              "wspace": 0.22})
        kept_order = order[kept]
        kept_all = np.ones(len(kept_order), dtype=bool)
        for col, (ordr, kp, shade, tag) in enumerate([
                (order, kept, True, "ALL completed trials"),
                (kept_order, kept_all, False, "KEPT trials only")]):
            n1 = raster_psth(axes[0][col], axes[1][col], cue_end.to_dict(), by_trial, ordr, kp,
                             "%s\naligned to CUE END (wait_Sin start)" % tag, shade)
            n2 = raster_psth(axes[2][col], axes[3][col], reward_t.to_dict(), by_trial, ordr, kp,
                             "aligned to REWARD (water_L/R start)", shade)
            print("   %-20s cue-end panel %d trials, reward panel %d trials" % (tag, n1, n2))
        keep_txt = ", ".join("%.0f-%.0fs" % (r["t0"], r["t1"]) for r in ranges) or "nothing marked"
        fig.suptitle("unit %d  ·  session %s  ·  kept windows: %s  ·  %d of %d completed trials kept"
                     % (uid, session_id, keep_txt, kept.sum(), len(order)), fontsize=11)
        png = os.path.join(OUT, "response_sanity_unit%d_%s.png" % (uid, session_id))
        fig.savefig(png, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("   wrote %s" % png)


if __name__ == "__main__":
    main()

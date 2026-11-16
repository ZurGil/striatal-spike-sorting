# -*- coding: utf-8 -*-
"""Do the good units respond to the cue and to reward, and is the alignment right?

    python scripts/plot_event_responses.py 20261112_082028 \
        --synced Z:\\Gil\\Popper\\20261112_082028.rec\\20261112_082028.kilosort\\synced

WHAT THIS IS FOR
----------------
Two questions at once, and the second is the real one. First, do the units Gil
called good carry trial-locked responses at all. Second -- and this is why it is
worth plotting rather than trusting -- if the ephys and the behaviour were
misaligned, a response locked to reward delivery would smear out or sit at the
wrong latency. A sharp, consistent response at t = 0 is evidence the sync is
right that no QC number can give you.

THE CLOCK QUESTION, WHICH THIS DELIBERATELY SIDESTEPS
-----------------------------------------------------
`spikes.parquet` carries `spike_time_in_trial` and `state_events.parquet` carries
`start_time_in_trial`, both in the SAME per-trial frame, produced by the
pipeline's own local-interpolation conversion. Subtracting one from the other
needs no clock arithmetic here at all. Doing it the other way -- taking a Bpod
offset and adding it to `trial_start_trodes` -- would reintroduce the clock drift
the pipeline measured (slope 1.0016, so 16 ms of error 10 s into a trial) and
blame the sync for an error made in this script.

WHICH TRIALS
------------
Only trials where the rat actually did the task: `TrialCompleted`, with no broken
fixation and no early withdrawal, and a choice recorded. Reward-aligned panels
additionally take only rewarded trials, since on an unrewarded one no valve ever
opened and there is no event to align to.

THE THREE EVENTS (from sync_pipeline/TASK_TIMELINE.md, not guessed)
    cue start   onset of `stimulus_delivery_min` -- the instant the hardware
                confirmed the stimulus is playing
    cue end     end of `stimulus_delivery`, falling back to the end of
                `stimulus_delivery_min` when the remainder state has zero
                duration (it does on these sessions: AuditoryStimulusTime equals
                MinSampleAud, so the cue is 0.35 s long and cue end is 0.35 s
                after cue start -- the two panels are NOT independent, and saying
                so is more useful than drawing them as if they were)
    reward      onset of `water_L` / `water_R`, the valve opening

OUTPUT
    outputs/responses_<session>.png          rasters and PSTHs, units x events
    outputs/responses_<session>.csv          per unit and event: baseline rate,
                                             response rate, change, p, latency
"""
from __future__ import print_function

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")

BIN_S = 0.020
WIN = (-1.0, 1.5)
BASELINE = (-0.5, -0.1)      # before the event
RESPONSE = (0.0, 0.4)        # after it


def trial_mask(trials):
    """Trials the rat actually completed. Everything else has no event to align to."""
    f = lambda c: pd.Series(trials[c]).astype(float) if c in trials.columns else None
    ok = np.ones(len(trials), bool)
    if "sync_valid" in trials.columns:
        ok &= trials["sync_valid"].astype(bool).values
    if "TrialCompleted" in trials.columns:
        ok &= trials["TrialCompleted"].astype(bool).values
    for col in ("FixBroke", "EarlyWithdrawal"):
        v = f(col)
        if v is not None:
            ok &= ~(v.values == 1)
    ch = f("ChoiceLeft")
    if ch is not None:
        ok &= ~np.isnan(ch.values)
    return ok


def event_times(states, name, which="start", occurrence=0):
    """occurrence is 0-based in state_events.parquet -- the first visit to a state."""
    s = states[(states.state_name == name) & (states.occurrence == occurrence)]
    col = "start_time_in_trial" if which == "start" else "end_time_in_trial"
    return dict(zip(s.trial_id.astype(int), s[col].astype(float)))


def build_events(states):
    cue_on = event_times(states, "stimulus_delivery_min", "start")
    cue_off = event_times(states, "stimulus_delivery", "end")
    cue_off_fallback = event_times(states, "stimulus_delivery_min", "end")
    for t, v in cue_off_fallback.items():
        cue_off.setdefault(t, v)
    reward = {}
    for side in ("water_L", "water_R"):
        reward.update(event_times(states, side, "start"))
    return {"cue start": cue_on, "cue end": cue_off, "reward": reward}


def psth(spike_times, event_by_trial, trial_ids, bins):
    """Per-trial binned counts, trials x bins, for the trials that have the event."""
    rows = []
    for t in trial_ids:
        e = event_by_trial.get(t)
        if e is None or not np.isfinite(e):
            continue
        st = spike_times.get(t)
        if st is None:
            rows.append(np.zeros(len(bins) - 1))
            continue
        rows.append(np.histogram(np.asarray(st) - e, bins=bins)[0])
    return np.asarray(rows) if rows else np.zeros((0, len(bins) - 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id")
    ap.add_argument("--synced", required=True)
    ap.add_argument("--n_units", type=int, default=6,
                    help="how many good units to draw (the most-spiking ones)")
    ap.add_argument("--label", default="good",
                    help="quality_label to take units from")
    ap.add_argument("--title_extra", default="")
    ap.add_argument("--units", default=None,
                    help="comma-separated unit ids to draw instead of the busiest ones")
    args = ap.parse_args()

    sid, sdir = args.session_id, args.synced
    trials = pd.read_parquet(os.path.join(sdir, "trials.parquet"))
    units = pd.read_parquet(os.path.join(sdir, "units.parquet"))
    states = pd.read_parquet(os.path.join(sdir, "state_events.parquet"))

    keep = trial_mask(trials)
    tid_all = trials.loc[keep, "trial_id"].astype(int).values
    rewarded = pd.Series(trials["Rewarded"]).astype(float).values == 1
    tid_rew = trials.loc[keep & rewarded, "trial_id"].astype(int).values
    print("%s: %d trials, %d completed, %d of those rewarded"
          % (sid, len(trials), len(tid_all), len(tid_rew)))

    events = build_events(states)
    for name, d in events.items():
        print("   %-10s present on %d trials" % (name, len(d)))

    good = units[units.quality_label.astype(str) == args.label].copy()
    print("   %d units labelled %s" % (len(good), args.label))
    if not len(good):
        raise SystemExit("no units with quality_label == %r" % args.label)

    # Read only the units we draw: spikes.parquet is 44 M rows and we need a handful.
    import pyarrow.parquet as pq

    # Pick the units with the most spikes. units.parquet does not carry a count, so
    # count them -- one column of spikes.parquet, which is cheap, rather than guessing
    # from unit id order and drawing six nearly silent units (which is what happened
    # the first time this ran).
    if args.units:
        unit_ids = [int(x) for x in args.units.split(",")]
    else:
        ids = pq.read_table(os.path.join(sdir, "spikes.parquet"),
                            columns=["unit_id"]).column("unit_id").to_pandas()
        counts = ids.value_counts()
        goodset = set(int(u) for u in good.unit_id.values)
        ranked = [int(u) for u in counts.index if int(u) in goodset]
        unit_ids = ranked[:args.n_units]
        print("   unit spike counts (rows, trial-duplicated): %s"
              % ", ".join("%d:%d" % (u, counts[u]) for u in unit_ids))
    print("   drawing units %s" % unit_ids)

    tbl = pq.read_table(os.path.join(sdir, "spikes.parquet"),
                        columns=["unit_id", "trial_id", "spike_time_in_trial"],
                        filters=[("unit_id", "in", unit_ids)])
    sp = tbl.to_pandas()
    print("   %d spike rows for those units" % len(sp))

    bins = np.arange(WIN[0], WIN[1] + BIN_S, BIN_S)
    centres = bins[:-1] + BIN_S / 2.0
    base_m = (centres >= BASELINE[0]) & (centres < BASELINE[1])
    resp_m = (centres >= RESPONSE[0]) & (centres < RESPONSE[1])

    ev_names = ["cue start", "cue end", "reward"]
    fig = plt.figure(figsize=(13.5, 2.5 * len(unit_ids) + 1.2))
    gs = GridSpec(len(unit_ids) * 2, len(ev_names), figure=fig,
                  height_ratios=[1, 2] * len(unit_ids), hspace=0.28, wspace=0.22)

    rows_csv = []
    from scipy import stats as sstats
    for i, uid in enumerate(unit_ids):
        st_by_trial = (sp[sp.unit_id == uid]
                       .groupby("trial_id")["spike_time_in_trial"].apply(list).to_dict())
        st_by_trial = {int(k): v for k, v in st_by_trial.items()}
        for j, ev in enumerate(ev_names):
            tids = tid_rew if ev == "reward" else tid_all
            counts = psth(st_by_trial, events[ev], tids, bins)
            if not len(counts):
                continue
            rate = counts.sum(0) / (len(counts) * BIN_S)
            sem = counts.std(0) / BIN_S / np.sqrt(len(counts))
            per_trial_base = counts[:, base_m].sum(1) / (base_m.sum() * BIN_S)
            per_trial_resp = counts[:, resp_m].sum(1) / (resp_m.sum() * BIN_S)
            t, p = sstats.ttest_rel(per_trial_resp, per_trial_base)

            # Trials x time as an image rather than a dot raster. At 40-60 sp/s a dot
            # raster is a solid black block and shows nothing; the image keeps every
            # trial visible and lets a response that is present on most trials be told
            # apart from one carried by a handful.
            ax_r = fig.add_subplot(gs[i * 2, j])
            ax_r.imshow(counts, aspect="auto", origin="lower", cmap="Greys",
                        interpolation="nearest",
                        extent=(WIN[0], WIN[1], 0, len(counts)),
                        vmin=0, vmax=max(1, np.percentile(counts, 99)))
            ax_r.axvline(0, color="#c0392b", lw=1)
            ax_r.set_xticks([]); ax_r.set_yticks([])
            ax_r.set_ylabel("%d trials" % len(counts), fontsize=7, labelpad=2)
            if i == 0:
                ax_r.set_title("%s" % ev, fontsize=11, pad=6)

            ax = fig.add_subplot(gs[i * 2 + 1, j])
            ax.fill_between(centres, rate - sem, rate + sem, color="#1f77b4", alpha=0.25,
                            linewidth=0)
            ax.plot(centres, rate, color="#1f77b4", lw=1.4)
            ax.axvline(0, color="#c0392b", lw=1)
            ax.axvspan(BASELINE[0], BASELINE[1], color="#999", alpha=0.10, linewidth=0)
            ax.set_xlim(WIN)
            ax.margins(y=0.12)
            if j == 0:
                ax.set_ylabel("unit %d\nsp/s" % uid, fontsize=9)
            if i == len(unit_ids) - 1:
                ax.set_xlabel("time from %s (s)" % ev, fontsize=9)
            ax.tick_params(labelsize=8)
            star = "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else ""))
            ax.text(0.98, 0.92, "%.1f\u2192%.1f sp/s %s"
                    % (per_trial_base.mean(), per_trial_resp.mean(), star),
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    color="#333")

            # latency: first bin after 0 that is 3 sd above baseline and stays up
            thr = per_trial_base.mean() + 3 * (rate[base_m].std() or 1e-9)
            after = np.nonzero((centres >= 0) & (rate > thr))[0]
            lat = centres[after[0]] if after.size else np.nan
            rows_csv.append({
                "session": sid, "unit_id": uid, "event": ev, "n_trials": len(counts),
                "baseline_sp_s": round(float(per_trial_base.mean()), 3),
                "response_sp_s": round(float(per_trial_resp.mean()), 3),
                "change_sp_s": round(float(per_trial_resp.mean() - per_trial_base.mean()), 3),
                "t": round(float(t), 2), "p": float(p),
                "latency_s": None if np.isnan(lat) else round(float(lat), 3),
            })

    fig.suptitle("%s \u2014 %s units, trial-locked responses%s\n"
                 "completed trials only (no broken fixation, no early withdrawal, a "
                 "choice made); reward panels are rewarded trials only"
                 % (sid, args.label, args.title_extra),
                 fontsize=12, y=0.995)
    fig.subplots_adjust(top=0.945, bottom=0.045, left=0.075, right=0.985)
    png = os.path.join(OUT, "responses_%s.png" % sid)
    fig.savefig(png, dpi=130)
    plt.close(fig)

    csv = os.path.join(OUT, "responses_%s.csv" % sid)
    pd.DataFrame(rows_csv).to_csv(csv, index=False)
    print("   wrote %s" % png)
    print("   wrote %s" % csv)

    d = pd.DataFrame(rows_csv)
    if not len(d):
        raise SystemExit("no event aligned to any trial -- check the state names against "
                         "this session's own state_events.parquet before trusting the "
                         "figure, which will be empty")
    for ev in ev_names:
        sub = d[d.event == ev]
        if len(sub):
            print("   %-10s %d of %d units modulated at p<0.05, median change %+.1f sp/s"
                  % (ev, int((sub.p < 0.05).sum()), len(sub), sub.change_sp_s.median()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

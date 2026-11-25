# -*- coding: utf-8 -*-
"""Trial-locked responses of the good units: every one screened, the strongest drawn.

    python scripts/plot_event_responses.py 20261112_082028 \
        --synced Z:\\Gil\\Popper\\without_error_catch_trials_before_20261117\\20261112_082028.rec\\20261112_082028.kilosort\\synced

WHAT THIS IS FOR
----------------
Two questions. Do the units Gil called good carry trial-locked responses, and --
the one that matters more -- is the ephys aligned to the behaviour? A response
locked to the valve opening smears out or lands at the wrong latency if the sync
is wrong, so a sharp peak at t = 0 is evidence no QC number can give you.

HOW THE FIRST VERSION OF THIS GOT IT WRONG, AND WHY IT IS WORTH WRITING DOWN
----------------------------------------------------------------------------
It drew the eight units with the most spikes and binned at 20 ms with no
smoothing. Busiest is not most responsive: those were high-rate cells with modest
modulation, and at 20 ms raw the trace is noisy enough to bury what modulation
there was. Gil read the figure as weak and he was right about the figure -- the
same six units as the validated October analysis, pulled from today's dataset,
reproduce it peak for peak to the decimal, so neither the sort nor the sync had
moved. This version therefore does what that analysis did:

  * every good unit is screened, and the drawn ones are chosen BY RESPONSIVENESS
  * 50 ms bins and Gaussian smoothing (sigma = 2 bins), as in the validated figure
  * a population panel, so the claim rests on all the units rather than six

THE CLOCK QUESTION, WHICH THIS DELIBERATELY SIDESTEPS
-----------------------------------------------------
`spikes.parquet` carries `spike_time_in_trial` and `state_events.parquet` carries
`start_time_in_trial`, both in the SAME per-trial frame, from the pipeline's own
local-interpolation conversion. Subtracting one from the other needs no clock
arithmetic here. Taking a Bpod offset and adding it to `trial_start_trodes`
instead would reintroduce the drift the pipeline measured -- 16 ms of it 10 s into
a trial -- and then blame the sync for an error made in this script.

WHICH TRIALS
------------
Only trials the rat actually completed: `TrialCompleted`, no broken fixation, no
early withdrawal, a choice recorded. Reward panels additionally take only rewarded
trials, because on an unrewarded one no valve opened and there is no event.

THE THREE EVENTS (from sync_pipeline/TASK_TIMELINE.md, not guessed)
    cue start   onset of `stimulus_delivery_min`, the instant playback is confirmed
    cue end     end of `stimulus_delivery`, falling back to the end of
                `stimulus_delivery_min`. On these sessions AuditoryStimulusTime
                equals MinSampleAud, so the cue is 0.35 s long and cue end is
                always 0.35 s after cue start -- the same data shifted, not two
                independent events.
    reward      onset of `water_L` / `water_R`, the valve opening

OUTPUT
    outputs/responses_<session>_<event>.png     six most responsive units each
    outputs/responses_<session>_population.png  all good units, z-scored, sorted
    outputs/responses_<session>.csv             every good unit, every event
"""
from __future__ import print_function

import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats as sstats
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")

BIN_S = 0.05
SMOOTH_BINS = 2.0
WIN = (-1.0, 2.0)
BASELINE = (-1.0, -0.5)
RESPONSE = (0.0, 0.5)
EVENTS = ("cue start", "cue end", "reward")


def trial_mask(trials):
    ok = np.ones(len(trials), bool)
    if "sync_valid" in trials.columns:
        ok &= trials["sync_valid"].astype(bool).values
    if "TrialCompleted" in trials.columns:
        ok &= trials["TrialCompleted"].astype(bool).values
    for col in ("FixBroke", "EarlyWithdrawal"):
        if col in trials.columns:
            ok &= ~(pd.Series(trials[col]).astype(float).values == 1)
    if "ChoiceLeft" in trials.columns:
        ok &= ~np.isnan(pd.Series(trials["ChoiceLeft"]).astype(float).values)
    return ok


def first_state(states, name, which="start"):
    s = (states[states.state_name == name]
         .sort_values(["trial_id", "start_time_in_trial"])
         .drop_duplicates("trial_id", keep="first"))
    col = "start_time_in_trial" if which == "start" else "end_time_in_trial"
    return dict(zip(s.trial_id.astype(int), s[col].astype(float)))


def build_events(states):
    cue_on = first_state(states, "stimulus_delivery_min", "start")
    cue_off = first_state(states, "stimulus_delivery", "end")
    for t, v in first_state(states, "stimulus_delivery_min", "end").items():
        cue_off.setdefault(t, v)
    reward = {}
    for side in ("water_L", "water_R"):
        reward.update(first_state(states, side, "start"))
    return {"cue start": cue_on, "cue end": cue_off, "reward": reward}


def accumulate(path, unit_ids, events, trial_sets, bins, base_m, resp_m):
    """One pass over spikes.parquet, building every unit's PSTH for every event.

    Per unit and event it keeps the summed histogram (for the mean rate) and the
    per-trial baseline and response counts (for the paired test). Whole
    trials x bins matrices for 400 units would not fit in memory; these two do,
    and they are all that is needed.
    """
    n_b = len(bins) - 1
    step = bins[1] - bins[0]
    uidx = {u: i for i, u in enumerate(unit_ids)}
    acc = {}
    for ev in EVENTS:
        tids = trial_sets[ev]
        acc[ev] = {"tidx": {t: i for i, t in enumerate(tids)},
                   "hist": np.zeros((len(unit_ids), n_b)),
                   "base": np.zeros((len(unit_ids), max(1, len(tids)))),
                   "resp": np.zeros((len(unit_ids), max(1, len(tids))))}

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=2000000,
                                 columns=["unit_id", "trial_id", "spike_time_in_trial"]):
        d = batch.to_pandas()
        d = d[d.unit_id.isin(uidx)]
        if not len(d):
            continue
        u = d.unit_id.map(uidx).to_numpy()
        tr = d.trial_id.to_numpy().astype(int)
        st = d.spike_time_in_trial.to_numpy()
        for ev in EVENTS:
            a = acc[ev]
            et = np.array([events[ev].get(t, np.nan) for t in tr])
            ti = np.array([a["tidx"].get(t, -1) for t in tr])
            keep = np.isfinite(et) & (ti >= 0)
            if not keep.any():
                continue
            b = np.floor((st[keep] - et[keep] - bins[0]) / step).astype(int)
            uu, tt = u[keep], ti[keep]
            inwin = (b >= 0) & (b < n_b)
            bb = b[inwin]
            np.add.at(a["hist"], (uu[inwin], bb), 1)
            inb = base_m[bb]
            inr = resp_m[bb]
            np.add.at(a["base"], (uu[inwin][inb], tt[inwin][inb]), 1)
            np.add.at(a["resp"], (uu[inwin][inr], tt[inwin][inr]), 1)
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id")
    ap.add_argument("--synced", required=True)
    ap.add_argument("--n_units", type=int, default=6)
    ap.add_argument("--label", default="good")
    args = ap.parse_args()

    sid, sdir = args.session_id, args.synced
    trials = pd.read_parquet(os.path.join(sdir, "trials.parquet"))
    units = pd.read_parquet(os.path.join(sdir, "units.parquet"))
    states = pd.read_parquet(os.path.join(sdir, "state_events.parquet"))

    keep = trial_mask(trials)
    rewarded = pd.Series(trials["Rewarded"]).astype(float).values == 1
    tid_all = trials.loc[keep, "trial_id"].astype(int).tolist()
    tid_rew = trials.loc[keep & rewarded, "trial_id"].astype(int).tolist()
    events = build_events(states)
    trial_sets = {"cue start": [t for t in tid_all if t in events["cue start"]],
                  "cue end": [t for t in tid_all if t in events["cue end"]],
                  "reward": [t for t in tid_rew if t in events["reward"]]}
    print("%s: %d trials, %d completed, %d rewarded"
          % (sid, len(trials), len(tid_all), len(tid_rew)))
    for ev in EVENTS:
        print("   %-10s %d trials" % (ev, len(trial_sets[ev])))

    good = units[units.quality_label.astype(str) == args.label]
    unit_ids = [int(u) for u in good.unit_id.values]
    print("   screening all %d %s units" % (len(unit_ids), args.label))

    bins = np.arange(WIN[0], WIN[1] + BIN_S / 2, BIN_S)
    centres = bins[:-1] + BIN_S / 2.0
    base_m = (centres >= BASELINE[0]) & (centres < BASELINE[1])
    resp_m = (centres >= RESPONSE[0]) & (centres < RESPONSE[1])
    acc = accumulate(os.path.join(sdir, "spikes.parquet"), unit_ids, events,
                     trial_sets, bins, base_m, resp_m)

    rows, rates, tstats, raw_base = [], {}, {}, {}
    for ev in EVENTS:
        a = acc[ev]
        n_tr = max(1, len(trial_sets[ev]))
        rate_raw = a["hist"] / (n_tr * BIN_S)
        rate_s = gaussian_filter1d(rate_raw, sigma=SMOOTH_BINS, axis=1, mode="nearest")
        rates[ev] = rate_s
        # Keep the UNSMOOTHED baseline spread for the z-score. Smoothing removes most
        # of the bin-to-bin variance, so a sigma taken from the smoothed trace is far
        # too small and every unit saturates the colour scale -- which is what the
        # first population panel did.
        raw_base[ev] = rate_raw[:, base_m]
        b = a["base"] / (base_m.sum() * BIN_S)
        r = a["resp"] / (resp_m.sum() * BIN_S)
        t, p = sstats.ttest_rel(r, b, axis=1)
        tstats[ev] = np.nan_to_num(t)
        for i, uid in enumerate(unit_ids):
            pk = int(np.argmax(np.abs(rate_s[i] - b[i].mean())))
            rows.append({"session": sid, "unit_id": uid, "event": ev, "n_trials": n_tr,
                         "baseline_sp_s": round(float(b[i].mean()), 3),
                         "response_sp_s": round(float(r[i].mean()), 3),
                         "change_sp_s": round(float(r[i].mean() - b[i].mean()), 3),
                         "t": round(float(np.nan_to_num(t[i])), 2),
                         "p": float(np.nan_to_num(p[i], nan=1.0)),
                         "peak_sp_s": round(float(rate_s[i].max()), 2),
                         "peak_time_s": round(float(centres[pk]), 3)})
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "responses_%s.csv" % sid), index=False)

    for ev in EVENTS:
        order = list(np.argsort(-np.abs(tstats[ev]))[:args.n_units])
        tids = trial_sets[ev]
        draw_ids = [unit_ids[i] for i in order]
        sp = pq.read_table(os.path.join(sdir, "spikes.parquet"),
                           columns=["unit_id", "trial_id", "spike_time_in_trial"],
                           filters=[("unit_id", "in", draw_ids)]).to_pandas()
        # Trials ordered by when the event fell within the trial, as in the validated
        # figure: a real ordering shows a response that tracks the trial's own timing,
        # which trial number cannot.
        tids_sorted = [tids[i] for i in np.argsort([events[ev][t] for t in tids])]

        fig, axes = plt.subplots(2, len(order), figsize=(3.0 * len(order), 5.6),
                                 sharex=True, squeeze=False,
                                 gridspec_kw={"height_ratios": [2, 1]})
        for col, i in enumerate(order):
            uid = unit_ids[i]
            ax_r, ax_p = axes[0, col], axes[1, col]
            bt = sp[sp.unit_id == uid].groupby("trial_id")["spike_time_in_trial"]
            for row_i, tid in enumerate(tids_sorted):
                if tid in bt.groups:
                    st = bt.get_group(tid).to_numpy() - events[ev][tid]
                    st = st[(st >= WIN[0]) & (st <= WIN[1])]
                    if st.size:
                        ax_r.scatter(st, np.full(st.size, row_i), s=2, c="#1f4e8c",
                                     marker="|", linewidths=0.5)
            ax_r.axvline(0, color="k", lw=0.9, ls="--")
            ax_r.set_xlim(WIN)
            ax_r.set_ylim(-1, len(tids_sorted))
            ax_r.set_title("unit %d" % uid, fontsize=10)
            if col == 0:
                ax_r.set_ylabel("trial, sorted by event time", fontsize=8)
            ax_p.plot(centres, rates[ev][i], color="#333", lw=1.7)
            ax_p.axvline(0, color="k", lw=0.9, ls="--")
            ax_p.set_xlabel("time from %s (s)" % ev, fontsize=9)
            if col == 0:
                ax_p.set_ylabel("firing rate (Hz)", fontsize=9)
            s = d[(d.event == ev) & (d.unit_id == uid)].iloc[0]
            ax_p.text(0.97, 0.93, "%.1f\u2192%.1f Hz, p=%.1g"
                      % (s.baseline_sp_s, s.response_sp_s, s.p),
                      transform=ax_p.transAxes, ha="right", va="top", fontsize=7.5,
                      color="#444")
        n_sig = int((d[d.event == ev].p < 0.05).sum())
        fig.suptitle("%s \u2014 the six most %s-responsive of %d %s units "
                     "(%d modulated at p<0.05); completed trials%s"
                     % (sid, ev, len(unit_ids), args.label, n_sig,
                        ", rewarded only" if ev == "reward" else ""), fontsize=11)
        fig.tight_layout()
        png = os.path.join(OUT, "responses_%s_%s.png" % (sid, ev.replace(" ", "_")))
        fig.savefig(png, dpi=130)
        plt.close(fig)
        print("   wrote %s" % png)

    fig, axes = plt.subplots(1, len(EVENTS), figsize=(4.4 * len(EVENTS), 5.2),
                             squeeze=False)
    for j, ev in enumerate(EVENTS):
        r = rates[ev]
        base = raw_base[ev]
        mu, sd = base.mean(1, keepdims=True), base.std(1, keepdims=True)
        z = (r - mu) / np.where(sd > 0, sd, np.inf)
        peak = np.array([centres[np.argmax(np.abs(zz))] for zz in z])
        ax = axes[0, j]
        im = ax.imshow(z[np.argsort(peak)], aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-4, vmax=4, extent=(WIN[0], WIN[1], 0, len(z)),
                       interpolation="nearest")
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel("time from %s (s)" % ev, fontsize=9)
        if j == 0:
            ax.set_ylabel("good units, sorted by time of peak", fontsize=9)
        n_sig = int((d[d.event == ev].p < 0.05).sum())
        ax.set_title("%s \u2014 %d of %d modulated" % (ev, n_sig, len(unit_ids)),
                     fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8, label="z vs own baseline")
    fig.suptitle("%s \u2014 every %s unit, z-scored against its own pre-event baseline"
                 % (sid, args.label), fontsize=12)
    fig.tight_layout()
    png = os.path.join(OUT, "responses_%s_population.png" % sid)
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print("   wrote %s" % png)

    for ev in EVENTS:
        sub = d[d.event == ev]
        i = sub.change_sp_s.abs().idxmax()
        print("   %-10s %d of %d good units modulated at p<0.05 | strongest %+.1f Hz "
              "(unit %d)" % (ev, int((sub.p < 0.05).sum()), len(sub),
                             sub.loc[i, "change_sp_s"], int(sub.loc[i, "unit_id"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

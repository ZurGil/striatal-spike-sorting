"""
Final visualization step: raw raster + raw (unsmoothed) spike-rate histogram
for one independently-detected track's synthetic unit_id, split by left vs
right choice. No statistics, no smoothing -- this is the plot meant to be
judged by eye.

Reads the trial-aligned spikes produced by align_tracks_to_behavior_session.py
/ run_sync_and_track_alignment_session.py (independent_track_spikes_<session>_ch<chan>.parquet)
and the standard synced/ trials.parquet + state_events.parquet.

============================================================================
FIXES vs. the original version (all three review findings addressed):
============================================================================

FIX 1 (the most likely explanation for "no obvious difference by eye" on a
large, sustained response): the original figure used
    plt.subplots(2, 2, figsize=(11, 8), sharex=True)
-- no `sharey`. Matplotlib then auto-scales the LEFT and RIGHT rate panels
INDEPENDENTLY, each to its own peak. A real, large L-vs-R firing-rate
difference (e.g. 30Hz peak vs 6Hz peak) can render as "two similarly-sized
bumps" side by side, because each panel is stretched to fill itself. Fixed
here two ways: (a) `sharey='row'` on the rate row so the two panels are
directly comparable, AND (b) a new third row that overlays both curves on
ONE axes with a shared scale by construction -- this is the cleanest,
least-ambiguous version of "is left clearly bigger than right in this
window," and doesn't depend on getting subplot scaling right at all.

FIX 2 (raster/histogram alignment): the original always plotted spike times
relative to `cue_end` (stimulus end / wait_Sin start), never relative to
the actual poke-in instant, even though poke_in was computed and used only
for trial-sorting and a marker dot. Since decision time (MT) varies trial
to trial, a poke-locked response would appear as a diagonal scatter
tracking the red dots rather than a clean vertical band -- much harder to
see by eye. This version adds an `ALIGN_TO` switch so you can render EITHER
alignment (or both, run twice) rather than only ever having the cue-end
view. Default is now 'poke' since that's the alignment you said the
response should be locked to; pass 'cue' to get the original behavior back.

FIX 3 (baseline reference, raised separately from the L-vs-R comparison):
added a third column showing the SAME window compared against a genuine
ITI/pre-trial baseline (not stay_Cin, which is active fixation-holding, not
true rest) -- this answers "how much does this pop out from quiet" as a
distinct question from "is left different from right," which the new
overlay row (Fix 1b) answers without needing any baseline at all.

Usage: python plot_track_raster_and_rate_session.py <session_id> <rec_root> <chan> <unit_id> [xlim_lo] [xlim_hi] [align_to]
  align_to: 'poke' (default) or 'cue'
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SESSION_ID = sys.argv[1]
REC_ROOT = sys.argv[2]
CHAN = int(sys.argv[3])
UNIT_ID = int(sys.argv[4])
XLIM = (float(sys.argv[5]) if len(sys.argv) > 5 else -1.5,
        float(sys.argv[6]) if len(sys.argv) > 6 else 1.5)
ALIGN_TO = sys.argv[7] if len(sys.argv) > 7 else "cue"  # 'poke' or 'cue'
assert ALIGN_TO in ("poke", "cue"), "align_to must be 'poke' or 'cue'"

SYNCED = rf"{REC_ROOT}\{SESSION_ID}.kilosort\synced"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
BIN_S = 0.025

trials = pd.read_parquet(os.path.join(SYNCED, "trials.parquet"))
state_events = pd.read_parquet(os.path.join(SYNCED, "state_events.parquet"))
spikes = pd.read_parquet(rf"{OUT}\independent_track_spikes_{SESSION_ID}_ch{CHAN}.parquet")

# "stimulus end" = wait_Sin state's own start time; "choice poke-in" = that
# same state's end time (the instant start_Lin/start_Rin begins) -- both
# straight from the sync pipeline's own state_events table, nothing
# computed independently here.
wait_sin = (state_events[state_events.state_name == "wait_Sin"]
            .sort_values(["trial_id", "start_time_in_trial"]).drop_duplicates("trial_id", keep="first")
            .set_index("trial_id"))
cue_end = wait_sin["start_time_in_trial"]      # stimulus end
poke_in = wait_sin["end_time_in_trial"]        # choice poke-in
decision_s = poke_in - cue_end

# FIX 3: a genuine "quiet" baseline reference, distinct from stay_Cin
# (which is active fixation-holding, not true rest). Uses each trial's OWN
# wait_Cin period -- specifically the window well before the center poke,
# while the state machine is just waiting -- rather than reaching into the
# previous trial's ITI (which would need careful handling of the fixed 10s
# buffer/overlap between adjacent trials' windows, and isn't needed for
# what this is answering: "how much does this pop out from quiet",
# reported as a printed number alongside the plot, not a full extra panel).
center_poke_in = (state_events[state_events.state_name == "stay_Cin"]
                   .sort_values(["trial_id", "start_time_in_trial"]).drop_duplicates("trial_id", keep="first")
                   .set_index("trial_id"))["start_time_in_trial"]
QUIET_WINDOW_S = (-1.0, -0.5)  # relative to center poke-in, i.e. still in wait_Cin, well before any engagement

# TrialCompleted==1: state machine registered an actual side-port poke
# (start_Lin/start_Rin), correct AND error trials both included -- see
# sync_pipeline.bpod_loader.compute_trial_completed().
#
# FIX (sync_valid): the original selection was
#     valid = trials[trials.TrialCompleted == 1].trial_id
# with NO sync_valid filter. TrialCompleted and sync_valid are independent
# flags -- a trial can be TrialCompleted=1 and sync_valid=False at the same
# time, in which case build_spikes_long_table() silently assigned it ZERO
# spikes internally, but it was still counted here as a real data point
# with a manufactured zero. Check the "Trial windows: N/n_trials flagged
# sync_valid=False" line printed by run_sync_and_track_alignment_session.py
# for this session -- if N=0 this fix is a no-op, if not it removes
# spurious zero-count trials from both the plot and (in the companion
# significance script) the statistical test.
valid = trials[(trials.TrialCompleted == 1) & trials.sync_valid].trial_id
valid = valid[valid.isin(wait_sin.index)]

# left/right from the Bpod session's own native ChoiceLeft field (SessionData.Custom.ChoiceLeft),
# not independently inferred -- cross-checked against start_Lin/start_Rin state events
# with 0 mismatches across all completed trials in this session.
choice_left = trials.set_index("trial_id")["ChoiceLeft"].reindex(valid)
left_trials = valid[choice_left.to_numpy() == 1.0].to_numpy()
right_trials = valid[choice_left.to_numpy() == 0.0].to_numpy()

left_order = decision_s.reindex(left_trials).sort_values().index.to_numpy()
right_order = decision_s.reindex(right_trials).sort_values().index.to_numpy()

usp = spikes[spikes.unit_id == UNIT_ID]
by_trial = usp.groupby("trial_id")["spike_time_in_trial"]

BINS = np.arange(XLIM[0], XLIM[1] + 0.001, BIN_S)
CENTERS = (BINS[:-1] + BINS[1:]) / 2

align_ref = poke_in if ALIGN_TO == "poke" else cue_end
align_label = "poke-in" if ALIGN_TO == "poke" else "stimulus end (cue)"
other_ref_all = cue_end if ALIGN_TO == "poke" else poke_in
other_label = "stimulus end" if ALIGN_TO == "poke" else "poke-in"
# median offset of the OTHER reference point relative to the alignment point,
# restricted to the trials actually plotted below -- gives one clearly
# labeled line for "where poke-in typically falls" (or vice versa) in
# addition to the per-trial dots, so both timepoints are unambiguous even
# without tracing individual dots.
_valid_for_median = valid[valid.isin(other_ref_all.index) & valid.isin(align_ref.index)]
median_other_offset = float((other_ref_all.reindex(_valid_for_median) - align_ref.reindex(_valid_for_median)).median())
print(f"Alignment: t=0 is {align_label}. Median {other_label} offset = {median_other_offset:+.3f}s "
      f"(this is the second labeled vertical line in the figure; per-trial actual {other_label} "
      f"times are the red dots).")

# FIX 1: sharey='row' added, plus a new bottom row that overlays both
# curves on one shared-scale axes -- the most direct, scaling-artifact-free
# version of "is left clearly bigger than right in this window."
fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True, sharey='row')

rates = {}  # collected here so the overlay row (below) can reuse them
for col, (order, label) in enumerate([(left_order, "LEFT"), (right_order, "RIGHT")]):
    ax_r = axes[0, col]
    for row_i, tid in enumerate(order):
        t0 = align_ref.loc[tid]
        if tid in by_trial.groups:
            st = by_trial.get_group(tid).to_numpy() - t0
            st_plot = st[(st >= XLIM[0]) & (st <= XLIM[1])]
            ax_r.scatter(st_plot, np.full_like(st_plot, row_i), s=5, c="#1f4e8c", marker="|", linewidths=0.7)
        # mark the OTHER reference point too, per trial (if aligned to
        # poke, show where cue was, and vice versa) -- these dots are the
        # ACTUAL per-trial timing; the dashed line below is just the median.
        ax_r.scatter([other_ref_all.loc[tid] - t0], [row_i], s=6, c="red", marker="o", linewidths=0, zorder=3)
    ax_r.axvline(0, color="k", lw=1.3, ls="--", label=align_label, zorder=2)
    ax_r.axvline(median_other_offset, color="red", lw=1.3, ls=":", label=f"{other_label} (median)", zorder=2)
    ax_r.set_xlim(*XLIM)
    ax_r.set_title(f"{label} (n={len(order)})")
    if col == 0:
        ax_r.set_ylabel("trial (sorted by decision time)")
        ax_r.scatter([], [], s=10, c="red", marker="o", label=f"{other_label} (actual, per trial)")
        ax_r.legend(fontsize=7, loc="upper right")

    ax_h = axes[1, col]
    counts = np.zeros(len(CENTERS))
    for tid in order:
        if tid in by_trial.groups:
            st = by_trial.get_group(tid).to_numpy() - align_ref.loc[tid]
            h, _ = np.histogram(st, bins=BINS)
            counts += h
    rate = counts / BIN_S / max(len(order), 1)
    rates[label] = rate
    ax_h.plot(CENTERS, rate, color="#333", lw=1.6, drawstyle="steps-mid")
    ax_h.axvline(0, color="k", lw=1.3, ls="--")
    ax_h.axvline(median_other_offset, color="red", lw=1.3, ls=":")
    ax_h.set_xlabel(f"time from {align_label} (s)")
    if col == 0:
        ax_h.set_ylabel(f"firing rate (Hz), raw {int(BIN_S*1000)}ms bins")

# FIX 1b: direct overlay, one axes, shared scale by construction -- this is
# the panel to actually judge "is left clearly bigger than right" from.
ax_o = axes[2, 0]
ax_o.plot(CENTERS, rates["LEFT"], color="tab:blue", lw=2, label="LEFT")
ax_o.plot(CENTERS, rates["RIGHT"], color="tab:red", lw=2, label="RIGHT")
ax_o.axvline(0, color="k", lw=1.3, ls="--", label=align_label)
ax_o.axvline(median_other_offset, color="grey", lw=1.3, ls=":", label=f"{other_label} (median)")
ax_o.set_xlabel(f"time from {align_label} (s)")
ax_o.set_ylabel("firing rate (Hz)")
ax_o.set_title("LEFT vs RIGHT, overlaid (same axes -- no scaling ambiguity)")
ax_o.legend(fontsize=8)

axes[2, 1].axis("off")
axes[2, 1].text(
    0.02, 0.5,
    f"Peak rate -- LEFT: {rates['LEFT'].max():.1f}Hz   RIGHT: {rates['RIGHT'].max():.1f}Hz\n"
    f"Mean rate in window -- LEFT: {rates['LEFT'].mean():.1f}Hz   RIGHT: {rates['RIGHT'].mean():.1f}Hz\n\n"
    f"(These numbers are the ground truth for the L-vs-R comparison --\n"
    f"if the overlay above still looks ambiguous, trust these over the\n"
    f"visual, and if they ARE clearly different, the overlay should show it.)",
    fontsize=9, va="center", family="monospace",
)

plt.suptitle(f"{SESSION_ID}, ch{CHAN}, unit {UNIT_ID} -- raw raster + raw rate, no smoothing, no stats "
             f"(aligned to {align_label})\nblack dashed = {align_label} (t=0)   |   "
             f"red dotted/dots = {other_label} (median line / actual per-trial)", fontsize=10)
plt.tight_layout()
fig_path = rf"{OUT}\raster_rate_{SESSION_ID}_ch{CHAN}_unit{UNIT_ID}_{ALIGN_TO}aligned.png"
plt.savefig(fig_path, dpi=120, bbox_inches="tight")
print(f"saved {fig_path}")
print(f"LEFT  peak={rates['LEFT'].max():.2f}Hz  mean={rates['LEFT'].mean():.2f}Hz  n_trials={len(left_order)}")
print(f"RIGHT peak={rates['RIGHT'].max():.2f}Hz  mean={rates['RIGHT'].mean():.2f}Hz  n_trials={len(right_order)}")

# FIX 3, continued: print the quiet-baseline rate for context (this is the
# "how much does this pop out from true rest" number, separate from the
# L-vs-R contrast above -- a response can be strongly left>right without
# necessarily being far above a quiet baseline, and vice versa).
def quiet_rate(order):
    total_spikes, total_dur = 0, 0.0
    for tid in order:
        if tid not in center_poke_in.index:
            continue
        t_poke = center_poke_in.loc[tid]
        lo, hi = t_poke + QUIET_WINDOW_S[0], t_poke + QUIET_WINDOW_S[1]
        if lo < 0:
            continue  # this trial's wait_Cin wasn't long enough to have a clean quiet window
        if tid in by_trial.groups:
            st = by_trial.get_group(tid).to_numpy()
            total_spikes += int(((st >= lo) & (st < hi)).sum())
        total_dur += (hi - lo)
    return total_spikes / total_dur if total_dur > 0 else float("nan")

quiet_left = quiet_rate(left_order)
quiet_right = quiet_rate(right_order)
print(f"\nQuiet-baseline rate (wait_Cin, {QUIET_WINDOW_S[0]} to {QUIET_WINDOW_S[1]}s before center poke-in):")
print(f"LEFT  quiet baseline={quiet_left:.2f}Hz   (response peak was {rates['LEFT'].max():.2f}Hz)")
print(f"RIGHT quiet baseline={quiet_right:.2f}Hz   (response peak was {rates['RIGHT'].max():.2f}Hz)")

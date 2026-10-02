"""Which neurons respond at reward delivery, and which at the end of the cue?

Session 20260916_110311 (rat Shamir), synced by sync_pipeline. Units restricted
to the 111 clusters Gil hand-labelled `good` in the review tool -- NOT Phy's
labels and NOT Kilosort's KSLabel (units.parquet's own `quality_label` column is
KSLabel and calls 189 units good; that is not the curation asked for here).

THE TWO EVENTS, taken from TASK_TIMELINE.md rather than guessed:

  REWARD DELIVERY = the start of `water_L` / `water_R` -- the instant the valve
    opens (Step 12). Present on 291 of the 478 completed trials. Note this is
    NOT `rewarded_Lin`/`rewarded_Rin`: those are the Step-11 WAITING states,
    named after whichever side was correct, and they fire thousands of times
    per session because the grace loop re-enters them.

  CUE END = the end of `stimulus_delivery` (Step 7), which this session is
    identical to the start of `wait_Sin` (Step 8) because
    `AuditoryStimulusTime - MinSampleAud` = 0.35 - 0.35 = 0 s. Verified in the
    data, not assumed: the two timestamps agree to floating point on every
    trial where both exist.

TRIAL SELECTION. `sync_valid & TrialCompleted` -- 478 of 953 trials. The other
475 are genuinely incomplete: the rat never poked centre, broke fixation,
withdrew early during sampling, or never chose a side. `TrialCompleted` is the
state-machine's own choice registration, not raw pokes (the raw-poke version
overcounted by ~44 trials on the pipeline's example session).

A CONFOUND THAT CANNOT BE DESIGNED AWAY HERE, SO IT IS REPORTED INSTEAD. In
this task cue offset, the side lights coming on, and the start of the rat's
movement are the SAME instant -- `MT` is measured from `wait_Sin`'s start, and
its median is 0.253 s. So any window after cue end longer than ~0.2 s is
contaminated by movement and by the choice poke itself. The cue-end test
therefore uses a 0.15 s window, chosen to close before the 10th-percentile
movement time (0.205 s), and even that cannot separate "responds to the
stimulus ending" from "responds to initiating a choice". The reward test has no
such problem: the median feedback delay is 1.61 s, so the pre-reward baseline
is a long quiet hold inside the port.

STATISTICS. Per unit, per event: spike counts in a test window vs a matched
baseline window, paired across trials, Wilcoxon signed-rank (two-sided,
non-parametric -- per-trial counts are small and skewed). Benjamini-Hochberg
FDR across the 111 units, separately for each event. A side preference is
tested on the test-window rate with Mann-Whitney U, left-choice vs
right-choice trials, FDR-corrected the same way.

Usage: python analyse_reward_and_cue_responses.py
"""
import os
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import wilcoxon, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYNC = (r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort"
        r"\synced_20261002")
VERDICTS = r"D:\Gil\spike_sorting_agent\outputs\manual_verdicts_20260916_110311.csv"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
FIGDIR = os.path.join(OUT, "reward_cue_figs")
os.makedirs(FIGDIR, exist_ok=True)

# windows, in seconds relative to the event
REWARD_BASE = (-0.30, 0.0)    # quiet hold in port during the feedback wait
REWARD_TEST = (0.0, 0.30)
CUE_BASE = (-0.30, 0.0)       # during the auditory stimulus (0.35 s long)
CUE_TEST = (0.0, 0.15)        # closes before p10 movement time (0.205 s)
PSTH_WIN = (-0.6, 0.8)
PSTH_BIN = 0.020
ALPHA = 0.05
MIN_MOD = 0.10                # |modulation| counted as a SUBSTANTIAL response
MIN_TRIALS = 20
MIN_SPIKES = 100              # a unit needs this many spikes in the session


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.full(len(p), np.nan)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


# ------------------------------------------------------------------ load
trials = pd.read_parquet(os.path.join(SYNC, "trials.parquet"))
se = pd.read_parquet(os.path.join(SYNC, "state_events.parquet"))
units = pd.read_parquet(os.path.join(SYNC, "units.parquet"))
meta = json.load(open(os.path.join(SYNC, "session_meta.json")))

ver = pd.read_csv(VERDICTS)
good_units = sorted(set(ver.loc[ver.verdict == "good", "unit"].astype(int))
                    & set(units.unit_id.astype(int)))

clean = trials[trials.sync_valid & trials.TrialCompleted].copy()
print(f"session 20260916_110311: {len(trials)} trials, "
      f"{len(clean)} completed and sync-valid ({100*len(clean)/len(trials):.0f}%)")
print(f"review-tool 'good' units: {len(good_units)} "
      f"(units.parquet KSLabel would say {int((units.quality_label=='good').sum())})")
print(f"sync: {meta['sync']['match_fraction']:.3f} matched "
      f"({meta['sync']['n_matched']}/{meta['sync']['n_bpod_events']}), "
      f"local-interp held-out error median "
      f"{meta['sync']['interp_holdout_median_ms']:.3f} ms / max "
      f"{meta['sync']['interp_holdout_max_ms']:.1f} ms; gaps="
      f"{meta['sync']['gaps']}")

# ------------------------------------------------------- event times
def first_state(name, field):
    s = se[se.state_name == name].sort_values(["trial_id", "occurrence"])
    s = s.groupby("trial_id").first()
    return s[field]


cue_end = first_state("stimulus_delivery", "end_time_in_trial")
wait_sin = first_state("wait_Sin", "start_time_in_trial")
stim_on = first_state("stimulus_delivery_min", "start_time_in_trial")
both = cue_end.index.intersection(wait_sin.index)
agree = np.allclose(cue_end.loc[both], wait_sin.loc[both], atol=1e-9)
print(f"cue-end definition check: stimulus_delivery end == wait_Sin start on "
      f"{len(both)} trials -> {'identical' if agree else 'DIFFER'}")

water = se[se.state_name.isin(["water_L", "water_R"])].copy()
water = water.sort_values(["trial_id", "occurrence"]).groupby("trial_id").first()
reward_time = water.start_time_in_trial
reward_side = water.state_name.map({"water_L": "left", "water_R": "right"})

ev = clean[["trial_id", "ChoiceLeft", "Rewarded", "ChoiceCorrect",
            "CatchTrial", "MT", "DV"]].set_index("trial_id")
ev["stim_on"] = stim_on
ev["cue_end"] = cue_end
ev["reward_time"] = reward_time
ev["side"] = np.where(ev.ChoiceLeft == 1, "left",
                      np.where(ev.ChoiceLeft == 0, "right", "none"))

# The nearest thing to a reward-omission control this session allows. ALL 187
# unrewarded completed trials end in `skipped_feedback` -- the rat gave up
# waiting -- including all 71 incorrect choices, because GUI.CatchError=1
# stretches the incorrect-choice wait to 20 s so it is never sat out. So there
# is NO delivered no-reward outcome to align to, and a matched omission
# contrast is impossible here. The give-up moment is a different behaviour (the
# rat is leaving the port, not receiving nothing at the expected time), so it
# tests only the weaker question: is a "reward" response specific to reward, or
# does it also appear when the trial simply ends without water?
giveup = se[se.state_name == "skipped_feedback"].sort_values(
    ["trial_id", "occurrence"]).groupby("trial_id").first()
ev["giveup_time"] = giveup.start_time_in_trial

n_rew = int(ev.reward_time.notna().sum())
n_cue = int(ev.cue_end.notna().sum())
print(f"\nevents on completed trials: cue end {n_cue}, reward delivery {n_rew}")
print(f"  choice side: {ev.side.value_counts().to_dict()}")
print(f"  reward by side: "
      f"{ev[ev.reward_time.notna()].side.value_counts().to_dict()}")
print(f"  MT median {clean.MT.median():.3f}s (p10 {clean.MT.quantile(.1):.3f}) "
      f"-- cue test window closes at {CUE_TEST[1]:.2f}s")
print(f"  FeedbackDelay median {clean.FeedbackDelay.median():.2f}s "
      f"-- reward baseline is a quiet in-port hold")

# ------------------------------------------------------- load spikes
print("\nloading spikes for good units...")
want = set(good_units)
keep = []
pf = pq.ParquetFile(os.path.join(SYNC, "spikes.parquet"))
for i in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(i, columns=["unit_id", "trial_id",
                                      "spike_time_in_trial"]).to_pandas()
    keep.append(t[t.unit_id.isin(want)])
sp = pd.concat(keep, ignore_index=True)
del keep
print(f"  {len(sp):,} spike rows for {sp.unit_id.nunique()} units")
print("  NOTE: 952/953 trial windows overlap the next trial, so a spike can "
      "appear under two trial_ids. Each trial's PSTH uses its own window, so "
      "this is consistent; it only matters if pooling raw spike counts.")

n_spikes_total = sp.unit_id.value_counts().to_dict()

# index once: unit -> trial -> sorted spike times. Filtering the frame per
# (unit, trial) inside the loops instead would be 111 units x 478 trials x 40M
# rows of scanning.
sp = sp.sort_values(["unit_id", "trial_id", "spike_time_in_trial"])
by_unit = {}
for u, g in sp.groupby("unit_id", sort=False):
    by_unit[int(u)] = {int(t): gg.to_numpy()
                       for t, gg in g.groupby("trial_id", sort=False)["spike_time_in_trial"]}
del sp


def counts_in(idx, ev_times, lo, hi):
    """Per-trial spike counts in [event+lo, event+hi), aligned to ev_times."""
    out = np.zeros(len(ev_times), dtype=int)
    for k, (tid, t0) in enumerate(ev_times.items()):
        s = idx.get(int(tid))
        if s is None:
            continue
        out[k] = np.searchsorted(s, t0 + hi, "left") - \
            np.searchsorted(s, t0 + lo, "left")
    return out


def analyse(event_name, times, base, test):
    rows = []
    sides = ev.loc[times.index, "side"].to_numpy()
    for u in good_units:
        idx = by_unit.get(u)
        if idx is None or n_spikes_total.get(u, 0) < MIN_SPIKES:
            continue
        nb = counts_in(idx, times, *base)
        nt = counts_in(idx, times, *test)
        if len(nt) < MIN_TRIALS:
            continue
        rb = nb / (base[1] - base[0])
        rt = nt / (test[1] - test[0])
        d = rt - rb
        try:
            p = float(wilcoxon(d, zero_method="zsplit").pvalue) \
                if np.any(d != 0) else 1.0
        except Exception:
            p = np.nan
        # side preference on the test-window rate
        ml, mr = sides == "left", sides == "right"
        p_side = np.nan
        if ml.sum() >= 10 and mr.sum() >= 10:
            try:
                p_side = float(mannwhitneyu(rt[ml], rt[mr],
                                            alternative="two-sided").pvalue)
            except Exception:
                p_side = np.nan
        rows.append(dict(
            unit=u, event=event_name, n_trials=len(nt),
            baseline_hz=float(rb.mean()), test_hz=float(rt.mean()),
            delta_hz=float(d.mean()),
            modulation=float((rt.mean() - rb.mean()) /
                             (rt.mean() + rb.mean() + 1e-12)),
            p=p, p_side=p_side,
            left_hz=float(rt[ml].mean()) if ml.sum() else np.nan,
            right_hz=float(rt[mr].mean()) if mr.sum() else np.nan,
            n_left=int(ml.sum()), n_right=int(mr.sum()),
            channel=int(units.set_index("unit_id").loc[u, "channel"]),
            depth_mm=float(units.set_index("unit_id").loc[u, "depth_mm"])))
    df = pd.DataFrame(rows)
    if len(df):
        df["q"] = bh_fdr(df.p.to_numpy())
        df["q_side"] = bh_fdr(df.p_side.to_numpy())
        df["direction"] = np.where(df.delta_hz > 0, "increase", "decrease")
    return df


print("\nanalysing cue end...")
cue_df = analyse("cue_end", ev.cue_end.dropna(), CUE_BASE, CUE_TEST)
print("analysing reward delivery...")
rew_df = analyse("reward", ev.reward_time.dropna(), REWARD_BASE, REWARD_TEST)
print("analysing the give-up moment (no-water control)...")
giv_df = analyse("giveup", ev.giveup_time.dropna(), REWARD_BASE, REWARD_TEST)

res = pd.concat([cue_df, rew_df, giv_df], ignore_index=True)
res.to_csv(os.path.join(OUT, "session_20260916_reward_cue_responses.csv"),
           index=False)

# ------------------------------------------------------------- report
for name, df, base, test in [
        ("END OF CUE STIMULUS", cue_df, CUE_BASE, CUE_TEST),
        ("REWARD DELIVERY (valve open)", rew_df, REWARD_BASE, REWARD_TEST),
        ("GIVE-UP MOMENT, no water (control)", giv_df, REWARD_BASE, REWARD_TEST)]:
    print("\n" + "=" * 88)
    print(f"{name}   baseline {base} s vs test {test} s, paired across trials")
    print("=" * 88)
    if not len(df):
        print("  no units testable")
        continue
    sig = df[df.q < ALPHA]
    up = sig[sig.direction == "increase"]
    dn = sig[sig.direction == "decrease"]
    strong = sig[sig.modulation.abs() >= MIN_MOD]
    print(f"  {len(df)} good units tested, {len(sig)} responsive at FDR<{ALPHA} "
          f"({100*len(sig)/len(df):.0f}%)  -- {len(up)} increase, {len(dn)} decrease")
    print(f"  of those, {len(strong)} are SUBSTANTIAL "
          f"(|modulation| >= {MIN_MOD:.2f}, i.e. the rate changes by >=~20%) "
          f"-- {100*len(strong)/len(df):.0f}% of all good units.")
    print(f"  (with {len(df) and df.n_trials.iloc[0]} trials a paired test "
          f"detects very small shifts, so the significance count alone "
          f"overstates how many neurons meaningfully respond.)")
    ss = df[df.q_side < ALPHA]
    print(f"  {len(ss)} units differ between LEFT and RIGHT choices "
          f"(FDR<{ALPHA}, {100*len(ss)/len(df):.0f}%)")
    show = sig.reindex(sig.delta_hz.abs().sort_values(ascending=False).index)
    print(f"\n  strongest responses (top {min(15,len(show))} by |delta|):")
    cols = ["unit", "channel", "depth_mm", "baseline_hz", "test_hz", "delta_hz",
            "modulation", "q", "left_hz", "right_hz", "q_side", "direction"]
    print(show[cols].head(15).round(3).to_string(index=False))

# units responsive to BOTH
if len(cue_df) and len(rew_df):
    c = set(cue_df.loc[cue_df.q < ALPHA, "unit"])
    r = set(rew_df.loc[rew_df.q < ALPHA, "unit"])
    print("\n" + "=" * 88)
    print("OVERLAP")
    print("=" * 88)
    print(f"  cue-end only : {len(c - r)}")
    print(f"  reward only  : {len(r - c)}")
    print(f"  both         : {len(c & r)}   {sorted(c & r)}")
    print(f"  neither      : {len(set(cue_df.unit) - c - r)}")

# is the reward response specific to reward?
if len(rew_df) and len(giv_df):
    print("\n" + "=" * 88)
    print("IS THE REWARD RESPONSE SPECIFIC TO REWARD?")
    print("same window, same baseline, aligned instead to the give-up moment")
    print("(the only no-water trial end this session has -- a DIFFERENT")
    print(" behaviour, not a matched omission, so this is suggestive only)")
    print("=" * 88)
    a = rew_df.set_index("unit")
    b = giv_df.set_index("unit")
    both_u = a.index.intersection(b.index)
    sigr = [u for u in both_u if a.loc[u, "q"] < ALPHA
            and abs(a.loc[u, "modulation"]) >= MIN_MOD]
    if sigr:
        also = [u for u in sigr if b.loc[u, "q"] < ALPHA
                and np.sign(b.loc[u, "delta_hz"]) == np.sign(a.loc[u, "delta_hz"])]
        print(f"  {len(sigr)} units respond substantially at reward.")
        print(f"  {len(also)} of them also respond the same way at the give-up "
              f"moment ({100*len(also)/len(sigr):.0f}%) -> not reward-specific")
        print(f"  {len(sigr)-len(also)} respond at reward but NOT at give-up "
              f"-> consistent with a reward-specific signal")
        cmp = pd.DataFrame({
            "reward_delta_hz": a.loc[sigr, "delta_hz"],
            "giveup_delta_hz": b.loc[sigr, "delta_hz"],
            "reward_q": a.loc[sigr, "q"], "giveup_q": b.loc[sigr, "q"]})
        cmp["reward_specific"] = ~cmp.index.isin(also)
        print()
        print(cmp.sort_values("reward_delta_hz", key=abs, ascending=False)
              .round(4).to_string())

# ------------------------------------------------------------- figures
edges = np.arange(PSTH_WIN[0], PSTH_WIN[1] + PSTH_BIN, PSTH_BIN)
centres = edges[:-1] + PSTH_BIN / 2


def psth_and_raster(u, times, label, ax_r, ax_p, test):
    idx = by_unit[u]
    sides = ev.loc[times.index, "side"]
    colors = {"left": "#1f6fb4", "right": "#c2541a"}
    y = 0
    counts = {"left": [], "right": []}
    for sd in ("left", "right"):
        tids = [t for t in times.index if sides.loc[t] == sd]
        for tid in tids:
            raw = idx.get(int(tid))
            s = (raw - times.loc[tid]) if raw is not None else np.array([])
            s = s[(s >= PSTH_WIN[0]) & (s < PSTH_WIN[1])]
            if len(s):
                ax_r.plot(s, np.full(len(s), y), "|", ms=2.4, lw=0.5,
                          color=colors[sd])
            counts[sd].append(np.histogram(s, bins=edges)[0])
            y += 1
    ax_r.axvline(0, color="#333", lw=1.1)
    ax_r.axvspan(test[0], test[1], color="#888", alpha=0.13, lw=0)
    ax_r.set_xlim(PSTH_WIN)
    ax_r.set_ylim(-1, max(y, 1))
    ax_r.set_ylabel("trial (sorted by choice)", fontsize=8)
    ax_r.set_title(label, fontsize=9.5)
    ax_r.tick_params(labelsize=7)
    for sd in ("left", "right"):
        if not counts[sd]:
            continue
        m = np.vstack(counts[sd]).mean(axis=0) / PSTH_BIN
        sem = np.vstack(counts[sd]).std(axis=0, ddof=1) / \
            np.sqrt(len(counts[sd])) / PSTH_BIN if len(counts[sd]) > 1 else np.zeros_like(m)
        ax_p.plot(centres, m, color=colors[sd], lw=1.6,
                  label=f"{sd} (n={len(counts[sd])})")
        ax_p.fill_between(centres, m - sem, m + sem, color=colors[sd],
                          alpha=0.2, lw=0)
    ax_p.axvline(0, color="#333", lw=1.1)
    ax_p.axvspan(test[0], test[1], color="#888", alpha=0.13, lw=0)
    ax_p.set_xlim(PSTH_WIN)
    ax_p.set_xlabel("time from event (s)", fontsize=8)
    ax_p.set_ylabel("rate (Hz)", fontsize=8)
    ax_p.legend(fontsize=7, frameon=False)
    ax_p.tick_params(labelsize=7)


made = []
for ename, df, times, test in [("cue_end", cue_df, ev.cue_end.dropna(), CUE_TEST),
                               ("reward", rew_df, ev.reward_time.dropna(), REWARD_TEST)]:
    if not len(df):
        continue
    sig = df[df.q < ALPHA]
    top = sig.reindex(sig.delta_hz.abs().sort_values(ascending=False).index).head(6)
    if not len(top):
        continue
    fig, axes = plt.subplots(2, len(top), figsize=(3.0 * len(top), 6.2),
                             sharex=True)
    if len(top) == 1:
        axes = axes.reshape(2, 1)
    for k, (_, r) in enumerate(top.iterrows()):
        psth_and_raster(int(r.unit), times,
                        f"unit {int(r.unit)}  ch{int(r.channel)}\n"
                        f"{r.baseline_hz:.1f}->{r.test_hz:.1f} Hz  q={r.q:.1e}",
                        axes[0, k], axes[1, k], test)
    lbl = "end of cue stimulus" if ename == "cue_end" else "reward delivery (valve open)"
    fig.suptitle(f"Session 20260916_110311 — strongest responses at {lbl}\n"
                 f"completed trials only, review-tool 'good' units, "
                 f"split by choice side", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(FIGDIR, f"{ename}_top_units.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    made.append(p)
    print(f"\nsaved {p}")

print("\nsaved session_20260916_reward_cue_responses.csv")

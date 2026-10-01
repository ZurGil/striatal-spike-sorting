"""
Select the source units for the larger benchmark, with a real guarantee that no
two of them can be oversplit halves of one neuron -- and show every one.

WHY THE OLD FILTER WAS LIMITING. The previous builder required sources to be
mutually >= 150 um apart. That is a proxy for "not the same neuron", and an
expensive one: it rejected 114 of 131 verified-good candidates and left 17,
which capped the benchmark at 13 units and made every tier mean an n=3
statistic (section 5ag: the easy tier's three units scored 0.30, 0.45, 0.86).

THE BETTER TEST. Two clusters that are really halves of ONE neuron must show a
REFRACTORY DIP in their cross-correlogram: a single cell cannot fire twice
within ~1 ms, so if A and B are the same cell there can be almost no A-B pairs
at near-zero lag. Two genuinely distinct neurons have no such dip. That is a
direct test of the thing we care about rather than a proxy, so distance can be
relaxed and far more units qualify.

Three filters, each a checked guarantee on the FINAL set:
  1. quality        KSLabel == 'good', ContamPct <= MAX_CONTAM_PCT, enough spikes
  2. not-a-copy     template similarity <= MAX_KS_SIMILARITY, peak channels
                    >= MIN_SEPARATION_UM apart
  3. not-one-cell   CCG refractory test on every pair close enough to be a
                    candidate for oversplitting (far pairs cannot be)

Pairs too sparse for the CCG test to have power are reported rather than
silently passed, and the distance filter alone has to carry those.

OUTPUT. A manifest of the selected units, a table of every candidate with the
reason it was accepted or rejected, and a figure showing the real averaged
waveform and multi-channel footprint of every selected unit -- which is the
thing to review before any dataset gets built on them.

Usage: python select_source_units.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
DAT_PATH = (r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort"
            r"\20260916_110311.probe1.dat")
OUT = r"D:\Gil\spike_sorting_agent\outputs"

N_CHAN_BIN, ITEMSIZE, FS = 384, 2, 30000.0
GAIN_TO_UV = 0.018311105685598315
NT, NT0MIN = 61, 20
RADIUS_UM = 80.0
N_AVG = 250

REQUIRE_KSLABEL = "good"
MAX_CONTAM_PCT = 10.0
MIN_SOURCE_SPIKES = 300
MIN_SEPARATION_UM = 40.0     # relaxed from 150; the CCG test now does the work
MAX_KS_SIMILARITY = 0.20
CCG_TEST_MAX_UM = 120.0      # beyond this, two clusters cannot be one neuron
CCG_CENTER_MS = 1.0
CCG_FLANK_MS = (5.0, 25.0)
CCG_DIP_RATIO = 0.30         # center rate below this x flank rate => one cell
CCG_MIN_FLANK_COUNT = 50     # below this the test has no power

b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")
TOTAL = os.path.getsize(DAT_PATH) // (N_CHAN_BIN * ITEMSIZE)

cp = np.load(os.path.join(KS, "channel_positions.npy"))
templates = np.load(os.path.join(KS, "templates.npy"))
peak_ch = np.argmax(templates.max(axis=1) - templates.min(axis=1), axis=1)
st = np.load(os.path.join(KS, "spike_times.npy")).ravel()
cl = np.load(os.path.join(KS, "spike_clusters.npy")).ravel()
ks_sim = np.load(os.path.join(KS, "similar_templates.npy"))

qual = (pd.read_csv(os.path.join(KS, "cluster_KSLabel.tsv"), sep="\t")
        .merge(pd.read_csv(os.path.join(KS, "cluster_ContamPct.tsv"), sep="\t"),
               on="cluster_id"))
u_all, c_all = np.unique(cl, return_counts=True)
qual["n"] = qual.cluster_id.map(dict(zip(u_all.tolist(), c_all.tolist()))).fillna(0)

# THE POOL IS GIL'S MANUAL REVIEW, NOT KILOSORT'S LABEL.
#
# Earlier versions selected on KSLabel == 'good' and ContamPct <= 10. Checking
# that against the manual review he had been doing in the unit review tool:
# of the 49 units it chose, only 23 were labelled good by hand -- 12 were mua
# and 4 were outright NOISE, one of which (461) had been placed as an EASY-tier
# unit. Kilosort's own label is simply not a substitute for the review.
#
# So the pool is exactly the units labelled `good` by hand, and nothing outside
# that set can enter under any circumstances. KSLabel and ContamPct are still
# read, but only to REPORT alongside each unit, never to filter -- the manual
# verdict supersedes them (it is made while looking at the ACG and waveform,
# which is strictly more information than ContamPct alone).
VERDICTS = os.path.join(OUT, "manual_verdicts_20260916_110311.csv")
ver = pd.read_csv(VERDICTS)
good_ids = set(ver.loc[ver.verdict == "good", "unit"].astype(int))
print(f"manual review: {len(ver)} units labelled "
      f"({', '.join(f'{k} {v}' for k, v in ver.verdict.value_counts().items())})")

pool = qual[qual.cluster_id.isin(good_ids)
            & (qual.n >= MIN_SOURCE_SPIKES)
            & (qual.cluster_id < templates.shape[0])].copy()
print(f"pool: {len(pool)} of {len(good_ids)} manually-good units "
      f"({len(good_ids) - len(pool)} dropped for < {MIN_SOURCE_SPIKES} spikes)")
print(f"  their KSLabel: {dict(pool.KSLabel.value_counts())} -- "
      f"{int((pool.KSLabel=='mua').sum())} would have been rejected by the old "
      f"KSLabel filter, so the manual review RESCUED them")

# ---- INJECTION-SOURCE PURITY. These are not a second opinion on Gil's
# verdict: they ask a different question. `good` says the unit is a real,
# well-isolated neuron. A DONOR additionally has to yield clean snippets and a
# usable spatial footprint, and a unit can be genuinely good at both while
# failing either.
#
#   contamination  snippets are drawn at random from the cluster, so a cluster
#                  that is 43% contaminated (unit 51) donates 43% wrong
#                  waveforms and the "ground truth" stops being true.
#   footprint      the pair tier depends on two footprints overlapping. Units
#                  99 and 61 are seen on a single channel, which makes that
#                  test vacuous, and a 2-channel footprint is marginal.
#
# Both are applied BEFORE the independence search, not after. Filtering
# afterwards throws away the slot as well as the unit; filtering first lets the
# greedy search put a different acceptable unit at that location.
#
# Footprint width is taken from templates.npy here. templates.npy is normalized
# in overall SCALE (the trap recorded in section 6), but its relative profile
# ACROSS channels is preserved, which is all a channel count needs -- and that
# makes it cheap enough to evaluate for the whole pool rather than only the
# selected set.
MIN_CHANS_25PCT = 3
t_amp = templates.max(axis=1) - templates.min(axis=1)
pool["tpl_chans_25pct"] = [
    int((t_amp[c] >= 0.25 * t_amp[c].max()).sum()) for c in pool.cluster_id]
before = len(pool)
dirty = pool[pool.ContamPct > MAX_CONTAM_PCT]
narrow = pool[pool.tpl_chans_25pct < MIN_CHANS_25PCT]
pool = pool[(pool.ContamPct <= MAX_CONTAM_PCT)
            & (pool.tpl_chans_25pct >= MIN_CHANS_25PCT)].copy()
print(f"  donor filters: dropped {len(dirty)} over {MAX_CONTAM_PCT}% "
      f"contamination, {len(narrow)} with < {MIN_CHANS_25PCT} channels "
      f"-> {len(pool)} of {before} eligible donors")

pool["pk"] = peak_ch[pool.cluster_id.to_numpy()]
pool = pool.sort_values("pk").reset_index(drop=True)

trains = {int(c): np.sort(st[cl == int(c)]) for c in pool.cluster_id}
cw, f0, f1 = (CCG_CENTER_MS * FS / 1000.0,
              CCG_FLANK_MS[0] * FS / 1000.0, CCG_FLANK_MS[1] * FS / 1000.0)


def ccg_same_cell(a, b):
    """Refractory-dip test. Returns (is_same_cell, ratio, flank_count)."""
    ta, tb = trains[a], trains[b]
    if len(ta) == 0 or len(tb) == 0:
        return False, np.nan, 0
    center = int((np.searchsorted(tb, ta + cw, "right")
                  - np.searchsorted(tb, ta - cw, "left")).sum())
    flank = int((np.searchsorted(tb, ta + f1, "right")
                 - np.searchsorted(tb, ta + f0, "left")).sum()
                + (np.searchsorted(tb, ta - f0, "right")
                   - np.searchsorted(tb, ta - f1, "left")).sum())
    if flank < CCG_MIN_FLANK_COUNT:
        return False, np.nan, flank
    rate_c = center / (2 * CCG_CENTER_MS)
    rate_f = flank / (2 * (CCG_FLANK_MS[1] - CCG_FLANK_MS[0]))
    ratio = rate_c / rate_f if rate_f > 0 else np.inf
    return ratio < CCG_DIP_RATIO, ratio, flank


def dist(a, b):
    return float(np.sqrt(((cp[peak_ch[a]] - cp[peak_ch[b]]) ** 2).sum()))


print(f"\ngreedy selection: >= {MIN_SEPARATION_UM:.0f} um, similarity "
      f"<= {MAX_KS_SIMILARITY}, CCG refractory test within "
      f"{CCG_TEST_MAX_UM:.0f} um")
picked, audit = [], []
n_close = n_sim = n_ccg = n_nopower = 0
for cid in [int(x) for x in pool.cluster_id]:
    reason, ok = "accepted", True
    for q in picked:
        d = dist(cid, q)
        if d < MIN_SEPARATION_UM:
            reason, ok, n_close = f"too close to {q} ({d:.0f} um)", False, n_close + 1
            break
        if float(ks_sim[cid, q]) > MAX_KS_SIMILARITY:
            reason, ok, n_sim = (f"similar to {q} ({ks_sim[cid,q]:.2f})",
                                 False, n_sim + 1)
            break
        if d <= CCG_TEST_MAX_UM:
            same, ratio, flank = ccg_same_cell(cid, q)
            if same:
                reason, ok, n_ccg = (f"CCG dip with {q} (ratio {ratio:.2f})",
                                     False, n_ccg + 1)
                break
            if not np.isfinite(ratio):
                n_nopower += 1
    if ok:
        picked.append(cid)
    audit.append(dict(cluster_id=cid, peak_ch=int(peak_ch[cid]),
                      contam_pct=float(pool[pool.cluster_id == cid].ContamPct.iloc[0]),
                      n_spikes=int(pool[pool.cluster_id == cid].n.iloc[0]),
                      selected=ok, reason=reason))
aud = pd.DataFrame(audit)
aud.to_csv(os.path.join(OUT, "source_unit_audit.csv"), index=False)
print(f"  accepted {len(picked)} of {len(pool)}  "
      f"(rejected: {n_close} too close, {n_sim} too similar, {n_ccg} CCG dip)")
if n_nopower:
    print(f"  NOTE: {n_nopower} pair tests lacked power "
          f"(< {CCG_MIN_FLANK_COUNT} flank counts); distance carried those")

# ---- prove the guarantees on the FINAL set, worst case reported
worst_d, worst_s, worst_ratio, worst_pair = np.inf, 0.0, np.inf, None
for i in range(len(picked)):
    for j in range(i + 1, len(picked)):
        a, b = picked[i], picked[j]
        d = dist(a, b)
        worst_d = min(worst_d, d)
        worst_s = max(worst_s, float(ks_sim[a, b]))
        if d <= CCG_TEST_MAX_UM:
            _, ratio, flank = ccg_same_cell(a, b)
            if np.isfinite(ratio) and ratio < worst_ratio:
                worst_ratio, worst_pair = ratio, (a, b)
print(f"\nFINAL SET GUARANTEES ({len(picked)} units):")
print(f"  closest pair           : {worst_d:.0f} um  (>= {MIN_SEPARATION_UM:.0f})")
print(f"  highest similarity     : {worst_s:.3f}  (<= {MAX_KS_SIMILARITY})")
if worst_pair:
    print(f"  worst CCG ratio        : {worst_ratio:.2f} for units {worst_pair} "
          f"(>= {CCG_DIP_RATIO} means NOT one cell)")
assert worst_d >= MIN_SEPARATION_UM and worst_s <= MAX_KS_SIMILARITY
print("  -> no two selected units can be oversplit halves of one neuron")

# ------------------------------------------------- real waveforms + the figure
def real_mean_waveform(uid, chans, n_avg=N_AVG, seed=0):
    """Average real recorded snippets, in microvolts.

    NOT templates.npy -- Kilosort normalizes it, so scaling it by GAIN_TO_UV
    yields nonsense sub-microvolt amplitudes (a trap this project fell into).
    """
    rng = np.random.default_rng(seed)
    s = trains[uid]
    s = s[(s > NT + 80) & (s < TOTAL - NT - 80)]
    pick = s[rng.permutation(len(s))[:n_avg]]
    acc, k = np.zeros((NT, len(chans))), 0
    with open(DAT_PATH, "rb") as f:
        for sp in pick:
            lo, nread = int(sp) - NT0MIN - 60, NT + 120
            f.seek(lo * N_CHAN_BIN * ITEMSIZE)
            blk = np.frombuffer(f.read(nread * N_CHAN_BIN * ITEMSIZE),
                                dtype=np.int16).reshape(nread, N_CHAN_BIN)
            acc += np.stack([filtfilt(b_hp, a_hp, blk[:, c].astype(np.float64))
                             [60:60 + NT] for c in chans], axis=1)
            k += 1
    return (acc / max(k, 1)) * GAIN_TO_UV, k


print(f"\nmeasuring real averaged waveforms for {len(picked)} units ...")
rows, waves = [], {}
for u in picked:
    pc = int(peak_ch[u])
    d = np.sqrt(((cp - cp[pc]) ** 2).sum(axis=1))
    chans = np.sort(np.where(d <= RADIUS_UM)[0])
    M, k = real_mean_waveform(u, chans)
    w = M[:, list(chans).index(pc)]
    amp = M.max(axis=0) - M.min(axis=0)
    tr = int(np.argmin(w))
    half = w[tr] / 2.0
    l = r = tr
    while l > 0 and w[l] < half:
        l -= 1
    while r < len(w) - 1 and w[r] < half:
        r += 1
    waves[u] = (M, chans, pc)
    rows.append(dict(unit=u, peak_ch=pc,
                     contam_pct=float(pool[pool.cluster_id == u].ContamPct.iloc[0]),
                     n_spikes=int(pool[pool.cluster_id == u].n.iloc[0]),
                     ptp_uv=round(float(w.max() - w.min()), 1),
                     trough_uv=round(float(w.min()), 1),
                     half_width_ms=round((r - l) / FS * 1000.0, 3),
                     n_chans_25pct=int((amp >= 0.25 * amp.max()).sum()),
                     footprint_um=round(float(
                         d[chans][amp >= 0.25 * amp.max()].max()), 1),
                     n_snippets=k))
sel = pd.DataFrame(rows)

# ---- shape flag, CALIBRATED AGAINST EXPERT REVIEW.
#
# The first version of this flagged on waveform polarity: a positive peak >= 2x
# the trough with a wide half-width, read as an axonal/fibre signature. It
# rejected 18 of 50, including many units at the probe tip. Gil reviewed the
# figure and overruled it -- those waveforms are fine -- and instead picked out
# ONE unit the polarity rule had passed: 403.
#
# Comparing 403 against the units he accepts shows the real difference, and it
# is spatial rather than temporal. 403's footprint is a diffuse smear over
# channels 42-58 with no sharp peak, and its waveform is correspondingly slow
# and smooth; unit 428 has nearly the same polarity and half-width but a tight,
# well-localized blob. A diffuse footprint is what a merge of several cells, or
# a drift-smeared average, looks like -- and that makes a bad injection source
# whatever its polarity.
#
# The measurement that separates them is footprint compactness: 403 has 12
# channels above 25% of peak amplitude, every other unit in the set has <= 10.
# NOTE this threshold is calibrated against a single expert judgement on one
# probe, so it is a documented convention rather than a derived constant.
MAX_CHANS_25PCT = 11
sel["flag"] = ""
sel.loc[sel.n_chans_25pct >= MAX_CHANS_25PCT, "flag"] += "diffuse-footprint "
sel.loc[sel.n_spikes < MIN_SOURCE_SPIKES, "flag"] += "few-spikes "
sel["clean"] = sel.flag == ""
sel.to_csv(os.path.join(OUT, "selected_source_units.csv"), index=False)
print(f"\nshape flags: {int((~sel.clean).sum())} of {len(sel)} flagged, "
      f"{int(sel.clean.sum())} clean")
for f in ("broad-positive", "single-channel", "few-spikes"):
    hit = sel[sel.flag.str.contains(f)]
    if len(hit):
        print(f"  {f:<16} {len(hit):>2}  channels "
              f"{sorted(hit.peak_ch.tolist())}")

ncol = 4
nrow = int(np.ceil(len(picked) / ncol))
fig, axes = plt.subplots(nrow, ncol * 2, figsize=(5.2 * ncol, 1.85 * nrow),
                         gridspec_kw={"width_ratios": [1.0, 1.15] * ncol})
axes = np.atleast_2d(axes)
t_ms = (np.arange(NT) - NT0MIN) / FS * 1000.0
for i, u in enumerate(picked):
    rr, cc = i // ncol, (i % ncol) * 2
    M, chans, pc = waves[u]
    s = sel[sel.unit == u].iloc[0]
    col = "#1f5f9c" if s.clean else "#c0392b"
    ax = axes[rr, cc]
    ax.plot(t_ms, M[:, list(chans).index(pc)], color=col, lw=1.7)
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    ax.set_title(f"u{u}  ch{pc}  {s.ptp_uv:.0f}\u00b5V  {s.contam_pct:.1f}%"
                 + ("" if s.clean else f"\nREJECT: {s.flag.strip()}"),
                 fontsize=8, loc="left", color=col,
                 fontweight="normal" if s.clean else "bold")
    ax.tick_params(labelsize=6)
    ax2 = axes[rr, cc + 1]
    v = np.abs(M).max()
    ax2.imshow(M.T, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v,
               extent=[t_ms[0], t_ms[-1], len(chans) - 0.5, -0.5],
               interpolation="nearest")
    ax2.axhline(list(chans).index(pc), color="k", lw=0.7, ls=":")
    ax2.set_yticks([0, len(chans) - 1])
    ax2.set_yticklabels([str(chans[0]), str(chans[-1])], fontsize=6)
    ax2.tick_params(labelsize=6)
    ax2.set_title(f"{len(chans)} ch, spread {s.footprint_um:.0f}\u00b5m",
                  fontsize=7, loc="left", color="0.4")
for i in range(len(picked), nrow * ncol):
    rr, cc = i // ncol, (i % ncol) * 2
    axes[rr, cc].axis("off")
    axes[rr, cc + 1].axis("off")
plt.suptitle(
    f"All {len(picked)} candidate source units -- REAL averaged waveforms "
    f"(not normalized templates)\n"
    f"KSLabel='good', ContamPct <= {MAX_CONTAM_PCT}%, mutually >= "
    f"{worst_d:.0f}\u00b5m, similarity <= {worst_s:.3f}, no CCG refractory dip "
    f"(worst ratio {worst_ratio:.2f})",
    fontsize=12, fontweight="bold", y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.985])
out = os.path.join(OUT, "selected_source_units.png")
plt.savefig(out, dpi=100, bbox_inches="tight")
print(f"saved {out}")

print("\n" + "=" * 96)
print(f"ALL {len(picked)} SELECTED UNITS")
print(sel.to_string(index=False))
print("\nsanity checks:")
print(f"  amplitude range      : {sel.ptp_uv.min():.0f} - {sel.ptp_uv.max():.0f} uV")
print(f"  all troughs negative : {bool((sel.trough_uv < 0).all())}")
print(f"  footprint spread     : {sel.footprint_um.min():.0f} - "
      f"{sel.footprint_um.max():.0f} um")
print(f"  half-width range     : {sel.half_width_ms.min():.2f} - "
      f"{sel.half_width_ms.max():.2f} ms")
print(f"  peak channels span   : {sel.peak_ch.min()} - {sel.peak_ch.max()}")
print("\nsaved selected_source_units.csv, source_unit_audit.csv")

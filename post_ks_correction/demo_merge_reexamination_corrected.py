"""
STAGE 2: re-examine merge candidates with the corrected tooling.

WHY NOW: the footprint-gated scan (stage 1) plus its attribution breakdown
handed us a clean, empirically-identified pair to test. For unit 408, unit
412 accounted for 162 of 311 detections and the footprint gate removed only
2 of them -- 408 and 412 are footprint-indistinguishable. Either they are the
same neuron split in two, or pillar 1c cannot separate them. That is exactly
a merge question. Unit 440/439 is carried along as a KNOWN NEGATIVE control:
the same gate removed 470 of 470, so that pair must come out "do not merge".

The research log lists re-examining session 20260901_085606's candidates
(332-333, 306-305, 303-306) here too. Those are deliberately NOT rerun: that
session's curation labels were flagged as untrustworthy, which is the reason
the project moved sessions in the first place, so re-deciding merges from its
labels would rest on the same bad foundation. The pairs tested here are from
the trusted session and were found from the data rather than from a label.

THE CRITERIA, as specified for this project:
  1. similar SHAPE          -- waveform correlation on the shared peak channel
  2. similar LOCATION+SPREAD -- footprint compared three ways: cosine
     similarity, Pearson r across channels (cosine alone was shown in section
     5m to overstate agreement, because all-positive amplitude vectors are
     never far apart in angle), and the AUC with which footprint separates
     the two units' real spikes
  3. refractory period      -- VETO ONLY. This can disprove a merge and can
     never prove one. A merge is never asserted because a refractory test
     came back non-significant; section 5l is the correction that established
     this and it is honoured here explicitly.
  4. drift alternative      -- if footprints differ, do the two units occupy
     separate time windows (drift) rather than coexisting?

The refractory check uses a jitter permutation null (section 5k), not a
Poisson null, because the two units can have very different firing rates.

Usage: python demo_merge_reexamination_corrected.py
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

from wavelet_features import (make_morlet, calibrate_reference_phase,
                               coarse_then_fine_shift, select_probe_frequency)
from spatial_footprint import (multichannel_template, spatial_energy_vector,
                                footprint_similarity)

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
DAT_PATH = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\20260916_110311.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN, FS, NT0MIN, N, ITEMSIZE = 384, 30000.0, 20, 61, 2
SEARCH_RADIUS, N_CYCLES, RADIUS_UM = 25, 3.0, 60.0
N_SPIKES = 250
REFRACTORY_MS, CCG_WIN_MS, CCG_BIN_MS = 1.5, 25.0, 0.5
JITTER_MS, N_PERM = 10.0, 2000

PAIRS = [(408, 412), (440, 439)]

templates = np.load(KS + r"\templates.npy")
spike_times = np.load(KS + r"\spike_times.npy").ravel()
spike_clusters = np.load(KS + r"\spike_clusters.npy").ravel()
channel_positions = np.load(KS + r"\channel_positions.npy")
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")
TOTAL = os.path.getsize(DAT_PATH) // (N_CHAN_BIN * ITEMSIZE)


def read_block(center, pad=250, filt_buf=60):
    s0 = int(center) - pad - filt_buf
    nr = (int(center) + pad + filt_buf) - s0
    if s0 < 0 or s0 + nr > TOTAL:
        return None, filt_buf
    with open(DAT_PATH, "rb") as f:
        f.seek(s0 * N_CHAN_BIN * ITEMSIZE)
        raw = f.read(nr * N_CHAN_BIN * ITEMSIZE)
    return np.frombuffer(raw, dtype=np.int16).reshape(nr, N_CHAN_BIN), filt_buf


def filt_channel(block, ch, fb):
    return filtfilt(b_hp, a_hp, block[:, ch].astype(np.float64))[fb:-fb] * GAIN_TO_UV


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def ccg(t1, t2, win_ms=CCG_WIN_MS, bin_ms=CCG_BIN_MS):
    """Cross-correlogram counts, t1/t2 in samples."""
    win = win_ms / 1000.0 * FS
    nb = int(2 * win_ms / bin_ms)
    edges = np.linspace(-win, win, nb + 1)
    out = np.zeros(nb)
    lo = np.searchsorted(t2, t1 - win, side="left")
    hi = np.searchsorted(t2, t1 + win, side="right")
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            out += np.histogram(t2[a:b] - t1[i], bins=edges)[0]
    return out, edges


def central_count(t1, t2, ms=REFRACTORY_MS):
    """Number of cross-pairs falling inside +/- ms of each other."""
    w = ms / 1000.0 * FS
    lo = np.searchsorted(t2, t1 - w, side="left")
    hi = np.searchsorted(t2, t1 + w, side="right")
    return int((hi - lo).sum())


rows = []
for uid, oid in PAIRS:
    print(f"\n{'='*78}")
    print(f"PAIR {uid} vs {oid}")

    # ---------- criterion 1: shape on the shared peak channel ----------
    ta, tb = templates[uid], templates[oid]
    ampa = ta.max(0) - ta.min(0)
    ampb = tb.max(0) - tb.min(0)
    pca, pcb = int(np.argmax(ampa)), int(np.argmax(ampb))
    dist = float(np.sqrt(((channel_positions[pca] - channel_positions[pcb]) ** 2).sum()))
    shape_r = float(pearsonr(ta[:, pca], tb[:, pca])[0])
    print(f"  peak channels: {uid}->ch{pca}, {oid}->ch{pcb}  ({dist:.0f} um apart)")
    print(f"  [1] SHAPE  waveform r on ch{pca} = {shape_r:.3f}")

    # ---------- criterion 2: location + spread ----------
    mct = multichannel_template(templates, uid, channel_positions,
                                radius_um=RADIUS_UM, min_amplitude_fraction=0.0)
    CH = list(mct["channels"])
    fa = ampa[CH] / np.linalg.norm(ampa[CH])
    fb_ = ampb[CH] / np.linalg.norm(ampb[CH])
    cos_t = float(np.dot(fa, fb_))
    r_t, p_t = pearsonr(ampa[CH], ampb[CH])
    print(f"  [2] LOCATION over {len(CH)} channels {CH}")
    print(f"      template footprint cosine = {cos_t:.3f}")
    print(f"      template footprint Pearson r = {r_t:.3f} (p={p_t:.3g})")

    # footprint measured on REAL spikes of each unit, and its separating power
    f0, _, _ = select_probe_frequency(templates[uid][:, pca], FS, n_cycles=N_CYCLES,
                                      nt0min=NT0MIN, criterion="timing")
    _, psi = make_morlet(f0, FS, n_cycles=N_CYCLES)
    TM = templates[uid][:, pca]
    refph = calibrate_reference_phase(TM, psi, FS, align_index=NT0MIN)
    FP_EXP = mct["per_channel_amplitude"] / np.linalg.norm(mct["per_channel_amplitude"])

    def fp_of(sample):
        block, fbuf = read_block(sample)
        if block is None:
            return None
        pad = 250
        trace = filt_channel(block, pca, fbuf)
        r = coarse_then_fine_shift(trace, pad, TM, psi, f0, FS, reference_phase=refph,
                                   search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
        tc = pad + r["total_shift"]
        lo, hi = int(np.floor(tc - NT0MIN)) - 2, int(np.ceil(tc - NT0MIN + N)) + 2
        if lo < 0 or hi > len(trace):
            return None
        grid = tc - NT0MIN + np.arange(N)
        snips = []
        for ch in CH:
            tr = filt_channel(block, ch, fbuf)
            ip = interp1d(np.arange(lo, hi), tr[lo:hi], kind="cubic",
                          bounds_error=False, fill_value=0.0)
            snips.append(ip(grid))
        return footprint_similarity(spatial_energy_vector(np.stack(snips, axis=1)), FP_EXP)

    rng = np.random.default_rng(0)
    sims = {}
    for which, u in (("self", uid), ("other", oid)):
        st_u = np.sort(spike_times[spike_clusters == u])
        pick = st_u[rng.choice(len(st_u), size=min(N_SPIKES, len(st_u)), replace=False)]
        vals = [v for v in (fp_of(int(s)) for s in pick) if v is not None]
        sims[which] = np.array(vals)
    fp_auc = auc(sims["self"], sims["other"])
    print(f"      real-spike footprint sim: {uid} mean {sims['self'].mean():.3f}, "
          f"{oid} mean {sims['other'].mean():.3f}")
    print(f"      AUC separating them by footprint = {fp_auc:.3f}   "
          f"({'indistinguishable' if fp_auc < 0.65 else 'separable'})")

    # ---------- criterion 3: refractory, VETO ONLY ----------
    t1 = np.sort(spike_times[spike_clusters == uid]).astype(np.int64)
    t2 = np.sort(spike_times[spike_clusters == oid]).astype(np.int64)
    obs = central_count(t1, t2)
    jw = int(JITTER_MS / 1000.0 * FS)
    null = np.empty(N_PERM)
    for k in range(N_PERM):
        t2j = np.sort(t2 + rng.integers(-jw, jw + 1, size=len(t2)))
        null[k] = central_count(t1, t2j)
    mu, sd = null.mean(), null.std()
    z = (obs - mu) / sd if sd > 0 else np.nan
    p_excess = float((null >= obs).mean())
    p_deficit = float((null <= obs).mean())
    print(f"  [3] REFRACTORY (veto only) within +/-{REFRACTORY_MS} ms")
    print(f"      observed cross-pairs = {obs}, jitter null = {mu:.1f} +/- {sd:.1f}, z = {z:+.2f}")
    print(f"      p(excess) = {p_excess:.4f}   p(deficit) = {p_deficit:.4f}")
    veto = p_excess < 0.05
    if veto:
        print(f"      -> VETO: the two fire together at zero lag more than chance.")
        print(f"         One neuron cannot spike twice inside its refractory period,")
        print(f"         so these are two different cells. Merge is ruled OUT.")
    else:
        print(f"      -> no veto. This is NOT evidence for merging, only the")
        print(f"         absence of evidence against it.")

    # ---------- criterion 4: drift alternative ----------
    ov_lo = max(t1.min(), t2.min())
    ov_hi = min(t1.max(), t2.max())
    span = max(t1.max(), t2.max()) - min(t1.min(), t2.min())
    overlap_frac = float(max(0, ov_hi - ov_lo) / span) if span > 0 else 0.0
    print(f"  [4] DRIFT  active-window overlap = {overlap_frac:.1%} "
          f"({'coexist -- drift cannot explain a footprint difference' if overlap_frac > 0.5 else 'largely separate in time -- drift is plausible'})")

    # ---------- verdict ----------
    shape_ok = shape_r >= 0.90
    loc_ok = (r_t >= 0.90) and (fp_auc < 0.65)
    if veto:
        verdict = "DO NOT MERGE (refractory veto)"
    elif shape_ok and loc_ok:
        verdict = "MERGE CANDIDATE (shape + location both match, no veto)"
    elif shape_ok and not loc_ok:
        verdict = "DO NOT MERGE (shape matches but footprint separates them)"
    else:
        verdict = "DO NOT MERGE (shape differs)"
    print(f"  VERDICT: {verdict}")

    rows.append(dict(unit_a=uid, unit_b=oid, peak_ch_a=pca, peak_ch_b=pcb,
                     dist_um=round(dist, 1), shape_r=round(shape_r, 4),
                     fp_cosine=round(cos_t, 4), fp_pearson_r=round(float(r_t), 4),
                     fp_pearson_p=float(p_t), fp_auc=round(fp_auc, 4),
                     refr_observed=obs, refr_null_mean=round(float(mu), 2),
                     refr_z=round(float(z), 3), p_excess=p_excess, p_deficit=p_deficit,
                     refractory_veto=bool(veto),
                     time_overlap_frac=round(overlap_frac, 4), verdict=verdict))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "merge_reexamination_corrected.csv"), index=False)
print(f"\n{'='*78}\nsaved merge_reexamination_corrected.csv")
print(df[["unit_a", "unit_b", "shape_r", "fp_pearson_r", "fp_auc",
          "refractory_veto", "verdict"]].to_string(index=False))

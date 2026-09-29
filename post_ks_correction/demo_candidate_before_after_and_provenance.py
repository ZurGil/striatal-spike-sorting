"""
Deeper look at the missed-spike candidates from 5h, addressing three
things directly:

1. "Assigned to a different unit doesn't mean it SHOULD be" -- for each
   candidate that coincides with a spatially-neighboring unit's own real
   spike, show the raw snippet against BOTH templates (ours and the
   neighbor's), not just ours. If a candidate matches both templates
   almost equally well, and the two templates themselves look similar,
   that's the same signature as the 342-vs-347 burst-split case earlier
   in this project -- these might be one neuron Kilosort split into two
   clusters, not two genuinely separate neurons colliding.

2. More than one candidate per unit -- shows the top 2 per unit, not just
   the single best.

3. "What was the score BEFORE wavelet correction, vs after" -- directly
   tests the jitter hypothesis. BEFORE = a plain, uncorrected matched-
   filter score at the exact sample the simple scan flagged (no whole-
   sample search, no phase correction -- as close as we can get to what a
   rigid, non-phase-tolerant matcher would see). AFTER = the full
   coarse+fine wavelet-aligned fit score, same as used throughout this
   project. A big BEFORE-to-AFTER jump supports "jitter caused this to be
   missed"; little to no jump means realignment isn't the explanation.

Usage: python demo_candidate_before_after_and_provenance.py
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavelet_features import make_morlet, wavelet_transform_at, calibrate_reference_phase, coarse_then_fine_shift
from nuisance_model import build_basis, fit_nuisance_prealigned
from noise_whitening import build_whitening_from_noise

VR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
DAT_PATH = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\20260901_085606.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN = 384
FS = 30000.0
NT0MIN = 20
N = 61
ITEMSIZE = 2
SEARCH_RADIUS = 25
N_CYCLES = 3.0
N_REF_SPIKES = 60
N_SCAN_SPIKES = 60
SCAN_HALF_WINDOW = 150
MARGIN = 20
GAP_SCAN_STEP = 2
TOP_N_CANDIDATES = 25
JITTER_WINDOW = 15
RADIUS_UM = 60.0
GOOD_UNITS = [303, 313, 245, 332, 306, 342]

templates = np.load(VR + r"\templates.npy")
spike_times = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()
channel_positions = np.load(VR + r"\channel_positions.npy")
n_templ = templates.shape[0]
peak_ch_all = np.argmax(templates.max(axis=1) - templates.min(axis=1), axis=1)
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")


def get_filtered(peak_ch, center_sample, pad, filt_buf=60):
    s0 = center_sample - pad - filt_buf
    s1 = center_sample + pad + filt_buf
    n_read = s1 - s0
    with open(DAT_PATH, "rb") as f:
        f.seek(int(s0) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read(n_read * N_CHAN_BIN * ITEMSIZE)
    block = np.frombuffer(raw, dtype=np.int16).reshape(n_read, N_CHAN_BIN)
    return filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64))[filt_buf:-filt_buf] * GAIN_TO_UV


def unit_pipeline(uid):
    templ_all = templates[uid]
    peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))
    template = templ_all[:, peak_ch]
    f0_scan = np.linspace(300, 4000, 50)
    pad_scan = int(np.ceil(N_CYCLES * FS / (2 * f0_scan.min()))) + 20
    trace = np.zeros(N + 2 * pad_scan)
    center = pad_scan + NT0MIN
    trace[center - NT0MIN: center - NT0MIN + N] = template
    mags = [abs(wavelet_transform_at(trace, make_morlet(f0, FS, N_CYCLES)[1], center)) for f0 in f0_scan]
    best_f0 = float(f0_scan[int(np.nanargmax(mags))])
    _, psi = make_morlet(best_f0, FS, n_cycles=N_CYCLES)
    ref_phase = calibrate_reference_phase(template, psi, FS, align_index=NT0MIN)
    s0_basis, _, q_basis = build_basis(template, dt=1.0)
    template_norm = template / (np.linalg.norm(template) + 1e-12)
    st = np.sort(spike_times[spike_clusters == uid])
    noise_start = int(st[0]) + 500000 if st[0] + 500000 < st[-1] else int(st[0]) - 500000
    noise_start = max(noise_start, 10000)
    with open(DAT_PATH, "rb") as f:
        f.seek(int(noise_start - 60) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read((60000 + 120) * N_CHAN_BIN * ITEMSIZE)
    block = np.frombuffer(raw, dtype=np.int16).reshape(60000 + 120, N_CHAN_BIN)
    noise_trace = filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64))[60:-60] * GAIN_TO_UV
    W, _, _ = build_whitening_from_noise(noise_trace, N, order=4)
    return dict(peak_ch=peak_ch, template=template, template_norm=template_norm, best_f0=best_f0,
                psi=psi, ref_phase=ref_phase, s0=s0_basis, q=q_basis, W=W, st=st)


def fit_before(peak_ch, sample, unit):
    """The SAME R2 fit used everywhere else (whitened amplitude+stretch
    fit), but on the UNCORRECTED snippet -- no coarse whole-sample search,
    no wavelet fine phase correction, just the raw window at the exact
    sample the simple scan flagged. As close as we can get to 'what a
    rigid, non-phase-tolerant matcher (like Kilosort's own matching
    pursuit) would see at this instant', on the SAME 0-1 R2 scale as the
    after-alignment number, so the two are directly comparable."""
    trace = get_filtered(peak_ch, sample, pad=100)
    window = trace[100 - NT0MIN: 100 - NT0MIN + N]
    fit = fit_nuisance_prealigned(window, unit["s0"], unit["q"], whitening_matrix=unit["W"])
    return fit["r_squared"]


def align_and_fit(peak_ch, sample, unit, pad=200):
    trace = get_filtered(peak_ch, sample, pad=pad)
    nominal_center = pad
    result = coarse_then_fine_shift(trace, nominal_center, unit["template"], unit["psi"], unit["best_f0"], FS,
                                     reference_phase=unit["ref_phase"], search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
    true_center = nominal_center + result["total_shift"]
    lo_int = int(np.floor(true_center - NT0MIN)) - 2
    hi_int = int(np.ceil(true_center - NT0MIN + N)) + 2
    if lo_int < 0 or hi_int > len(trace):
        return None
    interp = interp1d(np.arange(lo_int, hi_int), trace[lo_int:hi_int], kind="cubic",
                       bounds_error=False, fill_value=0.0)
    snippet = interp(true_center - NT0MIN + np.arange(N))
    fit = fit_nuisance_prealigned(snippet, unit["s0"], unit["q"], whitening_matrix=unit["W"])
    return dict(snippet=snippet, r_squared=fit["r_squared"], a=fit["a"])


def find_spatial_neighbor_match(uid, sample):
    my_pos = channel_positions[peak_ch_all[uid]]
    mask = np.abs(spike_times - sample) <= JITTER_WINDOW
    other = sorted(set(spike_clusters[mask].tolist()) - {uid})
    other = [oc for oc in other if oc < n_templ]
    best = None
    for oc in other:
        d = np.sqrt(((channel_positions[peak_ch_all[oc]] - my_pos) ** 2).sum())
        if d <= RADIUS_UM and (best is None or d < best[1]):
            best = (oc, d)
    return best


unit_cache = {uid: unit_pipeline(uid) for uid in GOOD_UNITS}

all_rows = []
plot_specs = []

for uid in GOOD_UNITS:
    U = unit_cache[uid]
    st = U["st"]
    detected_set = set(st.tolist())

    rng = np.random.default_rng(7)
    ref_idx = rng.choice(len(st), size=min(N_REF_SPIKES, len(st)), replace=False)
    ref_r2 = np.array([r["r_squared"] for r in
                        (align_and_fit(U["peak_ch"], int(s), U) for s in st[ref_idx]) if r is not None])
    bar_p25 = np.percentile(ref_r2, 25)

    rng2 = np.random.default_rng(11)
    scan_idx = rng2.choice(len(st), size=min(N_SCAN_SPIKES, len(st)), replace=False)
    scan_spikes = st[scan_idx]

    coarse_candidates = []
    for s in scan_spikes:
        lo, hi = int(s) - SCAN_HALF_WINDOW, int(s) + SCAN_HALF_WINDOW
        trace = get_filtered(U["peak_ch"], int(s), pad=SCAN_HALF_WINDOW + 60)
        center_in_trace = SCAN_HALF_WINDOW + 60
        local = []
        for c in range(lo, hi, GAP_SCAN_STEP):
            offset = c - int(s)
            lo_w = center_in_trace + offset - NT0MIN
            window = trace[lo_w: lo_w + N]
            if len(window) != N:
                continue
            local.append((c, np.dot(window, U["template_norm"])))
        if len(local) < 3:
            continue
        samples_arr = np.array([c for c, _ in local])
        scores_arr = np.array([sc for _, sc in local])
        for i in range(1, len(scores_arr) - 1):
            if scores_arr[i] >= scores_arr[i - 1] and scores_arr[i] >= scores_arr[i + 1]:
                cand_sample = int(samples_arr[i])
                if min(abs(cand_sample - ss) for ss in detected_set) > MARGIN:
                    coarse_candidates.append((cand_sample, scores_arr[i]))
    coarse_candidates = sorted(set(coarse_candidates), key=lambda x: -x[1])[:TOP_N_CANDIDATES]

    scored = []
    for cand_sample, coarse_score in coarse_candidates:
        after = align_and_fit(U["peak_ch"], cand_sample, U)
        if after is None:
            continue
        before_score = fit_before(U["peak_ch"], cand_sample, U)
        neighbor = find_spatial_neighbor_match(uid, cand_sample)
        scored.append(dict(uid=uid, sample=cand_sample, r2_after=after["r_squared"], a=after["a"],
                            snippet=after["snippet"], score_before=before_score,
                            neighbor_uid=neighbor[0] if neighbor else None,
                            neighbor_dist_um=neighbor[1] if neighbor else None))
        all_rows.append(dict(uid=uid, sample=cand_sample, r2_after=after["r_squared"],
                              score_before_uncorrected=before_score,
                              above_p25=after["r_squared"] >= bar_p25,
                              neighbor_uid=neighbor[0] if neighbor else None,
                              neighbor_dist_um=round(neighbor[1], 1) if neighbor else None))

    scored_df = pd.DataFrame(scored).sort_values("r2_after", ascending=False)
    top2 = scored_df.head(2)
    for _, row in top2.iterrows():
        spec = dict(uid=uid, sample=int(row["sample"]), r2_after=row["r2_after"], a=row["a"],
                    snippet=row["snippet"], score_before=row["score_before"], ref_p25=bar_p25)
        if pd.notna(row["neighbor_uid"]):
            nb_uid = int(row["neighbor_uid"])
            if nb_uid not in unit_cache:
                unit_cache[nb_uid] = unit_pipeline(nb_uid)
            NB = unit_cache[nb_uid]
            nb_after = align_and_fit(NB["peak_ch"], int(row["sample"]), NB)
            nb_before = fit_before(NB["peak_ch"], int(row["sample"]), NB)
            spec.update(neighbor_uid=nb_uid, neighbor_dist_um=row["neighbor_dist_um"],
                        neighbor_template=NB["template"], neighbor_r2_after=nb_after["r_squared"] if nb_after else np.nan,
                        neighbor_a=nb_after["a"] if nb_after else np.nan, neighbor_score_before=nb_before,
                        shape_corr_templates=float(np.corrcoef(U["template"], NB["template"])[0, 1]))
        plot_specs.append(spec)

all_df = pd.DataFrame(all_rows)
all_df.to_csv(os.path.join(OUT, "candidate_before_after_provenance_full.csv"), index=False)
print(f"total candidates scored: {len(all_df)}")
print(f"with a spatially-relevant neighbor match: {all_df['neighbor_uid'].notna().sum()}")
print(f"\nmean R2 BEFORE correction (uncorrected alignment) = {all_df['score_before_uncorrected'].mean():.3f}")
print(f"mean R2 AFTER correction (wavelet-aligned)        = {all_df['r2_after'].mean():.3f}")

# ============================================================
n_show = len(plot_specs)
fig, axes = plt.subplots(1, n_show, figsize=(4.6 * n_show, 5), squeeze=False)
t_ms = (np.arange(N) - NT0MIN) / FS * 1000

for col, spec in enumerate(plot_specs):
    ax = axes[0, col]
    ax.plot(t_ms, spec["snippet"], color="#333", lw=1.7, label="candidate raw snippet")
    ax.plot(t_ms, unit_cache[spec["uid"]]["template"] * spec["a"], color="#1f77b4", lw=1.4, ls="--",
            label=f"unit {spec['uid']} template*a={spec['a']:.1f}\n(R2 before={spec['score_before']:.2f}, R2 after={spec['r2_after']:.2f})")
    title = f"unit {spec['uid']}, sample {spec['sample']}\nR2 after={spec['r2_after']:.2f} (ref p25={spec['ref_p25']:.2f})"
    if "neighbor_uid" in spec:
        ax.plot(t_ms, spec["neighbor_template"] * spec["neighbor_a"], color="#d62728", lw=1.4, ls=":",
                label=f"unit {spec['neighbor_uid']} template*a={spec['neighbor_a']:.1f}\n"
                      f"(R2 before={spec['neighbor_score_before']:.2f}, R2 after={spec['neighbor_r2_after']:.2f})")
        title += (f"\nvs neighbor unit {spec['neighbor_uid']} ({spec['neighbor_dist_um']:.0f}um away, "
                  f"template shape corr={spec['shape_corr_templates']:.2f})")
    ax.axhline(0, color="grey", lw=0.4)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time (ms)")
    if col == 0:
        ax.set_ylabel("uV")
    ax.legend(fontsize=6.5, loc="upper right")

plt.suptitle("Candidates: raw snippet vs. BOTH templates, before/after wavelet correction", fontsize=13, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(OUT, "missed_spike_search_02_before_after_provenance.png")
plt.savefig(out_path, dpi=110, bbox_inches="tight")
print("\nsaved", out_path)

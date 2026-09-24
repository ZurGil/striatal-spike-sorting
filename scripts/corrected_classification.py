"""
Corrected unit classification: combines bombcell's shape-based criteria (noise via
spatial decay / peak-trough counts / duration / baseline flatness; non-somatic via
peak-trough amplitude ratio) with our own already-validated `violation_ratio` (ACG-
shoulder-normalized, acg_tools.py) in place of bombcell's RPV check, which this
session's investigation found to be broken here (saturates at ~1.0 for 82% of MUA
units, confirmed wrong against known examples, and doesn't improve with bombcell's
alternate llobet/ibl_sliding methods either -- see conversation).

Also treats non-somatic as informative, not disqualifying (confirmed in-session: it's
driven purely by mainPeakToTroughRatio>0.8, which flags real units including the
likely-TAN unit 30 and our clean reference units 285/295 -- see conversation).

Uses the gain-corrected bombcell run for amplitude/SNR-dependent checks (those are the
only MUA criteria affected by gain_to_uV) and falls back to the first run's qMetrics
for everything else if the corrected run isn't available yet.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from striatal_agent.acg_tools import violation_ratio
from striatal_agent.raw_io import ChannelGeometry, footprint_channels
from striatal_agent.structural_score import footprint_concentration_ratio, footprint_flatness_ratio, peak_trough_width_ms
from striatal_agent.config import AgentConfig

KS_DIR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\kilosort_original"
QM_PATH_GAINCORRECTED = r"D:\Gil\spike_sorting_agent\outputs\bombcell_results_gaincorrected\templates._bc_qMetrics.csv"
QM_PATH_FALLBACK = r"D:\Gil\spike_sorting_agent\outputs\bombcell_results\templates._bc_qMetrics.csv"
OUT_CSV = r"D:\Gil\spike_sorting_agent\outputs\corrected_classification.csv"

NOISE_PARAMS = dict(
    maxNPeaks=2, maxNTroughs=1, maxScndPeakToTroughRatio_noise=0.8,
    minSpatialDecaySlopeExp=0.01, maxSpatialDecaySlopeExp=0.1,
)
NON_SOMA_PARAMS = dict(maxMainPeakToTroughRatio_nonSomatic=0.8)
MUA_PARAMS = dict(
    maxPercSpikesMissing=20, minNumSpikes=300, minPresenceRatio=0.7,
    # minAmplitude raised from 40 to 50 (see conversation): the 326-unit manual ground
    # truth's lowest-amplitude confirmed GOOD unit sits at 54.59uV, so 40 was letting
    # real noise/mua through as a false GOOD. 50 catches 33 noise + 5 mua (vs. 16+3 at
    # 40) with zero GOOD units lost -- 55 would be the exact ground-truth boundary but
    # costs one GOOD unit (54.59uV) as a false positive, so 50 keeps a safety margin.
    minAmplitude=50, minSNR=5,
)
VIOLATION_RATIO_MUA_THRESHOLD = 0.4  # our metric's scale: 0.06-0.22 clean, 0.59-0.94 known-bad (calibrated earlier this session)


# too_many_peaks/too_many_troughs/2nd_peak_too_big: proven this session to false-flag
# real, high-SNR, clean-ACG, normal-footprint neurons (unit 59 originally via the count
# rule; then 8 more -- 134,34,58,244,14,225,74,82 -- via 2nd_peak_too_big, several
# bypassing the old count-only corroboration check entirely because they also carried
# 2nd_peak_too_big). Decision: these three are DISPLAY-ONLY from here on -- computed
# and shown per-unit, never used to decide a label, no corroboration exception either.
# duration_too_short/long and waveformBaselineFlatness were never independently
# validated and fire on almost nothing in this dataset (2 and 1 units of 401) --
# removed from the pipeline entirely rather than carry untested, low-value risk.
DISPLAY_ONLY_NOISE_REASONS = {"too_many_peaks", "too_many_troughs", "2nd_peak_too_big"}


def classify_noise_raw(r):
    """Every bombcell shape criterion that fired, unfiltered."""
    reasons = []
    if pd.isna(r["nPeaks"]):
        reasons.append("nPeaks_nan")
    if r["nPeaks"] > NOISE_PARAMS["maxNPeaks"]:
        reasons.append("too_many_peaks")
    if r["nTroughs"] > NOISE_PARAMS["maxNTroughs"]:
        reasons.append("too_many_troughs")
    if r["scndPeakToTroughRatio"] > NOISE_PARAMS["maxScndPeakToTroughRatio_noise"]:
        reasons.append("2nd_peak_too_big")
    if r["spatialDecaySlope"] < NOISE_PARAMS["minSpatialDecaySlopeExp"] or r["spatialDecaySlope"] > NOISE_PARAMS["maxSpatialDecaySlopeExp"]:
        reasons.append("spatial_decay_abnormal")
    return reasons


def classify_noise(r):
    """Only spatial_decay_abnormal (validated this session: all 10 units it flags show
    the unambiguous artifact signature -- huge amplitude, near-zero rate, single-channel
    isolation) and nPeaks_nan (a data-integrity check, not a shape judgment) can
    independently make a unit NOISE. The three shape-counting/ratio rules never decide
    anything now -- see DISPLAY_ONLY_NOISE_REASONS above."""
    reasons = classify_noise_raw(r)
    display_only = [x for x in reasons if x in DISPLAY_ONLY_NOISE_REASONS]
    deciding = [x for x in reasons if x not in DISPLAY_ONLY_NOISE_REASONS]
    return deciding, display_only


def classify_non_soma(r):
    return r["mainPeakToTroughRatio"] > NON_SOMA_PARAMS["maxMainPeakToTroughRatio_nonSomatic"]


def main():
    qm_path = QM_PATH_GAINCORRECTED if os.path.exists(QM_PATH_GAINCORRECTED) else QM_PATH_FALLBACK
    using_gain_corrected = qm_path == QM_PATH_GAINCORRECTED
    print(f"Using qMetrics from: {qm_path} (gain-corrected: {using_gain_corrected})")
    qm = pd.read_csv(qm_path).set_index("phy_clusterID")

    spike_times = np.load(os.path.join(KS_DIR, "spike_times.npy")).ravel()
    spike_clusters = np.load(os.path.join(KS_DIR, "spike_clusters.npy")).ravel()
    templates = np.load(os.path.join(KS_DIR, "templates.npy"))
    geometry = ChannelGeometry.from_json(
        r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\20260901_085606_custom_neuropixels_map.json")
    cfg = AgentConfig()

    rows = []
    for u in qm.index:
        r = qm.loc[u]
        is_non_soma = classify_non_soma(r)

        st = spike_times[spike_clusters == u]
        if len(st) >= 2:
            vratio, refr_count, expected = violation_ratio(st, cfg.fs, cfg)
        else:
            vratio = np.nan

        chans, peak_chan = footprint_channels(templates, u, cfg.footprint_n_channels,
                                               geometry, cfg.footprint_channel_radius_um)
        fp_conc = footprint_concentration_ratio(templates, u, chans)
        fp_flat = footprint_flatness_ratio(templates, u, chans)
        width_ms = peak_trough_width_ms(templates, u, peak_chan, cfg.fs)

        noise_reasons, overridden_count_reasons = classify_noise(r)
        is_noise = len(noise_reasons) > 0

        mua_reasons = []
        if fp_conc < cfg.footprint_concentration_flag_threshold:
            mua_reasons.append("footprint_cliff")
        if fp_flat >= cfg.footprint_flatness_flag_threshold:
            mua_reasons.append("footprint_flat")
        if width_ms >= cfg.width_outlier_threshold_ms:
            mua_reasons.append("width_outlier")
        if r["percentageSpikesMissing_gaussian"] > MUA_PARAMS["maxPercSpikesMissing"]:
            mua_reasons.append("spikesMissing")
        if r["nSpikes"] < MUA_PARAMS["minNumSpikes"]:
            mua_reasons.append("nSpikes")
        if r["presenceRatio"] < MUA_PARAMS["minPresenceRatio"]:
            mua_reasons.append("presenceRatio")
        if not pd.isna(r["rawAmplitude"]) and r["rawAmplitude"] < MUA_PARAMS["minAmplitude"]:
            mua_reasons.append("amplitude")
        if not pd.isna(r["signalToNoiseRatio"]) and r["signalToNoiseRatio"] < MUA_PARAMS["minSNR"]:
            mua_reasons.append("SNR")
        if not np.isnan(vratio) and vratio > VIOLATION_RATIO_MUA_THRESHOLD:
            mua_reasons.append("violation_ratio")
        is_mua = len(mua_reasons) > 0

        if is_noise:
            label = "NOISE"
        elif is_mua:
            label = "MUA"
        else:
            label = "GOOD"
        if is_non_soma and not is_noise:
            label += "+NON-SOMA"

        top4 = chans[np.argsort(-(templates[u][:, chans].max(axis=0) - templates[u][:, chans].min(axis=0)))[:4]]
        ks_template = templates[u][:, top4]
        mean_rate_hz = len(st) / ((st.max() - st.min()) / cfg.fs) if len(st) > 1 else 0.0

        rows.append(dict(
            unit_id=int(u), label=label, is_noise=is_noise, is_non_soma=is_non_soma,
            is_mua=is_mua, mua_reasons=",".join(mua_reasons), noise_reasons=",".join(noise_reasons),
            overridden_count_reasons=",".join(overridden_count_reasons),
            violation_ratio=vratio, footprint_concentration_ratio=fp_conc,
            footprint_flatness_ratio=fp_flat,
            peak_trough_width_ms=width_ms, n_spikes=int(r["nSpikes"]), mean_rate_hz=mean_rate_hz,
            waveformDuration_peakTrough=r["waveformDuration_peakTrough"],
            mainPeakToTroughRatio=r["mainPeakToTroughRatio"],
            rawAmplitude=r["rawAmplitude"], signalToNoiseRatio=r["signalToNoiseRatio"],
            bombcell_original_label=r["bc_unitType"], peak_channel=int(peak_chan),
            ks_template=ks_template.tolist(),
        ))

    df = pd.DataFrame(rows)
    df.drop(columns=["ks_template"]).to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} units to {OUT_CSV}")

    import json
    out_json = OUT_CSV.replace(".csv", "_with_waveforms.json")
    def _default(o):
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return str(o)

    with open(out_json, "w") as f:
        json.dump(rows, f, default=_default)
    print(f"Wrote {out_json} (includes waveforms)")

    print("\n=== CORRECTED CLASSIFICATION SUMMARY ===")
    print(df["label"].value_counts().to_string())

    print("\n=== comparison: corrected 'core' label vs bombcell original ===")
    core = df["label"].str.replace("+NON-SOMA", "", regex=False)
    print(pd.crosstab(df["bombcell_original_label"], core))

    print("\n=== how many units flip from bombcell MUA -> corrected GOOD (i.e. only RPV was the problem) ===")
    flipped = df[(df["bombcell_original_label"] == "MUA") & (core == "GOOD")]
    print(f"{len(flipped)} units")

    print("\n=== how many units had their peak/trough-count noise verdict overridden (strong on everything else) ===")
    overridden = df[df["overridden_count_reasons"] != ""]
    print(f"{len(overridden)} units, now labeled: {overridden['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()

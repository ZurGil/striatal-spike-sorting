"""
Session-parameterized version of D:\\Gil\\spike_sorting_agent\\scripts\\corrected_classification.py
(that original is hardcoded to 20260901_085606 -- kept as-is, don't edit it, it's the
calibration reference). This logic is otherwise session-independent given bombcell qMetrics
+ kilosort_original + channel map. See WORKFLOW.md 1f-pre for the full pipeline this fits into.

Prerequisite: run_bombcell_session.py must have already produced
outputs/bombcell_results_<session_id>/templates._bc_qMetrics.csv for this session.

Usage: python corrected_classification_session.py <session_id> <rec_root>
  <rec_root> e.g. F:\\Gil\\Shamir\\20260911_100049.rec
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\Gil\spike_sorting_agent")
from striatal_agent.acg_tools import violation_ratio
from striatal_agent.raw_io import ChannelGeometry, footprint_channels
from striatal_agent.structural_score import footprint_concentration_ratio, footprint_flatness_ratio, peak_trough_width_ms
from striatal_agent.config import AgentConfig

SESSION_ID = sys.argv[1]
REC_ROOT = sys.argv[2]

KS_DIR = rf"{REC_ROOT}\{SESSION_ID}.kilosort\kilosort4\kilosort_original"
QM_PATH = rf"D:\Gil\spike_sorting_agent\outputs\bombcell_results_{SESSION_ID}\templates._bc_qMetrics.csv"
CHANNEL_MAP_JSON = rf"{REC_ROOT}\{SESSION_ID}.kilosort\{SESSION_ID}_custom_neuropixels_map.json"
OUT_CSV = rf"D:\Gil\spike_sorting_agent\outputs\corrected_classification_{SESSION_ID}.csv"

NOISE_PARAMS = dict(
    maxNPeaks=2, maxNTroughs=1, maxScndPeakToTroughRatio_noise=0.8,
    minSpatialDecaySlopeExp=0.01, maxSpatialDecaySlopeExp=0.1,
)
NON_SOMA_PARAMS = dict(maxMainPeakToTroughRatio_nonSomatic=0.8)
MUA_PARAMS = dict(
    maxPercSpikesMissing=20, minNumSpikes=300, minPresenceRatio=0.7,
    minAmplitude=50, minSNR=5,
)
VIOLATION_RATIO_MUA_THRESHOLD = 0.4
DISPLAY_ONLY_NOISE_REASONS = {"too_many_peaks", "too_many_troughs", "2nd_peak_too_big"}


def classify_noise_raw(r):
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
    reasons = classify_noise_raw(r)
    display_only = [x for x in reasons if x in DISPLAY_ONLY_NOISE_REASONS]
    deciding = [x for x in reasons if x not in DISPLAY_ONLY_NOISE_REASONS]
    return deciding, display_only


def classify_non_soma(r):
    return r["mainPeakToTroughRatio"] > NON_SOMA_PARAMS["maxMainPeakToTroughRatio_nonSomatic"]


def main():
    print(f"Using qMetrics from: {QM_PATH}")
    qm = pd.read_csv(QM_PATH).set_index("phy_clusterID")

    spike_times = np.load(os.path.join(KS_DIR, "spike_times.npy")).ravel()
    spike_clusters = np.load(os.path.join(KS_DIR, "spike_clusters.npy")).ravel()
    templates = np.load(os.path.join(KS_DIR, "templates.npy"))
    geometry = ChannelGeometry.from_json(CHANNEL_MAP_JSON)
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

    print(f"\n=== CORRECTED CLASSIFICATION SUMMARY ({SESSION_ID}) ===")
    print(df["label"].value_counts().to_string())

    print("\n=== comparison: corrected 'core' label vs bombcell original ===")
    core = df["label"].str.replace("+NON-SOMA", "", regex=False)
    print(pd.crosstab(df["bombcell_original_label"], core))


if __name__ == "__main__":
    main()

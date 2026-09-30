"""
Shows detected events next to BOTH candidate explanations:
  - the template of the unit this analysis matched them to ("new unit")
  - the template of the unit they are ALREADY assigned to ("assigned unit"),
    where one exists

Both templates are drawn as they appear ON THE SAME CHANNEL the raw data
was read from (templates[other_unit][:, this_channel]), not on each unit's
own peak channel -- otherwise the two curves wouldn't be comparable.

Two categories shown per unit:
  ALREADY ASSIGNED -- matched this unit's template, but Kilosort already
    has them under a neighbouring unit. The question these pose is which
    unit they really belong to (or whether the two units are one).
  NEW -- no nearby unit has them at all; first detected in this analysis.

Usage: python demo_show_new_detections.py
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

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
DAT_PATH = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\20260916_110311.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN, FS, NT0MIN, N, ITEMSIZE = 384, 30000.0, 20, 61, 2
SEARCH_RADIUS, N_CYCLES = 25, 3.0

OWN_P25 = {440: 0.720, 408: 0.708}

templates = np.load(KS + r"\templates.npy")
spike_times = np.load(KS + r"\spike_times.npy").ravel()
spike_clusters = np.load(KS + r"\spike_clusters.npy").ravel()
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")

det = pd.read_csv(os.path.join(OUT, "scan20min_detections.csv"))


def get_filtered(ch, center, pad=250, fb=60):
    s0 = center - pad - fb
    nr = (center + pad + fb) - s0
    with open(DAT_PATH, "rb") as f:
        f.seek(int(s0) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read(nr * N_CHAN_BIN * ITEMSIZE)
    blk = np.frombuffer(raw, dtype=np.int16).reshape(nr, N_CHAN_BIN)
    return filtfilt(b_hp, a_hp, blk[:, ch].astype(np.float64))[fb:-fb] * GAIN_TO_UV


def unit_pipeline(uid):
    ta = templates[uid]
    pc = int(np.argmax(ta.max(axis=0) - ta.min(axis=0)))
    tm = ta[:, pc]
    f0s = np.linspace(300, 4000, 50)
    ps = int(np.ceil(N_CYCLES * FS / (2 * f0s.min()))) + 20
    tr = np.zeros(N + 2 * ps); ctr = ps + NT0MIN
    tr[ctr - NT0MIN: ctr - NT0MIN + N] = tm
    mg = [abs(wavelet_transform_at(tr, make_morlet(f, FS, N_CYCLES)[1], ctr)) for f in f0s]
    f0 = float(f0s[int(np.nanargmax(mg))])
    _, psi = make_morlet(f0, FS, n_cycles=N_CYCLES)
    s0b, _, qb = build_basis(tm, dt=1.0)
    return dict(peak_ch=pc, template=tm, f0=f0, psi=psi,
                ref_phase=calibrate_reference_phase(tm, psi, FS, align_index=NT0MIN),
                s0=s0b, q=qb)


def aligned_snippet(trace, pad, cfg):
    r = coarse_then_fine_shift(trace, pad, cfg["template"], cfg["psi"], cfg["f0"], FS,
                                reference_phase=cfg["ref_phase"], search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
    tc = pad + r["total_shift"]
    lo, hi = int(np.floor(tc - NT0MIN)) - 2, int(np.ceil(tc - NT0MIN + N)) + 2
    if lo < 0 or hi > len(trace):
        return None
    ip = interp1d(np.arange(lo, hi), trace[lo:hi], kind="cubic", bounds_error=False, fill_value=0.0)
    return ip(tc - NT0MIN + np.arange(N))


def best_amp(snippet, templ_on_channel):
    """Least-squares amplitude of a template against a snippet."""
    tn = templ_on_channel
    denom = float(np.dot(tn, tn))
    return float(np.dot(snippet, tn) / denom) if denom > 0 else 0.0


panels = []
for uid in [440, 408]:
    cfg = unit_pipeline(uid)
    pc = cfg["peak_ch"]
    d = det[(det.uid == uid) & (det.r2_full >= OWN_P25[uid])].copy()
    assigned = d[d.neighbor_uid.notna()].sort_values("r2_full", ascending=False)
    new = d[d.neighbor_uid.isna()].sort_values("r2_full", ascending=False)

    for _, row in assigned.head(3).iterrows():
        panels.append(dict(uid=uid, cfg=cfg, sample=int(row["sample"]), r2=row["r2_full"],
                            other=int(row["neighbor_uid"]), kind="ASSIGNED"))
    for _, row in new.head(3).iterrows():
        panels.append(dict(uid=uid, cfg=cfg, sample=int(row["sample"]), r2=row["r2_full"],
                            other=None, kind="NEW"))

n = len(panels)
ncol = 6
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.6 * nrow), squeeze=False)
t_ms = (np.arange(N) - NT0MIN) / FS * 1000

for k, p in enumerate(panels):
    ax = axes[k // ncol][k % ncol]
    cfg = p["cfg"]
    pc = cfg["peak_ch"]
    trace = get_filtered(pc, p["sample"], pad=250)
    snip = aligned_snippet(trace, 250, cfg)
    if snip is None:
        ax.axis("off")
        continue

    ax.plot(t_ms, snip, color="#222", lw=1.8, label="raw (aligned)")

    a_own = best_amp(snip, cfg["template"])
    ax.plot(t_ms, cfg["template"] * a_own, color="#1f77b4", lw=1.5, ls="--",
            label=f"unit {p['uid']} (matched here)")

    if p["other"] is not None:
        other_on_ch = templates[p["other"]][:, pc]
        a_other = best_amp(snip, other_on_ch)
        ax.plot(t_ms, other_on_ch * a_other, color="#d62728", lw=1.5, ls=":",
                label=f"unit {p['other']} (already assigned)")
        shape_corr = np.corrcoef(cfg["template"], other_on_ch)[0, 1]
        title = (f"{p['kind']}  unit {p['uid']} vs {p['other']}\nsample {p['sample']}, R2={p['r2']:.2f}\n"
                 f"templates on ch{pc} corr={shape_corr:.3f}")
    else:
        title = f"{p['kind']}  unit {p['uid']}\nsample {p['sample']}, R2={p['r2']:.2f}\nno nearby unit has this event"

    ax.set_title(title, fontsize=9)
    ax.axhline(0, color="grey", lw=0.4)
    ax.set_xlabel("time (ms)")
    if k % ncol == 0:
        ax.set_ylabel("uV")
    ax.legend(fontsize=6.5, loc="lower right")

for k in range(n, nrow * ncol):
    axes[k // ncol][k % ncol].axis("off")

plt.suptitle("Detected events vs. both candidate explanations: the unit matched here, and the unit already holding them",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out = os.path.join(OUT, "new_detections_vs_assigned_unit.png")
plt.savefig(out, dpi=115, bbox_inches="tight")
print("saved", out)
for p in panels:
    print(f"  {p['kind']:<9} unit {p['uid']} sample {p['sample']} R2={p['r2']:.2f} other={p['other']}")

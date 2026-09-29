"""
Real-data validation of noise_whitening.py, unit 342's channel (same unit
used throughout this project). Two checks:

(1) Does the whitening operator actually flatten real background noise's
    autocorrelation? Fit the AR model on one real noise segment, apply the
    resulting whitening operator to a DIFFERENT, held-out real noise
    segment, and compare average autocorrelation before/after.

(2) Does whitening actually improve discrimination of a real spike from
    real background noise, beyond a plain matched filter? Inject the
    unit's own real template into many real (not synthetic) held-out noise
    windows at a modest amplitude, compare against real noise-only windows,
    and measure d' (signal/noise separation) for a plain matched filter vs.
    the whitened version of the same comparison.

Usage: python demo_noise_whitening_validation.py
"""
import os
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from noise_whitening import build_whitening_from_noise

VR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
DAT_PATH = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\20260901_085606.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN = 384
FS = 30000.0
NT0MIN = 20
N = 61
UID = 342
ITEMSIZE = 2
AR_ORDER = 4

templates = np.load(VR + r"\templates.npy")
spike_times = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()
templ_all = templates[UID]
peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))
template = templ_all[:, peak_ch]

st = np.sort(spike_times[spike_clusters == UID])
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")


def read_filtered_chunk(start_sample, n_read, filt_buf=200):
    with open(DAT_PATH, "rb") as f:
        f.seek(int(start_sample - filt_buf) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read((n_read + 2 * filt_buf) * N_CHAN_BIN * ITEMSIZE)
    block = np.frombuffer(raw, dtype=np.int16).reshape(n_read + 2 * filt_buf, N_CHAN_BIN)
    return filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64))[filt_buf:-filt_buf] * GAIN_TO_UV


# two INDEPENDENT real noise stretches, well away from this unit's own
# spikes and from each other -- one to FIT the AR model, one held out to
# VALIDATE on
fit_noise = read_filtered_chunk(int(st[0]) + 500000, 60000)      # ~2s, for fitting
held_out_noise = read_filtered_chunk(int(st[0]) + 2000000, 60000)  # ~2s, disjoint, for checking

W, phi, sigma2 = build_whitening_from_noise(fit_noise, N, order=AR_ORDER)
print(f"unit {UID}, channel {peak_ch}: fitted AR({AR_ORDER}) model")
print(f"  phi = {np.round(phi, 4)}")
print(f"  innovation sigma2 = {sigma2:.4f}")
roots = np.roots(np.concatenate(([1.0], -phi)))
print(f"  companion-matrix roots (should all be INSIDE the unit circle, i.e. |root|<1, "
      f"for stationarity): {np.round(np.abs(roots), 3)}")

# ============================================================
# CHECK 1: autocorrelation flattening on HELD-OUT real noise
# ============================================================
rng = np.random.default_rng(0)
n_windows = 400
starts = rng.integers(0, len(held_out_noise) - N, size=n_windows)
raw_windows = np.array([held_out_noise[s:s + N] for s in starts])
white_windows = raw_windows @ W.T  # apply W to each row

def mean_autocorr(windows, max_lag=15):
    out = []
    for lag in range(max_lag + 1):
        vals = [np.corrcoef(w[:-lag or None], w[lag:])[0, 1] if lag > 0 else 1.0
                for w in windows]
        out.append(np.nanmean(vals))
    return np.array(out)

raw_acf = mean_autocorr(raw_windows)
white_acf = mean_autocorr(white_windows)
print(f"\nheld-out noise, mean autocorrelation at lag 1: raw={raw_acf[1]:.3f}, whitened={white_acf[1]:.3f}")
print(f"                                   at lag 5: raw={raw_acf[5]:.3f}, whitened={white_acf[5]:.3f}")

# ============================================================
# CHECK 2: does whitening improve real spike-vs-noise discrimination?
# inject the REAL template into REAL held-out noise windows, compare a
# plain matched filter to a whitened one via d' (signal/noise separation)
# ============================================================
INJECT_AMP = 0.35  # fraction of the template's own peak-to-trough scale -- a modest, low-SNR injection
template_norm = template / np.linalg.norm(template)
whitened_template = W @ template
whitened_template_norm = whitened_template / np.linalg.norm(whitened_template)

n_trials = 300
noise_starts = rng.integers(0, len(held_out_noise) - N, size=n_trials)
signal_starts = rng.integers(0, len(held_out_noise) - N, size=n_trials)

scale = INJECT_AMP * (template.max() - template.min())
injected_template = template / (template.max() - template.min()) * scale

plain_signal, plain_noise = [], []
white_signal, white_noise = [], []
for s_noise, s_sig in zip(noise_starts, signal_starts):
    noise_win = held_out_noise[s_noise:s_noise + N]
    sig_win = held_out_noise[s_sig:s_sig + N] + injected_template

    plain_noise.append(np.dot(noise_win, template_norm))
    plain_signal.append(np.dot(sig_win, template_norm))

    white_noise.append(np.dot(W @ noise_win, whitened_template_norm))
    white_signal.append(np.dot(W @ sig_win, whitened_template_norm))

plain_signal, plain_noise = np.array(plain_signal), np.array(plain_noise)
white_signal, white_noise = np.array(white_signal), np.array(white_noise)


def dprime(signal, noise):
    pooled_std = np.sqrt(0.5 * (signal.var() + noise.var()))
    return (signal.mean() - noise.mean()) / pooled_std


dp_plain = dprime(plain_signal, plain_noise)
dp_white = dprime(white_signal, white_noise)
print(f"\ninjected amplitude = {INJECT_AMP:.2f}x this unit's own template scale, n_trials={n_trials}")
print(f"d' (signal vs. noise separation), PLAIN matched filter:    {dp_plain:.3f}")
print(f"d' (signal vs. noise separation), WHITENED matched filter: {dp_white:.3f}")
print(f"improvement: {(dp_white/dp_plain - 1)*100:+.1f}%" if dp_plain > 0 else "")

# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

ax = axes[0]
lags = np.arange(len(raw_acf))
ax.plot(lags, raw_acf, marker="o", color="#d62728", label="raw noise")
ax.plot(lags, white_acf, marker="s", color="#1f77b4", label="whitened noise")
ax.axhline(0, color="grey", lw=0.5)
ax.set_xlabel("lag (samples)")
ax.set_ylabel("mean autocorrelation")
ax.set_title(f"CHECK 1: does whitening flatten real,\nheld-out noise's autocorrelation?\n"
             f"(unit {UID}'s channel, AR({AR_ORDER}) model fit on a\nDIFFERENT noise segment)", fontsize=10.5)
ax.legend(fontsize=8.5)

ax = axes[1]
ax.hist(plain_noise, bins=25, alpha=0.55, color="#888", label="noise-only")
ax.hist(plain_signal, bins=25, alpha=0.55, color="#d62728", label=f"real template injected\n({INJECT_AMP:.2f}x scale)")
ax.set_xlabel("plain matched-filter score")
ax.set_ylabel("count")
ax.set_title(f"CHECK 2a: PLAIN matched filter\nd' = {dp_plain:.3f}", fontsize=10.5)
ax.legend(fontsize=8)

ax = axes[2]
ax.hist(white_noise, bins=25, alpha=0.55, color="#888", label="noise-only")
ax.hist(white_signal, bins=25, alpha=0.55, color="#1f77b4", label=f"real template injected\n({INJECT_AMP:.2f}x scale)")
ax.set_xlabel("whitened matched-filter score")
ax.set_ylabel("count")
ax.set_title(f"CHECK 2b: WHITENED matched filter\nd' = {dp_white:.3f}", fontsize=10.5)
ax.legend(fontsize=8)

plt.suptitle(f"Real-data validation of temporal whitening, unit {UID} channel {peak_ch}", fontsize=13, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(OUT, "whitening_01_validation.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("\nsaved", out_path)

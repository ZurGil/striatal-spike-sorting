"""The stimulus, reconstructed exactly — and what it does and does not allow.

WHAT WAS ESTABLISHED (recover_sources.py, reproduced in the header here):
the Dual2AFC_nat 'Natural' stimulus is a LINEAR SUPERPOSITION of two fixed
natural sounds,

    x(t; w) = w * A(t) + (1 - w) * B(t)

with A = frogs2.wav, B = Passer_Montanus.wav and w = Custom.AuditoryOmega,
saved for every trial, with DV = 2w - 1 exactly. Two of the four saved
waveforms determine A and B per sample; predicting the other two from w alone
gives r = 1.000000 and normalised RMSE = 0.0000. So the exact played waveform
of all 953 trials is recoverable, without the rig's .wav files.

WHY GIL'S EARLY-VS-LATE COMPARISON HAS NO REFERENT IN THIS TASK. Fitting the
mixing ratio in independent 10 ms frames returns the trial's own w in every
frame, with SD = 0.0000. The ratio is CONSTANT for the whole 350 ms. There is
therefore no trial on which evidence for the correct side arrived early and
none on which it arrived late: every trial delivers one fixed ratio throughout,
and two trials with the same w are the same sound sample-for-sample. The
variability the comparison needs does not exist -- not missing from the record,
absent from the design. A click train has it; a constant-ratio morph does not.

WHAT IS STILL AVAILABLE, AND IT IS THE INTERESTING VERSION. Although the RATIO
is constant, how much the sound REVEALS about that ratio is not. Differentiating
the mixture with respect to w gives

    d x(t; w) / d w = A(t) - B(t)

so the instantaneous evidence rate about w is governed by (A - B)^2: moments
where the two sounds differ carry information, moments where they coincide
carry none, no matter what w is. That profile is FIXED across trials (A and B
are fixed), which is precisely why it cannot be used to split trials -- but it
can be used WITHIN the trial. If the population is integrating the stimulus,
the choice signal should grow fastest during the moments the stimulus is most
informative.

This script reconstructs A and B, computes the evidence-rate profile over the
350 ms that is actually played, and tests that prediction against the measured
decoding time course.

Usage: python stimulus_evidence_profile.py
"""
import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKL = (r"C:\Users\Adam\AppData\Local\Temp\claude\D--Gil-spike-sorting-agent"
       r"\8186c809-76fa-41ff-bbca-59d44125e5fc\scratchpad\ratroot_shamir\bpod"
       r"\Shamir01_Dual2AFC_nat_Sep16_2026_Session1.pkl")
OUT = r"D:\Gil\spike_sorting_agent\outputs"
FIG = os.path.join(OUT, "reward_cue_figs")
FS = 192000.0
STIM_DUR = 0.35
BIN = 0.025          # matches the decoding time course's step

sd = pickle.load(open(PKL, "rb"))["SessionData"]
c = sd["Custom"]
snd, om = c["AudSound"], np.asarray(c["AuditoryOmega"], float)
I = [i for i in range(len(snd)) if np.asarray(snd[i]).size > 0]
X = {i: np.asarray(snd[i], float).ravel() for i in I}
w = {i: float(om[i]) for i in I}

i1, i2 = max(((a, b) for a in I for b in I if a < b),
             key=lambda ab: abs(w[ab[0]] - w[ab[1]]))
M = np.array([[w[i1], 1 - w[i1]], [w[i2], 1 - w[i2]]])
A, B = np.linalg.solve(M, np.vstack([X[i1], X[i2]]))
print(f"recovered A and B from trials {i1} (w={w[i1]:.4f}) and {i2} (w={w[i2]:.4f})")
for j in I:
    if j in (i1, i2):
        continue
    pred = w[j] * A + (1 - w[j]) * B
    print(f"  held-out trial {j}: r = {np.corrcoef(pred, X[j])[0,1]:.6f}")

n_play = int(STIM_DUR * FS)
A, B = A[:n_play], B[:n_play]
D = A - B                                  # d(stimulus)/d(omega)

# ---- evidence-rate profile, two ways
nb = int(STIM_DUR / BIN)
edges = (np.arange(nb + 1) * BIN * FS).astype(int)
raw = np.array([np.mean(D[edges[k]:edges[k + 1]] ** 2) for k in range(nb)])

# spectral version: how different are A and B in this bin, in a hearing-like
# representation (log power across frequency), which is closer to what the
# animal can actually use than raw waveform subtraction
f, t, SA = spectrogram(A, fs=FS, nperseg=1024, noverlap=768)
_, _, SB = spectrogram(B, fs=FS, nperseg=1024, noverlap=768)
keep = (f >= 500) & (f <= 20000)
LA = 10 * np.log10(SA[keep] + 1e-20)
LB = 10 * np.log10(SB[keep] + 1e-20)
spec_d = np.sqrt(((LA - LB) ** 2).mean(axis=0))
spec = np.array([spec_d[(t >= k * BIN) & (t < (k + 1) * BIN)].mean()
                 for k in range(nb)])

prof = pd.DataFrame(dict(t_start=np.arange(nb) * BIN,
                         t_mid=(np.arange(nb) + 0.5) * BIN,
                         evidence_power=raw, evidence_spectral=spec))
prof.to_csv(os.path.join(OUT, "stimulus_evidence_profile.csv"), index=False)
print(f"\nevidence-rate profile over the played {STIM_DUR*1000:.0f} ms, "
      f"{BIN*1000:.0f} ms bins:")
print(f"  (A-B)^2 power : min {raw.min():.4g}  max {raw.max():.4g}  "
      f"ratio {raw.max()/max(raw.min(),1e-30):.1f}x")
print(f"  spectral diff : min {spec.min():.2f}  max {spec.max():.2f} dB")
print(f"  the two profiles agree: Spearman rho = "
      f"{spearmanr(raw, spec).statistic:+.3f}")

# ---- does decoding grow fastest when the stimulus is most informative?
dyn_p = os.path.join(OUT, "stimulus_dynamics_20260916_110311.csv")
if os.path.exists(dyn_p):
    dyn = pd.read_csv(dyn_p)
    d = dyn[(dyn.t_mid > 0) & (dyn.t_mid <= STIM_DUR)].sort_values("t_mid").copy()
    d["d_auc"] = d.choice_auc.diff()
    d = d.dropna(subset=["d_auc"])
    ev_at = np.interp(d.t_mid.to_numpy(), prof.t_mid.to_numpy(), spec)
    ev_pw = np.interp(d.t_mid.to_numpy(), prof.t_mid.to_numpy(), raw)
    print("\n" + "=" * 88)
    print("DOES THE CHOICE SIGNAL GROW FASTEST WHEN THE STIMULUS IS MOST INFORMATIVE?")
    print("=" * 88)
    print(f"  n = {len(d)} windows inside the stimulus")
    for nm, e in [("spectral difference", ev_at), ("(A-B)^2 power", ev_pw)]:
        rs = spearmanr(e, d.d_auc.to_numpy())
        rp = pearsonr(e, d.d_auc.to_numpy())
        print(f"  increment in choice AUC vs {nm:<20} "
              f"Spearman rho={rs.statistic:+.3f} p={rs.pvalue:.3f}   "
              f"Pearson r={rp.statistic:+.3f} p={rp.pvalue:.3f}")
    print("\n  [positive = the decision grows while informative sound is arriving,")
    print("   which is what integration predicts. Null = the signal rises on its")
    print("   own schedule regardless of what the stimulus is doing.]")

# ---- figure
fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.2), sharex=True)
ax = axes[0]
ax.plot(prof.t_mid, spec, "o-", color="#8c3b2e", lw=2, ms=4,
        label="spectral difference |A−B| (dB)")
ax.set_ylabel("evidence rate (dB)", fontsize=9)
ax.set_title("How informative is the sound, moment by moment?\n"
             "fixed across all trials — A and B are the same two recordings every time",
             fontsize=10.5)
ax.legend(fontsize=8, frameon=False)
ax.tick_params(labelsize=8)
ax = axes[1]
if os.path.exists(dyn_p):
    ax.plot(d.t_mid, d.d_auc, "o-", color="#2e6f8c", lw=2, ms=4)
    ax.axhline(0, color="#aaa", lw=0.8)
    ax.set_ylabel("increment in choice AUC\nper 25 ms step", fontsize=9)
ax.set_xlabel("time from stimulus onset (s)", fontsize=9)
ax.tick_params(labelsize=8)
fig.tight_layout()
p = os.path.join(FIG, "stimulus_evidence_profile.png")
fig.savefig(p, dpi=125, bbox_inches="tight")
print(f"\nsaved {p}")
print("saved stimulus_evidence_profile.csv")

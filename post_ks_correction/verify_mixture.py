"""Pin down the remaining facts about the mixture:
  1. re-confirm the superposition on held-out waveforms
  2. which recovered source is which sound (spectral character)
  3. which side high omega rewards
  4. whether the mixture is power-normalised or a plain weighted sum
  5. whether a 2-source model is forced, or whether 3 sources would fit too
"""
import pickle
import numpy as np
from scipy.signal import welch

PKL = (r"C:\Users\Adam\AppData\Local\Temp\claude\D--Gil-spike-sorting-agent"
       r"\8186c809-76fa-41ff-bbca-59d44125e5fc\scratchpad\ratroot_shamir\bpod"
       r"\Shamir01_Dual2AFC_nat_Sep16_2026_Session1.pkl")
FS = 192000.0

sd = pickle.load(open(PKL, "rb"))["SessionData"]
c = sd["Custom"]
snd = c["AudSound"]
om = np.asarray(c["AuditoryOmega"], float)
dv = np.asarray(c["DV"], float)
lr = np.asarray(c["LeftRewarded"], float)
I = [i for i in range(len(snd)) if np.asarray(snd[i]).size > 0]
X = {i: np.asarray(snd[i], float).ravel() for i in I}
w = {i: float(om[i]) for i in I}

# ---- 1. solve and verify
i1, i2 = 953, 954
M = np.array([[w[i1], 1 - w[i1]], [w[i2], 1 - w[i2]]])
A, B = np.linalg.solve(M, np.vstack([X[i1], X[i2]]))
print("1. SUPERPOSITION x = w*A + (1-w)*B")
for j in (955, 956):
    pred = w[j] * A + (1 - w[j]) * B
    err = np.abs(pred - X[j]).max()
    print(f"   held-out trial {j} (w={w[j]:.4f}): max abs error = {err:.3e}"
          f"   (signal range {X[j].ptp():.3f})")

# ---- 2. which source is which?
print("\n2. WHAT ARE A AND B?")
for nm, s in [("A (w=1 endpoint)", A), ("B (w=0 endpoint)", B)]:
    f, P = welch(s, fs=FS, nperseg=8192)
    P = P / P.sum()
    cf = (f * P).sum()
    lo = P[(f >= 200) & (f < 3000)].sum()
    hi = P[(f >= 3000) & (f <= 20000)].sum()
    # spectral flatness: tonal (bird song) vs broadband (frog chorus)
    band = P[(f >= 200) & (f <= 20000)] + 1e-30
    flat = np.exp(np.log(band).mean()) / band.mean()
    print(f"   {nm}: centroid {cf:7.0f} Hz | power 0.2-3kHz {100*lo:4.1f}% "
          f"| 3-20kHz {100*hi:4.1f}% | spectral flatness {flat:.4f}")
print("   (lower flatness = more tonal/harmonic, e.g. bird song;")
print("    higher = more noise-like/broadband, e.g. a frog chorus)")

# ---- 3. which side does high omega reward?
ok = ~np.isnan(lr) & ~np.isnan(dv)
hi_w = ok & (dv > 0.5)
lo_w = ok & (dv < -0.5)
print("\n3. OMEGA -> SIDE")
print(f"   DV = 2*omega - 1 (verified earlier)")
print(f"   trials with DV > +0.5: LeftRewarded = 1 on {100*lr[hi_w].mean():.1f}%")
print(f"   trials with DV < -0.5: LeftRewarded = 1 on {100*lr[lo_w].mean():.1f}%")
side = "RIGHT" if lr[hi_w].mean() < 0.5 else "LEFT"
print(f"   => high omega (more of sound A) means {side} is correct")

# ---- 4. plain weighted sum, or power-normalised?
print("\n4. IS THE MIXTURE POWER-NORMALISED?")
for j in I:
    pred_rms = np.sqrt((w[j] ** 2) * (A ** 2).mean()
                       + ((1 - w[j]) ** 2) * (B ** 2).mean()
                       + 2 * w[j] * (1 - w[j]) * (A * B).mean())
    print(f"   trial {j} (w={w[j]:.4f}): actual rms {X[j].std():.5f}, "
          f"plain-sum prediction {pred_rms:.5f}")
print("   (equal => a plain weighted sum, with no renormalisation; the overall")
print("    loudness therefore varies with omega, dipping near w=0.5)")

# ---- 5. is a 2-source model forced?
print("\n5. COULD IT BE MORE THAN TWO SOURCES?")
Xall = np.vstack([X[i] for i in I])
u, s, vt = np.linalg.svd(Xall - 0, full_matrices=False)
print(f"   singular values of the 4 waveforms: {np.round(s / s[0], 6)}")
print("   (two dominant values and the rest at numerical zero means the four")
print("    stimuli lie in a strictly 2-dimensional space -> exactly 2 sources)")

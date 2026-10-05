"""Can the two source sounds be recovered from the four saved waveforms?

If the protocol builds the stimulus as a linear superposition
      x(w) = w*A + (1-w)*B
then TWO waveforms at known w determine A and B exactly, per sample:
      [w1, 1-w1; w2, 1-w2] [A; B] = [x1; x2]
and the remaining two waveforms are then a held-out test. If the prediction is
near-perfect, every trial's stimulus is reconstructible from its own omega --
which IS saved for all 953 trials -- and the time-resolved evidence analysis
becomes possible with no protocol change and no access to the rig.

If the prediction fails, the morph is something else (spectral morphing,
time-varying mixing, a per-trial random draw of source segments), and the four
examples are not enough to invert it.

Also tested: whether the mixture is STATIC within a trial. Even with a constant
mixing ratio, how much the sound reveals about A vs B varies moment to moment,
because the two natural sounds are loud at different times. That time-resolved
discriminability is exactly what an early-vs-late evidence analysis needs.
"""
import pickle
import numpy as np

P = (r"C:\Users\Adam\AppData\Local\Temp\claude\D--Gil-spike-sorting-agent"
     r"\8186c809-76fa-41ff-bbca-59d44125e5fc\scratchpad\ratroot_shamir\bpod"
     r"\Shamir01_Dual2AFC_nat_Sep16_2026_Session1.pkl")
FS = 192000.0

sd = pickle.load(open(P, "rb"))["SessionData"]
c = sd["Custom"]
snd, om = c["AudSound"], np.asarray(c["AuditoryOmega"], float)
I = [i for i in range(len(snd)) if np.asarray(snd[i]).size > 0]
X = {i: np.asarray(snd[i], float).ravel() for i in I}
w = {i: float(om[i]) for i in I}
print("saved waveforms:", {i: round(w[i], 4) for i in I})

# ---- solve for A and B from the two most widely separated omegas
i1, i2 = max(((a, b) for a in I for b in I if a < b),
             key=lambda ab: abs(w[ab[0]] - w[ab[1]]))
print(f"\nsolving with trials {i1} (w={w[i1]:.4f}) and {i2} (w={w[i2]:.4f})")
M = np.array([[w[i1], 1 - w[i1]], [w[i2], 1 - w[i2]]])
print(f"  mixing matrix det = {np.linalg.det(M):.4f}")
AB = np.linalg.solve(M, np.vstack([X[i1], X[i2]]))
A, B = AB[0], AB[1]
print(f"  recovered A: rms {A.std():.4f}, B: rms {B.std():.4f}")

# ---- held-out prediction on the other two
print("\nHELD-OUT TEST — predict the other waveforms from omega alone:")
ok = True
for j in I:
    if j in (i1, i2):
        continue
    pred = w[j] * A + (1 - w[j]) * B
    r = np.corrcoef(pred, X[j])[0, 1]
    nrmse = np.sqrt(((pred - X[j]) ** 2).mean()) / X[j].std()
    print(f"  trial {j} (w={w[j]:.4f}):  r = {r:.6f}   normalised RMSE = {nrmse:.4f}")
    ok &= (r > 0.999)

print()
if ok:
    print("LINEAR SUPERPOSITION CONFIRMED. Every trial's stimulus is reconstructible")
    print("from its own omega, so the time-resolved evidence analysis is possible.")
else:
    print("NOT a simple linear superposition of two fixed sources.")
    print("The four examples are not enough to invert the morph; the two .wav")
    print("files and the generating code are needed.")

# ---- is the mixture static within a trial?
# Project each short frame of a saved waveform onto the recovered A and B.
# If the mixing ratio is constant, the per-frame estimate of w should be flat.
print("\nIS THE MIXING RATIO CONSTANT WITHIN A TRIAL?")
FR = int(0.010 * FS)          # 10 ms frames
for j in I:
    x = X[j]
    est = []
    for k in range(0, len(x) - FR, FR):
        a, b, y = A[k:k + FR], B[k:k + FR], x[k:k + FR]
        G = np.array([[a @ a, a @ b], [a @ b, b @ b]])
        try:
            coef = np.linalg.solve(G, np.array([a @ y, b @ y]))
        except np.linalg.LinAlgError:
            continue
        s = coef.sum()
        if abs(s) > 1e-9:
            est.append(coef[0] / s)
    est = np.array(est)
    print(f"  trial {j} (true w={w[j]:.4f}): per-frame w  "
          f"median {np.median(est):.4f}, IQR {np.percentile(est,25):.4f}"
          f"-{np.percentile(est,75):.4f}, SD {est.std():.4f}")
print("\n  [a flat per-frame w means the RATIO is constant; evidence then varies")
print("   over time only through how distinguishable A and B are moment to")
print("   moment, which is still a real, computable time-resolved signal]")

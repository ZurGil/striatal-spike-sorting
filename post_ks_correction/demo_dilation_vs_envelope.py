"""
Two different ways to make a wavelet "longer," compared directly:

(A) ENVELOPE-ONLY STRETCH (what make_morlet's n_cycles does): oscillation
    speed (f0) stays fixed; only the fading window gets longer, so more of
    the SAME fast wiggles become visible before they fade out.
    psi(t) = exp(i*2*pi*f0*t) * window(t/duration)

(B) TRUE DILATION (the standard textbook meaning of "wavelet scale"): the
    ENTIRE waveform is stretched in time by a factor `a`, so the
    oscillation itself slows down (fewer, wider wiggles) AT THE SAME TIME
    the envelope widens -- both effects are linked, not independent.
    psi_a(t) = exp(i*2*pi*(f0/a)*t) * window(t/a)
    At a=1 this is the original probe; at a=2, the effective frequency
    HALVES (2000 -> 1000 Hz) and the envelope duration DOUBLES, together.

Usage: python demo_dilation_vs_envelope.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavelet_features import make_morlet

OUT = r"D:\Gil\spike_sorting_agent\outputs"
fs = 30000.0
f0_hz = 2000.0


def make_dilated_morlet(f0_hz, fs, a, n_cycles=3.0):
    """True dilation: psi_a(t) = exp(i*2*pi*(f0/a)*t) * window(t/a).
    At a=1, identical to make_morlet(f0_hz, fs, n_cycles). At a=2, the
    oscillation is half as fast AND the window is twice as wide -- both
    together, not independently."""
    effective_f0 = f0_hz / a
    return make_morlet(effective_f0, fs, n_cycles=n_cycles)


# ============================================================
# FIGURE: the two probes, both "elongated" by the same visual
# amount, side by side.
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 8))

_, psi_base = make_morlet(f0_hz, fs, n_cycles=3.0)
_, psi_envelope_only = make_morlet(f0_hz, fs, n_cycles=9.0)  # (A): 3x more cycles, same f0
_, psi_dilated = make_dilated_morlet(f0_hz, fs, a=3.0, n_cycles=3.0)  # (B): 3x dilation

t_base_ms = (np.arange(len(psi_base)) - len(psi_base) // 2) / fs * 1000
t_env_ms = (np.arange(len(psi_envelope_only)) - len(psi_envelope_only) // 2) / fs * 1000
t_dil_ms = (np.arange(len(psi_dilated)) - len(psi_dilated) // 2) / fs * 1000

ax = axes[0, 0]
ax.plot(t_base_ms, psi_base.real, color="#888", lw=1.3, label="original (3 cycles, 2000Hz)")
ax.set_xlim(-3, 3)
ax.set_title("ORIGINAL probe (for reference)\n3 cycles fit inside the window, oscillates at 2000Hz", fontsize=9.5)
ax.legend(fontsize=7.5)
ax.set_xlabel("time (ms)")

ax = axes[0, 1]
ax.plot(t_base_ms, psi_base.real, color="#ccc", lw=1.0, label="original")
ax.plot(t_env_ms, psi_envelope_only.real, color="#1f77b4", lw=1.4, label="9 cycles, STILL 2000Hz")
ax.set_xlim(-3, 3)
ax.set_title("(A) ENVELOPE-ONLY stretch (what n_cycles does)\n"
             "same wiggle SPEED, just more of them visible\n"
             "before the window fades them out", fontsize=9.5)
ax.legend(fontsize=7.5)
ax.set_xlabel("time (ms)")

ax = axes[1, 0]
ax.plot(t_base_ms, psi_base.real, color="#ccc", lw=1.0, label="original")
ax.plot(t_dil_ms, psi_dilated.real, color="#d62728", lw=1.4, label="dilated a=3, now 667Hz")
ax.set_xlim(-3, 3)
ax.set_title("(B) TRUE DILATION (what you're asking about)\n"
             "wiggles are SLOWER (2000Hz -> 667Hz) AND\n"
             "the window is 3x wider, together", fontsize=9.5)
ax.legend(fontsize=7.5)
ax.set_xlabel("time (ms)")

ax = axes[1, 1]
ax.plot(t_env_ms, psi_envelope_only.real, color="#1f77b4", lw=1.4, label="(A) envelope-only, 9 cycles @ 2000Hz")
ax.plot(t_dil_ms, psi_dilated.real, color="#d62728", lw=1.4, label="(B) true dilation, 3 cycles @ 667Hz")
ax.set_xlim(-3, 3)
ax.set_title("BOTH directly overlaid\ncount the wiggles: (A) is fast+many,\n(B) is slow+few, even though both got 'longer'", fontsize=9.5)
ax.legend(fontsize=7.5)
ax.set_xlabel("time (ms)")

plt.suptitle("Two different meanings of 'a longer wavelet' -- they are NOT the same operation", fontsize=13)
plt.tight_layout()
out1 = os.path.join(OUT, "wavelet_06_dilation_vs_envelope.png")
plt.savefig(out1, dpi=115, bbox_inches="tight")
plt.close(fig)
print("saved", out1)


# ============================================================
# Numeric check: does the green (safe-zone) band widen for each
# kind of elongation?
# ============================================================
print("\n=== does the half-cycle safe-zone boundary change? ===")
print(f"{'method':<28}{'effective f0 (Hz)':>18}{'half-cycle boundary (samples)':>32}")
half0 = (fs / f0_hz) / 2.0
print(f"{'original (n_cycles=3)':<28}{f0_hz:18.1f}{half0:32.2f}")

for n_cycles in [3.0, 9.0, 20.0]:
    half = (fs / f0_hz) / 2.0  # f0 unchanged -- envelope-only stretch
    print(f"{'(A) envelope-only, n=' + str(n_cycles):<28}{f0_hz:18.1f}{half:32.2f}")

for a in [2.0, 3.0, 5.0]:
    effective_f0 = f0_hz / a
    half = (fs / effective_f0) / 2.0
    print(f"{'(B) true dilation, a=' + str(a):<28}{effective_f0:18.1f}{half:32.2f}")

print("\n(A) leaves the boundary completely unchanged, no matter how many")
print("    cycles you add, because f0 never changes.")
print("(B) widens the boundary in direct proportion to the dilation factor,")
print("    because slowing the oscillation down also lengthens its period,")
print("    and the boundary IS half of that period.")

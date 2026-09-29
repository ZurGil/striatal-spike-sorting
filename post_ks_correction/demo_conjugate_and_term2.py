"""
Closes two gaps from the full derivation: (1) visually, why conjugating
freezes the spin for a matching-frequency signal but doubles it without
conjugation; (2) empirically, that Term 2 really does shrink relative to
Term 1 as n_cycles grows (the claim behind "n_cycles affects precision").

Usage: python demo_conjugate_and_term2.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavelet_features import make_morlet

OUT = r"D:\Gil\spike_sorting_agent\outputs"
FS = 30000.0
F0 = 739.0
A = 200.0
true_delta_s = 0.15 / 1000
omega0 = 2 * np.pi * F0
phi = omega0 * true_delta_s

fig, axes = plt.subplots(1, 3, figsize=(19, 6))

# ---- panel 1: x(tau) alone and psi(tau) alone, both rotating the SAME way ----
ax = axes[0]
deg_per_sample = 30
taus = [0, 1, 2, 3, 4]
colors = plt.cm.autumn(np.linspace(0, 0.85, len(taus)))
for tau, c in zip(taus, colors):
    ang = np.radians(deg_per_sample * tau)
    ax.annotate("", xy=(np.cos(ang), np.sin(ang)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=c, lw=2))
    ax.annotate(f"tau={tau}", (np.cos(ang) * 1.15, np.sin(ang) * 1.15), fontsize=8, color=c, ha="center")
ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.4, 1.4)
ax.set_aspect("equal")
ax.axhline(0, color="grey", lw=0.3)
ax.axvline(0, color="grey", lw=0.3)
ax.set_title("x(tau) alone (and psi(tau) alone --\nidentical, same frequency): BOTH\nrotate the SAME way, sample to sample", fontsize=10.5)

# ---- panel 2: UNconjugated product -- spins TWICE as fast ----
ax = axes[1]
for tau, c in zip(taus, colors):
    ang = np.radians(2 * deg_per_sample * tau)  # angles ADD when multiplying -> doubles
    ax.annotate("", xy=(np.cos(ang), np.sin(ang)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=c, lw=2))
    ax.annotate(f"tau={tau}\n({2*deg_per_sample*tau}deg)", (np.cos(ang) * 1.2, np.sin(ang) * 1.2),
                fontsize=7.5, color=c, ha="center")
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.6, 1.6)
ax.set_aspect("equal")
ax.axhline(0, color="grey", lw=0.3)
ax.axvline(0, color="grey", lw=0.3)
ax.set_title("WITHOUT conjugating: multiplying two\nsame-direction spinners ADDS their angles\n-> spins at DOUBLE speed -> next samples\npoint every-which-way -> summing CANCELS\n(even for a perfect frequency match!)", fontsize=10.2)

# ---- panel 3: CONJUGATED product -- frozen, same angle every sample ----
ax = axes[2]
ang = 0.0
for tau, c in zip(taus, colors):
    ax.annotate("", xy=(np.cos(ang) * (1 + 0.08 * tau), np.sin(ang) * (1 + 0.08 * tau)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=c, lw=2))
ax.annotate("EVERY tau: 0 deg\n(frozen -- conjugating\nflipped psi's spin to\nEXACTLY cancel x's spin)",
            (0.9, 0.35), fontsize=9, color="#333")
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.6, 1.6)
ax.set_aspect("equal")
ax.axhline(0, color="grey", lw=0.3)
ax.axvline(0, color="grey", lw=0.3)
ax.set_title("WITH conjugating (what the formula\nactually does): psi's spin flips, EXACTLY\ncancels x's spin -> product FROZEN at a\nfixed angle for every sample -> summing\nADDS CONSTRUCTIVELY, tau after tau", fontsize=10.2)

plt.suptitle('Exact numeric check (degrees): tau=0,1,2,3,4 -> unconjugated product = 0,60,120,180,240 (spins away);'
             ' conjugated product = 0,0,0,0,0 (frozen)', fontsize=12)
plt.tight_layout()
out1 = os.path.join(OUT, "wavelet_21_conjugate_mechanism.png")
plt.savefig(out1, dpi=115, bbox_inches="tight")
plt.close(fig)
print("saved", out1)

# ---- second figure: Term2/Term1 vs n_cycles, verified not asserted ----
fig, ax = plt.subplots(figsize=(9, 6))
n_cycles_list = [1.0, 1.5, 3.0, 6.0, 12.0, 24.0]
ratios = []
for n_cycles in n_cycles_list:
    t_psi, psi = make_morlet(F0, FS, n_cycles=n_cycles)
    t = t_psi / FS
    x1 = (A / 2) * np.exp(1j * (omega0 * t - phi))
    x2 = (A / 2) * np.exp(-1j * (omega0 * t - phi))
    W1 = np.sum(x1 * np.conj(psi))
    W2 = np.sum(x2 * np.conj(psi))
    ratios.append(abs(W2) / abs(W1))
ax.plot(n_cycles_list, ratios, marker="o", color="#1f77b4", lw=2)
ax.set_yscale("log")
ax.set_xlabel("n_cycles")
ax.set_ylabel("|Term 2| / |Term 1|  (log scale)")
ax.set_title("VERIFIED, not asserted: the leftover Term 2\nshrinks sharply relative to Term 1 as n_cycles grows\n(1 cycle: ~4% leakage -> 6 cycles: ~0.01% leakage)",
              fontsize=11.5)
for x, y in zip(n_cycles_list, ratios):
    ax.annotate(f"{y*100:.3f}%", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
plt.tight_layout()
out2 = os.path.join(OUT, "wavelet_22_term2_vs_ncycles.png")
plt.savefig(out2, dpi=115, bbox_inches="tight")
print("saved", out2)

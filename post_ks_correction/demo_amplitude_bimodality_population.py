"""
Is the amplitude finding a property of three neurons, or of the population?

The 370x amplitude-over-timing result in RESEARCH_LOG 5z rests on a median of
THREE injected units (302, 408, 440); the other two never fragmented, so they
contributed nothing. Three numbers, one session, one animal. That is too thin
a base for the conclusion that was drawn from it, and this widens it before
any more work is built on top.

TWO THINGS MEASURED, both on the real session rather than injected data:

  1. How bimodal is each real unit's amplitude distribution, using
     KILOSORT'S OWN `bimod_score` from swarmsplitter.py -- copied verbatim
     below rather than reimplemented, so the number is exactly the one that
     drives its split decisions. Its threshold is score >= 0.6 keeps a split,
     < 0.6 merges. A unit already sitting above 0.6 on amplitude alone is a
     unit Kilosort has reason to cut in two.

  2. How much of each unit's amplitude spread is explainable by BURSTING --
     correlating each spike's amplitude against the time since that unit's
     previous spike. If amplitude drops within bursts and recovers, that
     correlation should be positive and that is the mechanism the original
     handoff document predicted.

Kilosort's own `amplitudes.npy` is used, so this needs no raw voltage and no
sorting run -- it is a population-scale check that costs seconds.

A CAVEAT THAT MATTERS: `bimod_score` expects its input roughly on a [-2, 2]
scale (it histograms over exactly that range and looks for a dip near the
centre). Amplitudes are positive and arbitrarily scaled, so each unit's
amplitudes are z-scored first. That is a judgement call, not a neutral step,
and it is why the raw dip statistic is reported alongside.

Usage: python demo_amplitude_bimodality_population.py
"""
import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
FS = 30000.0
MIN_SPIKES = 300


def bimod_score(xproj):
    """Verbatim from kilosort/swarmsplitter.py so the number is the one
    Kilosort actually acts on."""
    xbin, _ = np.histogram(xproj, np.linspace(-2, 2, 400))
    xbin = gaussian_filter1d(xbin.astype("float32"), 4)
    imin = np.argmin(xbin[175:225])
    xmin = np.min(xbin[175:225])
    xm1 = np.max(xbin[:imin + 175])
    xm2 = np.max(xbin[imin + 175:])
    if xm1 <= 0 or xm2 <= 0:
        return np.nan
    return float(1 - np.maximum(xmin / xm1, xmin / xm2))


st = np.load(KS + r"\spike_times.npy").ravel()
cl = np.load(KS + r"\spike_clusters.npy").ravel()
amp = np.load(KS + r"\amplitudes.npy").ravel()
templates = np.load(KS + r"\templates.npy")
cp = np.load(KS + r"\channel_positions.npy")
peak_ch = np.argmax(templates.max(axis=1) - templates.min(axis=1), axis=1)

units, counts = np.unique(cl, return_counts=True)
units = units[counts >= MIN_SPIKES]
print(f"session 20260916_110311: {len(units)} units with >= {MIN_SPIKES} spikes "
      f"(of {len(np.unique(cl))} total)\n")

rows = []
for u in units:
    m = cl == u
    a = amp[m].astype(np.float64)
    t = np.sort(st[m]).astype(np.float64)
    if a.std() <= 0:
        continue
    z = (a - a.mean()) / a.std()
    score = bimod_score(z)

    # bursting: amplitude vs time since the unit's own previous spike
    ao = amp[m][np.argsort(st[m])].astype(np.float64)
    isi = np.diff(t) / FS * 1000.0            # ms
    if len(isi) > 50:
        li = np.log10(np.clip(isi, 0.1, None))
        r_burst = float(np.corrcoef(li, ao[1:])[0, 1])
        # the burst signature specifically: are short-ISI spikes smaller?
        short = ao[1:][isi < 10.0]
        longg = ao[1:][isi >= 100.0]
        burst_drop = (float(1 - short.mean() / longg.mean())
                      if len(short) >= 20 and len(longg) >= 20 and longg.mean() > 0
                      else np.nan)
        frac_short = float((isi < 10.0).mean())
    else:
        r_burst, burst_drop, frac_short = np.nan, np.nan, np.nan

    rows.append(dict(unit=int(u), n_spikes=int(m.sum()), peak_ch=int(peak_ch[u]),
                     amp_mean=round(float(a.mean()), 3),
                     amp_cv=round(float(a.std() / a.mean()), 4),
                     bimod_score=round(score, 4) if np.isfinite(score) else np.nan,
                     r_amp_vs_logisi=round(r_burst, 4) if np.isfinite(r_burst) else np.nan,
                     burst_amp_drop=round(burst_drop, 4) if np.isfinite(burst_drop) else np.nan,
                     frac_isi_under_10ms=round(frac_short, 4) if np.isfinite(frac_short) else np.nan))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "amplitude_bimodality_population.csv"), index=False)

ok = df[df["bimod_score"].notna()]
print("AMPLITUDE BIMODALITY, using Kilosort's own bimod_score")
print(f"  Kilosort keeps a split when this score >= 0.6")
for q in (10, 25, 50, 75, 90):
    print(f"    p{q:<3} = {np.percentile(ok['bimod_score'], q):.3f}")
above = float((ok["bimod_score"] >= 0.6).mean())
print(f"  units at or above the 0.6 split threshold on AMPLITUDE ALONE: "
      f"{above:.1%} ({int((ok['bimod_score']>=0.6).sum())} of {len(ok)})")

print(f"\nAMPLITUDE VARIABILITY")
print(f"  median coefficient of variation: {ok['amp_cv'].median():.3f}")
print(f"  p90 coefficient of variation:    {np.percentile(ok['amp_cv'],90):.3f}")

b = df[df["r_amp_vs_logisi"].notna()]
print(f"\nBURSTING AS THE MECHANISM ({len(b)} units with enough spikes)")
print(f"  correlation of amplitude with log(previous ISI):")
print(f"    median {b['r_amp_vs_logisi'].median():+.3f}, "
      f"positive in {float((b['r_amp_vs_logisi']>0).mean()):.1%} of units")
d = b[b["burst_amp_drop"].notna()]
if len(d):
    print(f"  amplitude drop for spikes with ISI < 10ms vs ISI > 100ms:")
    print(f"    median {d['burst_amp_drop'].median():+.1%}, "
          f"a real drop (>2%) in {float((d['burst_amp_drop']>0.02).mean()):.1%} of units")
    print(f"  median fraction of spikes with ISI < 10 ms: "
          f"{d['frac_isi_under_10ms'].median():.3f}")

print(f"\nTHE FIVE UNITS THE EARLIER RESULT RESTED ON")
sel = df[df["unit"].isin([302, 408, 439, 440, 31])]
print(sel[["unit", "n_spikes", "amp_cv", "bimod_score", "r_amp_vs_logisi",
           "burst_amp_drop"]].to_string(index=False))
print(f"\n  for comparison, population medians: "
      f"amp_cv {ok['amp_cv'].median():.3f}, bimod {ok['bimod_score'].median():.3f}")
print("\nsaved amplitude_bimodality_population.csv")

"""
Survey every channel on the three criteria that define an injection site, so
tiers can be built from MEASURED site properties instead of a single density
ranking.

WHY THIS EXISTS. The previous builder ranked channels by one number -- the
count of spikes whose peak channel sits within +-6 -- and called the bottom of
that list "quiet". Section 5ag showed what that bought: the "quiet" easy sites
still hosted 5-11 neurons within 60 um firing ~10-15 spikes/s, so easy-tier
precision (0.538) was mostly measuring a merge with a resident neuron. A site
has to be chosen on what actually makes it hard, and those are three separate
things that do not have to agree:

  1. NOISE LEVEL      the background voltage fluctuation with spikes excluded.
                      Robust (MAD) sigma of the 300 Hz-highpassed trace, in uV.
                      This is thermal/biological noise, not other cells.
  2. EVENT DENSITY    how many spikes Kilosort already detects nearby. A site
                      can be electrically calm yet sit in a busy neighbourhood.
  3. NEAREST UNIT     distance to the closest sorted cluster's peak channel.
                      This is what decides whether an injected neuron gets
                      merged with a resident, which is the failure 5ag found.

THE TIER REQUIREMENTS THESE SERVE (Gil's design):
  easy       low noise, few events, NO units nearby -- genuinely easy
  hard       the exact opposite on all three
  collision  QUIET, like easy -- the challenge is temporal separation, so
             channel noise must not be confounded with it
  pair       QUIET, like easy -- the challenge is separating two neurons, so
             channel noise must not be confounded with it either

That last pair of requirements is the real change. Previously collision and
pair sat on "quietest" by density alone, so their difficulty mixed two causes.

WHAT IT REPORTS. Per-channel values, the joint distribution, and then the
feasibility question that decides the whole experiment design: how many
NON-OVERLAPPING quiet sites and hard sites does this probe actually contain?
Easy + collision + pair all draw from the quiet pool, so if the probe holds
only a handful of quiet slots, the benchmark has to span several recordings
rather than cram units together.

Usage: python survey_probe_sites.py
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
DAT_PATH = (r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort"
            r"\20260916_110311.probe1.dat")
OUT = r"D:\Gil\spike_sorting_agent\outputs"

N_CHAN_BIN, ITEMSIZE, FS = 384, 2, 30000.0
GAIN_TO_UV = 0.018311105685598315
BG_START = 60_000_000
DURATION_S = 120.0
N_SAMPLES = int(DURATION_S * FS)

NOISE_CHUNKS = 120          # chunks sampled across the window for noise
NOISE_CHUNK_LEN = 3000      # 100 ms each
DENSITY_HALFWIDTH = 6       # +-6 channels, as before, for comparability
MIN_CLUSTER_SPIKES = 100    # a cluster must fire this much to count as "a unit"
EDGE_GUARD = 25             # no site within this many channels of either end

# A site qualifies as QUIET only if it passes all three, as percentiles of the
# probe's own distribution -- absolute thresholds would not transfer.
QUIET_NOISE_PCTL = 40
QUIET_DENSITY_PCTL = 25
QUIET_MIN_UNIT_DIST_UM = 60.0
HARD_NOISE_PCTL = 60
HARD_DENSITY_PCTL = 75
HARD_MAX_UNIT_DIST_UM = 30.0

SITE_SPACING = 22           # channels between distinct injection sites

b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")
TOTAL = os.path.getsize(DAT_PATH) // (N_CHAN_BIN * ITEMSIZE)
assert BG_START + N_SAMPLES <= TOTAL

cp = np.load(os.path.join(KS, "channel_positions.npy"))
templates = np.load(os.path.join(KS, "templates.npy"))
peak_ch = np.argmax(templates.max(axis=1) - templates.min(axis=1), axis=1)
st = np.load(os.path.join(KS, "spike_times.npy")).ravel()
cl = np.load(os.path.join(KS, "spike_clusters.npy")).ravel()
inwin = (st >= BG_START) & (st < BG_START + N_SAMPLES)
cl_w = cl[inwin]

# ---------------------------------------------------------------- 1. noise
print(f"measuring noise on {N_CHAN_BIN} channels from {NOISE_CHUNKS} chunks "
      f"of {NOISE_CHUNK_LEN/FS*1000:.0f} ms ...")
rng = np.random.default_rng(0)
starts = np.linspace(BG_START, BG_START + N_SAMPLES - NOISE_CHUNK_LEN,
                     NOISE_CHUNKS).astype(np.int64)
mads = np.zeros((NOISE_CHUNKS, N_CHAN_BIN))
with open(DAT_PATH, "rb") as f:
    for i, s in enumerate(starts):
        f.seek(int(s) * N_CHAN_BIN * ITEMSIZE)
        blk = np.frombuffer(f.read(NOISE_CHUNK_LEN * N_CHAN_BIN * ITEMSIZE),
                            dtype=np.int16).reshape(NOISE_CHUNK_LEN, N_CHAN_BIN)
        x = filtfilt(b_hp, a_hp, blk.astype(np.float64), axis=0)
        # MAD -> sigma is robust: a few spikes in the chunk cannot inflate it
        mads[i] = np.median(np.abs(x - np.median(x, axis=0)), axis=0)
noise_uv = np.median(mads, axis=0) / 0.6745 * GAIN_TO_UV
print(f"  noise: {noise_uv.min():.1f} - {noise_uv.max():.1f} uV "
      f"(median {np.median(noise_uv):.1f})")

# ------------------------------------------------------------- 2. density
busy = np.zeros(N_CHAN_BIN)
for cid, n in zip(*np.unique(cl_w, return_counts=True)):
    if cid < len(peak_ch):
        busy[peak_ch[cid]] += n
density = np.array([busy[max(0, i - DENSITY_HALFWIDTH):i + DENSITY_HALFWIDTH + 1].sum()
                    for i in range(N_CHAN_BIN)])
print(f"  event density (+-{DENSITY_HALFWIDTH} ch, 120 s): "
      f"{density.min():.0f} - {density.max():.0f} "
      f"(median {np.median(density):.0f})")

# --------------------------------------------------------- 3. nearest unit
uids, ucounts = np.unique(cl_w, return_counts=True)
real_units = [int(u) for u, n in zip(uids, ucounts)
              if n >= MIN_CLUSTER_SPIKES and u < len(peak_ch)]
upk = np.array([peak_ch[u] for u in real_units])
print(f"  {len(real_units)} clusters with >= {MIN_CLUSTER_SPIKES} spikes in "
      f"the window count as resident units")
nearest_um = np.array([
    float(np.min(np.sqrt(((cp[upk] - cp[c]) ** 2).sum(axis=1))))
    for c in range(N_CHAN_BIN)])
n_within_60 = np.array([
    int((np.sqrt(((cp[upk] - cp[c]) ** 2).sum(axis=1)) <= 60.0).sum())
    for c in range(N_CHAN_BIN)])
print(f"  nearest resident unit: {nearest_um.min():.0f} - "
      f"{nearest_um.max():.0f} um (median {np.median(nearest_um):.0f})")

df = pd.DataFrame(dict(
    channel=np.arange(N_CHAN_BIN), x_um=cp[:, 0], y_um=cp[:, 1],
    noise_uv=noise_uv.round(2), density=density.astype(int),
    nearest_unit_um=nearest_um.round(1), n_units_within_60um=n_within_60))
df["edge_ok"] = (df.channel >= EDGE_GUARD) & (df.channel < N_CHAN_BIN - EDGE_GUARD)

nz_q = np.percentile(noise_uv[df.edge_ok], QUIET_NOISE_PCTL)
dn_q = np.percentile(density[df.edge_ok], QUIET_DENSITY_PCTL)
nz_h = np.percentile(noise_uv[df.edge_ok], HARD_NOISE_PCTL)
dn_h = np.percentile(density[df.edge_ok], HARD_DENSITY_PCTL)

df["is_quiet"] = (df.edge_ok & (df.noise_uv <= nz_q) & (df.density <= dn_q)
                  & (df.nearest_unit_um >= QUIET_MIN_UNIT_DIST_UM))
df["is_hard"] = (df.edge_ok & (df.noise_uv >= nz_h) & (df.density >= dn_h)
                 & (df.nearest_unit_um <= HARD_MAX_UNIT_DIST_UM))
df.to_csv(os.path.join(OUT, "probe_site_survey.csv"), index=False)

print("\n" + "=" * 78)
print("THRESHOLDS (percentiles of this probe's own distribution)")
print(f"  QUIET: noise <= {nz_q:.1f} uV (p{QUIET_NOISE_PCTL}), "
      f"density <= {dn_q:.0f} (p{QUIET_DENSITY_PCTL}), "
      f"nearest unit >= {QUIET_MIN_UNIT_DIST_UM:.0f} um")
print(f"  HARD : noise >= {nz_h:.1f} uV (p{HARD_NOISE_PCTL}), "
      f"density >= {dn_h:.0f} (p{HARD_DENSITY_PCTL}), "
      f"nearest unit <= {HARD_MAX_UNIT_DIST_UM:.0f} um")
print(f"\nqualifying CHANNELS: {int(df.is_quiet.sum())} quiet, "
      f"{int(df.is_hard.sum())} hard, of {int(df.edge_ok.sum())} eligible")

# ---- do the three criteria agree? if they did, one ranking would have sufficed
e = df[df.edge_ok]
print("\nDO THE CRITERIA AGREE? (Spearman, on eligible channels)")
print(f"  noise vs density      : {e.noise_uv.corr(e.density, method='spearman'):+.3f}")
print(f"  noise vs nearest unit : {e.noise_uv.corr(e.nearest_unit_um, method='spearman'):+.3f}")
print(f"  density vs nearest    : {e.density.corr(e.nearest_unit_um, method='spearman'):+.3f}")
print("  (weak correlations mean the old single-density ranking was NOT a")
print("   proxy for the other two, which is why 5ag found residents nearby)")


def pack(mask, spacing=SITE_SPACING):
    """Greedily pack non-overlapping sites into the qualifying channels."""
    out = []
    for c in np.where(mask)[0]:
        if all(abs(int(c) - s) >= spacing for s in out):
            out.append(int(c))
    return out


quiet_sites = pack(df.is_quiet.to_numpy())
hard_sites = pack(df.is_hard.to_numpy())
print("\n" + "=" * 78)
print(f"FEASIBILITY: non-overlapping sites at {SITE_SPACING}-channel spacing")
print(f"  QUIET sites available: {len(quiet_sites)}  -> {quiet_sites}")
print(f"  HARD  sites available: {len(hard_sites)}  -> {hard_sites}")
print(f"\n  easy + collision + pair ALL draw from the quiet pool.")
print(f"  per recording that is at most {len(quiet_sites)} quiet placements "
      f"and {len(hard_sites)} hard ones.")

json.dump(dict(quiet_sites=quiet_sites, hard_sites=hard_sites,
               noise_quiet_thresh=float(nz_q), density_quiet_thresh=float(dn_q),
               noise_hard_thresh=float(nz_h), density_hard_thresh=float(dn_h),
               quiet_min_unit_dist_um=QUIET_MIN_UNIT_DIST_UM,
               hard_max_unit_dist_um=HARD_MAX_UNIT_DIST_UM,
               site_spacing=SITE_SPACING),
          open(os.path.join(OUT, "probe_sites.json"), "w"), indent=2)

print("\nthe 12 quietest eligible channels, with all three criteria:")
print(e.nsmallest(12, "density")[
    ["channel", "noise_uv", "density", "nearest_unit_um",
     "n_units_within_60um", "is_quiet"]].to_string(index=False))
print("\nthe 8 busiest eligible channels:")
print(e.nlargest(8, "density")[
    ["channel", "noise_uv", "density", "nearest_unit_um",
     "n_units_within_60um", "is_hard"]].to_string(index=False))
print("\nsaved probe_site_survey.csv, probe_sites.json")

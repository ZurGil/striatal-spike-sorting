"""
Score every channel by the AMPLITUDE of the activity on it, not the count of
units near it -- and establish the Trodes <-> Kilosort channel mapping.

WHY THIS REPLACES THE COUNT-BASED SURVEY. survey_probe_sites.py required "no
units nearby" and found ZERO qualifying channels: this striatal probe has 308
clusters over 384 channels, and the most isolated channel on the shank is 52 um
from the nearest unit. But Gil's actual criterion is different and achievable --
what matters is not whether neighbouring units EXIST, it is whether they are
LARGE. A neighbour at 20 uV is background; a neighbour at 300 uV is a competitor
that will swallow an injected neuron into its cluster (which is what section
5ag found happening).

So the quantity to measure is the amplitude a site actually sees:

  noise_uv          MAD sigma of the 300 Hz-highpassed trace. Thermal and
                    biological noise, spikes excluded by the robust estimator.
  p999_uv           99.9th percentile of |voltage|. How big the large events on
                    this channel are, whatever their source. Needs no unit
                    identity, so no sorting errors leak into it.
  max_neighbor_uV   the single largest resident unit as seen ON THIS CHANNEL,
                    from its REAL averaged waveform in microvolts. This is the
                    direct measure of "is there a big competitor here".
  sum_neighbor_uV   all residents summed, for total competing drive.

HOW THE WAVEFORMS ARE MEASURED CHEAPLY. Averaging 308 units x 250 snippets
would mean ~10 GB of scattered reads. Instead this makes ONE pass over sampled
contiguous chunks of the background window, highpasses each chunk, and
accumulates every unit's snippets from it. Units get fewer snippets than a
dedicated pass would give, which is fine for an amplitude estimate.

THE TRODES MAPPING. The .rec header lists 384 <SpikeNTrode> entries in .dat
column order with ids descending 1466 -> 1083, and each carries the pad
coordinates. This script VERIFIES that ntrode = 1466 - ks_channel by checking
the coordinates agree with channel_positions.npy, rather than trusting the
arithmetic.

Usage: python survey_probe_amplitude.py
"""
import os
import re
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

REC = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.rec"
KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
DAT_PATH = (r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort"
            r"\20260916_110311.probe1.dat")
OUT = r"D:\Gil\spike_sorting_agent\outputs"

N_CHAN_BIN, ITEMSIZE, FS = 384, 2, 30000.0
GAIN_TO_UV = 0.018311105685598315
BG_START, DURATION_S = 60_000_000, 120.0
N_SAMPLES = int(DURATION_S * FS)
NT, NT0MIN = 61, 20

N_CHUNKS, CHUNK_LEN = 300, 6000      # 300 x 200 ms = 60 s of the window
MIN_CLUSTER_SPIKES = 100
NEIGHBOR_RADIUS_UM = 60.0
DENSITY_HALFWIDTH = 6
EDGE_GUARD = 25
SITE_SPACING = 22

b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")
cp = np.load(os.path.join(KS, "channel_positions.npy"))
templates = np.load(os.path.join(KS, "templates.npy"))
peak_ch = np.argmax(templates.max(axis=1) - templates.min(axis=1), axis=1)
st = np.load(os.path.join(KS, "spike_times.npy")).ravel()
cl = np.load(os.path.join(KS, "spike_clusters.npy")).ravel()

# ------------------------------------------------- 0. the Trodes mapping
print("=" * 78)
print("TRODES <-> KILOSORT CHANNEL MAPPING")
with open(REC, "rb") as f:
    head = f.read(4_000_000)
cfg = head[:head.find(b"</Configuration>") + 16].decode("latin-1")
nt_ids = [int(m) for m in re.findall(r'<SpikeNTrode[^>]*\sid="(\d+)"', cfg)]
sc = re.findall(r'<SpikeChannel\s[^>]*/>', cfg)
sc_ml = [int(re.search(r'coord_ml="(-?\d+)"', s).group(1)) for s in sc]
sc_dv = [int(re.search(r'coord_dv="(-?\d+)"', s).group(1)) for s in sc]
assert len(nt_ids) == len(sc) == N_CHAN_BIN

ml_ok = np.allclose(np.array(sc_ml, float), cp[:, 0])
dv_ok = np.allclose(np.array(sc_dv, float), cp[:, 1])
print(f"  config order matches channel_positions.npy: "
      f"ml {ml_ok}, dv {dv_ok}")
nt = np.array(nt_ids)
linear = np.array_equal(nt, nt[0] - np.arange(N_CHAN_BIN))
print(f"  nTrode ids run {nt[0]} -> {nt[-1]}, strictly descending by 1: {linear}")
assert ml_ok and dv_ok and linear, "mapping is not the simple linear one"
NT_BASE = int(nt[0])
print(f"\n  ==> ntrode = {NT_BASE} - ks_channel")
print(f"  ==> ks_channel = {NT_BASE} - ntrode")
for a, b in ((1275, 1211),):
    print(f"\n  Gil's quiet stretch, Trodes {a}-{b}  ->  "
          f"Kilosort channels {NT_BASE-a}-{NT_BASE-b}")
USER_LO, USER_HI = NT_BASE - 1275, NT_BASE - 1211

# ------------------------------------------- 1. noise + big-event amplitude
uids, ucnt = np.unique(cl[(st >= BG_START) & (st < BG_START + N_SAMPLES)],
                       return_counts=True)
units = [int(u) for u, n in zip(uids, ucnt)
         if n >= MIN_CLUSTER_SPIKES and u < len(peak_ch)]
print(f"\n{len(units)} resident units with >= {MIN_CLUSTER_SPIKES} spikes")
trains = {u: np.sort(st[cl == u]) for u in units}

starts = np.linspace(BG_START, BG_START + N_SAMPLES - CHUNK_LEN,
                     N_CHUNKS).astype(np.int64)
acc = {u: np.zeros((NT, N_CHAN_BIN)) for u in units}
cnt = {u: 0 for u in units}
mads, p999 = np.zeros((N_CHUNKS, N_CHAN_BIN)), np.zeros((N_CHUNKS, N_CHAN_BIN))

print(f"one pass over {N_CHUNKS} chunks of {CHUNK_LEN/FS*1000:.0f} ms "
      f"({N_CHUNKS*CHUNK_LEN/FS:.0f} s total) ...")
with open(DAT_PATH, "rb") as f:
    for i, s0 in enumerate(starts):
        f.seek(int(s0) * N_CHAN_BIN * ITEMSIZE)
        blk = np.frombuffer(f.read(CHUNK_LEN * N_CHAN_BIN * ITEMSIZE),
                            dtype=np.int16).reshape(CHUNK_LEN, N_CHAN_BIN)
        x = filtfilt(b_hp, a_hp, blk.astype(np.float64), axis=0)
        mads[i] = np.median(np.abs(x - np.median(x, axis=0)), axis=0)
        p999[i] = np.percentile(np.abs(x), 99.9, axis=0)
        lo, hi = int(s0), int(s0) + CHUNK_LEN
        for u in units:
            t = trains[u]
            sel = t[(t >= lo + NT0MIN + 5) & (t < hi - (NT - NT0MIN) - 5)]
            for sp in sel:
                j = int(sp) - lo - NT0MIN
                acc[u] += x[j:j + NT]
                cnt[u] += 1
        if (i + 1) % 60 == 0:
            print(f"  {i+1}/{N_CHUNKS} chunks")

noise_uv = np.median(mads, axis=0) / 0.6745 * GAIN_TO_UV
p999_uv = np.median(p999, axis=0) * GAIN_TO_UV
print(f"  noise  : {noise_uv.min():.1f} - {noise_uv.max():.1f} uV "
      f"(median {np.median(noise_uv):.1f})")
print(f"  p99.9  : {p999_uv.min():.1f} - {p999_uv.max():.1f} uV "
      f"(median {np.median(p999_uv):.1f})")

# ---- per-unit amplitude on every channel, from its real averaged waveform
ok = [u for u in units if cnt[u] >= 10]
print(f"  {len(ok)} of {len(units)} units got >= 10 snippets "
      f"(median {int(np.median([cnt[u] for u in units]))})")
amp = np.zeros((len(ok), N_CHAN_BIN))
for k, u in enumerate(ok):
    M = acc[u] / cnt[u] * GAIN_TO_UV
    amp[k] = M.max(axis=0) - M.min(axis=0)
upk = np.array([peak_ch[u] for u in ok])
print(f"  unit peak amplitudes: {amp.max(axis=1).min():.0f} - "
      f"{amp.max(axis=1).max():.0f} uV "
      f"(median {np.median(amp.max(axis=1)):.0f})")

# neighbour amplitude AS SEEN ON each channel, restricted to units whose peak
# is within NEIGHBOR_RADIUS_UM (a distant unit's tail is not a competitor)
maxn, sumn, nnb = np.zeros(N_CHAN_BIN), np.zeros(N_CHAN_BIN), np.zeros(N_CHAN_BIN, int)
for c in range(N_CHAN_BIN):
    d = np.sqrt(((cp[upk] - cp[c]) ** 2).sum(axis=1))
    near = np.where(d <= NEIGHBOR_RADIUS_UM)[0]
    if len(near):
        a = amp[near, c]
        maxn[c], sumn[c], nnb[c] = a.max(), a.sum(), len(near)

busy = np.zeros(N_CHAN_BIN)
cl_w = cl[(st >= BG_START) & (st < BG_START + N_SAMPLES)]
for cid, n in zip(*np.unique(cl_w, return_counts=True)):
    if cid < len(peak_ch):
        busy[peak_ch[cid]] += n
density = np.array([busy[max(0, i - DENSITY_HALFWIDTH):i + DENSITY_HALFWIDTH + 1].sum()
                    for i in range(N_CHAN_BIN)]).astype(int)

df = pd.DataFrame(dict(
    ks_channel=np.arange(N_CHAN_BIN), ntrode=NT_BASE - np.arange(N_CHAN_BIN),
    x_um=cp[:, 0], y_um=cp[:, 1], noise_uv=noise_uv.round(2),
    p999_uv=p999_uv.round(1), max_neighbor_uv=maxn.round(1),
    sum_neighbor_uv=sumn.round(1), n_neighbors=nnb, density=density))
df["edge_ok"] = ((df.ks_channel >= EDGE_GUARD)
                 & (df.ks_channel < N_CHAN_BIN - EDGE_GUARD))
df.to_csv(os.path.join(OUT, "probe_amplitude_survey.csv"), index=False)

# ------------------------------------- 2. does Gil's visual read hold up?
print("\n" + "=" * 78)
print(f"GIL'S STRETCH: Trodes 1275-1211 = KS channels {USER_LO}-{USER_HI}")
inu = (df.ks_channel >= USER_LO) & (df.ks_channel <= USER_HI)
e = df[df.edge_ok]
for col in ("noise_uv", "p999_uv", "max_neighbor_uv", "sum_neighbor_uv", "density"):
    v, rest = df.loc[inu, col], e.loc[~inu[e.index], col]
    pct = float((rest < v.median()).mean() * 100)
    print(f"  {col:<17} median {v.median():>8.1f}  vs rest of probe "
          f"{rest.median():>8.1f}   -> p{pct:.0f} of the probe")

print("\n" + "=" * 78)
print("THE 20 QUIETEST CHANNELS BY max_neighbor_uv (eligible only)")
q = e.nsmallest(20, "max_neighbor_uv")
print(q[["ks_channel", "ntrode", "noise_uv", "p999_uv", "max_neighbor_uv",
         "sum_neighbor_uv", "n_neighbors", "density"]].to_string(index=False))
print("\nTHE 10 LOUDEST CHANNELS BY max_neighbor_uv")
print(e.nlargest(10, "max_neighbor_uv")[
    ["ks_channel", "ntrode", "noise_uv", "p999_uv", "max_neighbor_uv",
     "sum_neighbor_uv", "n_neighbors", "density"]].to_string(index=False))

print("\nCRITERIA AGREEMENT (Spearman, eligible channels)")
for a, b in (("noise_uv", "max_neighbor_uv"), ("density", "max_neighbor_uv"),
             ("p999_uv", "max_neighbor_uv"), ("noise_uv", "density")):
    print(f"  {a:<16} vs {b:<17}: {e[a].corr(e[b], method='spearman'):+.3f}")


def pack(channels, spacing=SITE_SPACING):
    out = []
    for c in channels:
        if all(abs(int(c) - s) >= spacing for s in out):
            out.append(int(c))
    return out


QUIET_N = 35.0     # percentile cut used for reporting feasibility
nz = np.percentile(e.noise_uv, QUIET_N)
mn = np.percentile(e.max_neighbor_uv, QUIET_N)
quiet = e[(e.noise_uv <= nz) & (e.max_neighbor_uv <= mn)]
loud = e[(e.noise_uv >= np.percentile(e.noise_uv, 100 - QUIET_N))
         & (e.max_neighbor_uv >= np.percentile(e.max_neighbor_uv, 100 - QUIET_N))]
qs = pack(quiet.sort_values("max_neighbor_uv").ks_channel.tolist())
ls = pack(loud.sort_values("max_neighbor_uv", ascending=False).ks_channel.tolist())
print("\n" + "=" * 78)
print(f"FEASIBILITY at p{QUIET_N:.0f} cuts (noise <= {nz:.1f} uV, "
      f"max neighbour <= {mn:.0f} uV)")
print(f"  QUIET channels {len(quiet)} -> {len(qs)} non-overlapping sites "
      f"at {SITE_SPACING}-ch spacing")
print(f"    {qs}")
print(f"    as Trodes: {[NT_BASE - c for c in qs]}")
print(f"  LOUD  channels {len(loud)} -> {len(ls)} non-overlapping sites")
print(f"    {ls}")
print(f"    as Trodes: {[NT_BASE - c for c in ls]}")

json.dump(dict(nt_base=NT_BASE, quiet_sites=qs, loud_sites=ls,
               noise_cut=float(nz), max_neighbor_cut=float(mn),
               site_spacing=SITE_SPACING,
               user_range_ks=[int(USER_LO), int(USER_HI)]),
          open(os.path.join(OUT, "probe_amplitude_sites.json"), "w"), indent=2)
print("\nsaved probe_amplitude_survey.csv, probe_amplitude_sites.json")

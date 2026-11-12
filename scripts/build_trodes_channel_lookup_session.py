"""
Session-parameterized version of build_trodes_channel_lookup.py (original hardcoded
to 20260901_085606 -- leave that one alone, it's fine as a reference).

If the session has no standalone <session>.trodesconf file (not all do -- e.g.
20260911_100049 didn't), extract one from the raw .rec file's own embedded header:
the SpikeNTrode/SpikeChannel XML config is embedded directly in every .rec file's
header (first few hundred KB), terminated by </SpikeConfiguration> or </Configuration>.
See WORKFLOW.md -- same trick already used to verify gain_to_uV per-session.

Usage: python build_trodes_channel_lookup_session.py <session_id> <rec_root>
  <rec_root> e.g. F:\\Gil\\Shamir\\20260911_100049.rec
"""
import re, os, sys, json
import numpy as np
import pandas as pd

SESSION_ID = sys.argv[1]
REC_ROOT = sys.argv[2]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from striatal_agent.paths import out_path

# A standalone .trodesconf is "rarely present" on this PC, where the .rec is always at
# hand to read the header out of. On the cluster it is the ONLY route: scratch holds
# probe1.dat and the map json, never the 300 GB .rec. stage_session_to_ris.ps1 therefore
# extracts this file here and ships it with the data, for the same reason it generates
# the probe map here -- the thing needed to make it cannot exist on a compute node.
TRODESCONF_STANDALONE_GUESS = os.path.join(REC_ROOT, f"{SESSION_ID}.trodesconf")
REC_FILE = os.path.join(REC_ROOT, f"{SESSION_ID}.rec")
CHANMAP_JSON = os.path.join(REC_ROOT, f"{SESSION_ID}.kilosort",
                            f"{SESSION_ID}_custom_neuropixels_map.json")
_KSO = os.path.join(REC_ROOT, f"{SESSION_ID}.kilosort", "kilosort4", "kilosort_original")
TEMPLATES = os.path.join(_KSO, "templates.npy")
SPIKE_CLUSTERS = os.path.join(_KSO, "spike_clusters.npy")
OUT_CSV = out_path(f"unit_trodes_channel_lookup_{SESSION_ID}.csv")


def get_conf_text():
    if os.path.exists(TRODESCONF_STANDALONE_GUESS):
        print(f"using standalone {TRODESCONF_STANDALONE_GUESS}")
        with open(TRODESCONF_STANDALONE_GUESS, encoding="utf-8", errors="ignore") as f:
            return f.read()
    print(f"no standalone trodesconf found, extracting from {REC_FILE}'s own header")
    with open(REC_FILE, "rb") as f:
        header = f.read(5_000_000)
    text = header.decode("utf-8", errors="replace")
    end = max(text.find("</SpikeConfiguration>"), text.rfind("</Configuration>"))
    if end == -1:
        return text
    return text[:text.find(">", end) + 1]


conf = get_conf_text()
ntrode_blocks = re.findall(
    r'<SpikeNTrode[^>]*id="(\d+)"[^>]*>\s*<SpikeChannel[^>]*coord_ap="(-?\d+)"[^>]*coord_dv="(-?\d+)"[^>]*coord_ml="(-?\d+)"[^>]*hwChan="(\d+)"',
    conf)
print(f"Parsed {len(ntrode_blocks)} SpikeNTrode/SpikeChannel entries")

coord_to_trodes = {}
for trodes_id, ap, dv, ml, hwchan in ntrode_blocks:
    coord_to_trodes[(int(ml), int(dv))] = dict(trodes_id=int(trodes_id), hwChan=int(hwchan), coord_ap=int(ap))

with open(CHANMAP_JSON) as f:
    cm = json.load(f)
xc, yc = np.array(cm["xc"]), np.array(cm["yc"])

chan_to_trodes = {}
missing = []
for ch in range(len(xc)):
    key = (int(xc[ch]), int(yc[ch]))
    if key in coord_to_trodes:
        chan_to_trodes[ch] = coord_to_trodes[key]
    else:
        missing.append(ch)
print(f"Matched {len(chan_to_trodes)}/{len(xc)} channels to a Trodes SpikeNTrode id ({len(missing)} unmatched)")

templates = np.load(TEMPLATES)
spike_clusters = np.load(SPIKE_CLUSTERS).ravel()
uniq, counts = np.unique(spike_clusters, return_counts=True)
count_map = dict(zip(uniq.tolist(), counts.tolist()))

rows = []
for u in range(templates.shape[0]):
    ptp = templates[u].max(axis=0) - templates[u].min(axis=0)
    peak_chan = int(np.argmax(ptp))
    info = chan_to_trodes.get(peak_chan)
    rows.append(dict(
        unit_id=u, peak_channel_index=peak_chan,
        trodes_ntrode_id=info["trodes_id"] if info else None,
        trodes_hwChan=info["hwChan"] if info else None,
        peak_amplitude=float(ptp.max()),
        n_spikes=count_map.get(u, 0),
    ))

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"Wrote {len(df)} units to {OUT_CSV}")

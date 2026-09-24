"""
Builds a Phy-style cluster_info.tsv for a session's raw kilosort4 folder when
Phy was never launched on it (so no cluster_info.tsv exists yet) -- the
sync_pipeline's units_spikes.build_units_table() requires one (columns:
cluster_id, ch, depth, group, KSLabel) and refuses to run without it.

Since there's no Phy/manual curation for this session yet, `group` here is
built directly from our own corrected_classification_<session>_with_waveforms.json
(bombcell + violation-ratio-corrected labels), mapped down to Phy's
good/mua/noise vocabulary (the +NON-SOMA suffix is dropped for this file only
-- it doesn't affect anything sync_pipeline does with `group`, which is just
used to exclude noise and to fill quality_label). If real Phy curation
happens later for this session, rebuild cluster_info.tsv from Phy's own
output instead (see WORKFLOW.md stage 2) -- this script is a stand-in for
"no curation pass has happened yet", not a permanent substitute for it.

`depth` = the y-coordinate (µm) of the unit's peak_channel from this
session's own channel_positions.npy (384x2 array, columns are [x, y] in µm) --
exactly what Phy itself would have used.

Usage: python build_cluster_info_from_classification_session.py <session_id> <rec_root>
"""
import os, sys, json
import numpy as np
import pandas as pd

SESSION_ID = sys.argv[1]
REC_ROOT = sys.argv[2]

KS4_DIR = rf"{REC_ROOT}\{SESSION_ID}.kilosort\kilosort4"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
CLASSIFICATION_JSON = rf"{OUT}\corrected_classification_{SESSION_ID}_with_waveforms.json"
OUT_PATH = rf"{KS4_DIR}\cluster_info.tsv"


def map_group(label: str) -> str:
    base = label.replace("+NON-SOMA", "")
    return {"GOOD": "good", "MUA": "mua", "NOISE": "noise"}[base]


def main():
    with open(CLASSIFICATION_JSON) as f:
        records = json.load(f)
    print(f"loaded {len(records)} unit records from {CLASSIFICATION_JSON}")

    chan_pos = np.load(os.path.join(KS4_DIR, "channel_positions.npy"))  # (n_chan, 2) -> [x, y] um

    rows = []
    for rec in records:
        uid = rec["unit_id"]
        ch = int(rec["peak_channel"])
        group = map_group(rec["label"])
        rows.append(dict(
            cluster_id=uid,
            ch=ch,
            depth=float(chan_pos[ch, 1]),
            group=group,
            KSLabel="mua" if group != "good" else "good",  # cluster_KSLabel.tsv has no non-mua/good/noise distinction anyway
        ))

    info = pd.DataFrame(rows).sort_values("cluster_id")
    info.to_csv(OUT_PATH, sep="\t", index=False)
    print(f"wrote {len(info)} rows to {OUT_PATH}")
    print(info["group"].value_counts())


if __name__ == "__main__":
    main()

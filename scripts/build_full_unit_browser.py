import json
import numpy as np
import pandas as pd

with open(r"D:\Gil\spike_sorting_agent\outputs\corrected_classification_with_waveforms.json") as f:
    class_data = {r["unit_id"]: r for r in json.load(f)}

celltypes = pd.read_csv(r"D:\Gil\spike_sorting_agent\outputs\striatal_celltypes.csv").set_index("unit_id")
trodes_lookup = pd.read_csv(r"D:\Gil\spike_sorting_agent\outputs\unit_trodes_channel_lookup.csv").set_index("unit_id")

rows = []
for uid, r in class_data.items():
    ct = celltypes.loc[uid] if uid in celltypes.index else None
    tr = trodes_lookup.loc[uid] if uid in trodes_lookup.index else None
    row = dict(r)
    row["cell_type"] = ct["cell_type"] if ct is not None else "Unknown"
    row["postSpikeSuppression_ms"] = float(ct["postSpikeSuppression_ms"]) if ct is not None else None
    row["propLongISI"] = float(ct["propLongISI"]) if ct is not None else None
    row["trodes_ntrode_id"] = int(tr["trodes_ntrode_id"]) if tr is not None and not pd.isna(tr["trodes_ntrode_id"]) else None
    rows.append(row)

rows.sort(key=lambda r: r["unit_id"])

with open(r"D:\Gil\spike_sorting_agent\outputs\full_unit_browser_data.json", "w") as f:
    json.dump(rows, f)

print(f"Merged {len(rows)} units")
import os
size = os.path.getsize(r"D:\Gil\spike_sorting_agent\outputs\full_unit_browser_data.json")
print(f"JSON size: {size/1e6:.2f} MB")

# quick sanity: label distribution, cell type distribution
print(pd.Series([r["label"] for r in rows]).value_counts())
print(pd.Series([r["cell_type"] for r in rows]).value_counts())

import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import bombcell
import numpy as np
import pandas as pd

param = bombcell.get_default_parameters(
    kilosort_path=r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\kilosort_original",
    raw_file=r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\20260901_085606.probe1.dat",
    kilosort_version=4, gain_to_uV=0.018311105685598315,
)

props_df = pd.read_parquet(r"D:\Gil\spike_sorting_agent\outputs\bombcell_results_gaincorrected\templates._bc_ephysProperties.parquet")
props_df = props_df.sort_values("unit_id").reset_index(drop=True)
ephys_properties = props_df.to_dict("records")

print("Classifying striatal cell types (MSN/FSI/TAN/UIN)...")
cell_types = bombcell.classification.classify_striatum_cells(ephys_properties, param)

labels, counts = np.unique(cell_types, return_counts=True)
print("\n=== CELL TYPE SUMMARY ===")
for l, c in zip(labels, counts):
    print(f"  {l}: {c}")

props_df["cell_type"] = cell_types
out_csv = r"D:\Gil\spike_sorting_agent\outputs\striatal_celltypes.csv"
props_df.to_csv(out_csv, index=False)
print(f"\nWrote {out_csv}")

import json
import os
import re
import sys
import numpy as np

sys.path.append(r"C:\Users\Adam\Documents\Trodes_2-4-2_Windows64\Resources\TrodesToPython")
import readTrodesExtractedDataFile3 as trodes

# Target current working directory
kilosort_dir = os.getcwd()

# Extract session timestamp pattern (e.g., 20260805_103305) from path
match = re.search(r"\d{8}_\d{6}", kilosort_dir)
session_prefix = match.group(0) if match else "session"

# Find channelmap .dat file
dat_files = [f for f in os.listdir(kilosort_dir) if f.endswith('.dat') and 'channelmap' in f]

if not dat_files:
    print(f"\n[ERROR] No channelmap .dat file found in {kilosort_dir}\n")
else:
    channelmap_path = os.path.join(kilosort_dir, dat_files[0])
    
    # Save file with session info in the name
    output_filename = f"{session_prefix}_custom_neuropixels_map.json"
    output_path = os.path.join(kilosort_dir, output_filename)
    
    cmap = trodes.readTrodesExtractedDataFile(channelmap_path)
    raw_data = cmap["data"]

    xc = raw_data["ml"].flatten().astype(float)
    yc = raw_data["dv"].flatten().astype(float)
    n_chan = len(xc)

    probe_dict = {
        "chanMap": list(range(n_chan)),
        "xc": xc.tolist(),
        "yc": yc.tolist(),
        "kcoords": [0] * n_chan,  # Force all channels onto a single shank group
        "n_chan": n_chan,
    }

    with open(output_path, "w") as f:
        json.dump(probe_dict, f, indent=4)

    print(f"\n[SUCCESS] Extracted {n_chan} channels.")
    print(f"[SUCCESS] Saved to: {output_filename}\n")
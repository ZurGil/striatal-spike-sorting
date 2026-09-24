"""
Run KiloSort4 on the 2026-09-10 recording, replicating the exact non-default
settings used for the 2026-09-01 session (Th_universal=9.2, Th_learned=7.0 --
everything else confirmed to match KS4 4.0.24's own DEFAULT_SETTINGS by
comparing against the prior run's saved ops.npy). Uses the freshly generated
probe map (make_probe_map.py output) for this session.
"""
from kilosort import run_kilosort
from kilosort.io import load_probe

SESSION_DIR = r"F:\Gil\Shamir\20260910_095200.rec\20260910_095200.kilosort"
PROBE_JSON = SESSION_DIR + r"\20260910_095200_custom_neuropixels_map.json"
DATA_FILE = SESSION_DIR + r"\20260910_095200.probe1.dat"

settings = {
    "n_chan_bin": 384,
    "fs": 30000.0,
    "Th_universal": 9.2,
    "Th_learned": 7.0,
}

probe = load_probe(PROBE_JSON)

run_kilosort(
    settings=settings,
    probe=probe,
    filename=DATA_FILE,
    data_dtype="int16",
    do_CAR=True,
    invert_sign=False,
    save_preprocessed_copy=False,
    clear_cache=False,
)

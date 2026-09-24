"""
Session auto-discovery: given a session root path
    D:\\...\\<session_id>.rec\\
locates every input the pipeline needs by the naming convention documented in
claude_code_kickoff_prompt.md, and defines the two non-destructive output folders
(spec Section 9) inside the session's .kilosort\\ directory.
"""
import os
from dataclasses import dataclass


@dataclass
class SessionPaths:
    session_id: str
    session_root: str
    kilosort_input_dir: str        # <session_id>.kilosort\
    probe_dat: str                 # <session_id>.probe1.dat
    channelmap_dat: str            # <session_id>.channelmap_probe1.dat (Trodes-native; verified redundant with json in this session)
    channel_map_json: str          # <session_id>_custom_neuropixels_map.json
    timestamps_dat: str            # <session_id>.timestamps.dat
    ks4_output_dir: str            # kilosort4\ (KS4's own output, possibly phy-curated already)
    kilosort_original_dir: str     # kilosort4\kilosort_original\  (agent-written, non-destructive baseline)
    agent_processed_dir: str       # kilosort4\agent_processed\    (agent-written, all decisions)

    def ensure_output_dirs(self):
        os.makedirs(self.kilosort_original_dir, exist_ok=True)
        os.makedirs(self.agent_processed_dir, exist_ok=True)


def discover_session(session_root: str) -> SessionPaths:
    session_root = os.path.normpath(session_root)
    base = os.path.basename(session_root)
    if not base.endswith(".rec"):
        raise ValueError(f"Expected a '<session_id>.rec' folder, got: {base}")
    session_id = base[: -len(".rec")]

    kilosort_input_dir = os.path.join(session_root, f"{session_id}.kilosort")
    if not os.path.isdir(kilosort_input_dir):
        raise FileNotFoundError(f"Missing expected folder: {kilosort_input_dir}")

    probe_dat = os.path.join(kilosort_input_dir, f"{session_id}.probe1.dat")
    channelmap_dat = os.path.join(kilosort_input_dir, f"{session_id}.channelmap_probe1.dat")
    channel_map_json = os.path.join(kilosort_input_dir, f"{session_id}_custom_neuropixels_map.json")
    timestamps_dat = os.path.join(kilosort_input_dir, f"{session_id}.timestamps.dat")
    ks4_output_dir = os.path.join(kilosort_input_dir, "kilosort4")

    for p, label in [(probe_dat, "probe1.dat"), (channel_map_json, "channel map json"),
                      (timestamps_dat, "timestamps.dat"), (ks4_output_dir, "kilosort4 output dir")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing expected {label}: {p}")

    return SessionPaths(
        session_id=session_id,
        session_root=session_root,
        kilosort_input_dir=kilosort_input_dir,
        probe_dat=probe_dat,
        channelmap_dat=channelmap_dat,
        channel_map_json=channel_map_json,
        timestamps_dat=timestamps_dat,
        ks4_output_dir=ks4_output_dir,
        kilosort_original_dir=os.path.join(ks4_output_dir, "kilosort_original"),
        agent_processed_dir=os.path.join(ks4_output_dir, "agent_processed"),
    )

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
                      (ks4_output_dir, "kilosort4 output dir")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing expected {label}: {p}")

    # timestamps.dat is required for an ordinary session and legitimately ABSENT for a
    # concatenated one. Nothing in striatal_agent reads it -- it is carried on this
    # dataclass and validated here so a session staged without it is caught early, since
    # scripts/build_session_timeline_data.py does need it and runs much later.
    #
    # A joint session (two recordings of one day joined into a single raw file, see
    # WORKFLOW.md STAGE 0) has one timestamps.dat PER PART and none for the whole,
    # because the Trodes sample counter restarts when Trodes restarts -- and the two
    # parts' counters OVERLAP, so a concatenated timestamps.dat would be non-monotonic at
    # the seam and every elapsed time computed from it downstream would be wrong. Writing
    # one to satisfy this check would be a booby trap, so instead the check is waived when
    # the concatenation provenance file is sitting beside the sort, which is proof this is
    # a joint session rather than a mis-staged ordinary one. timestamps_dat is then None,
    # so anything that tries to use it fails immediately and visibly instead of reading a
    # fabricated file. The concat timeline builder reads each part's own file.
    if not os.path.exists(timestamps_dat):
        import glob
        concat_prov = [f for f in glob.glob(os.path.join(kilosort_input_dir, "concat_*.json"))]
        if not concat_prov:
            raise FileNotFoundError(
                f"Missing expected timestamps.dat: {timestamps_dat}\n"
                "If this is a CONCATENATED session it has none by design -- put its "
                "concat_*.json provenance file next to the sort in "
                f"{kilosort_input_dir} and this check is waived."
            )
        timestamps_dat = None

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

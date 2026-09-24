"""
Section 9 -- non-destructive output architecture.

kilosort_original/  -- the TRUE pre-phy-curation KS4 output, reconstructed per this
                        session's finding: spike_templates.npy was never touched by phy
                        (only spike_clusters.npy + cluster_*.tsv were), so the original
                        clustering = spike_templates.npy verbatim. The stale
                        cluster_KSLabel/ContamPct/Amplitude/info.tsv files are NOT carried
                        forward (verified in-session to be a mid-curation snapshot, not
                        the true original -- see conversation). Read-only, permanent
                        baseline to diff against forever.

agent_processed/     -- starts as a copy of kilosort_original, then the pipeline layers
                        its own edits on top: recovered spikes as separate child clusters
                        (Section 9.2, unit 15 -> 1015 convention), and cluster_<label>.tsv
                        score columns (Section 9.1) that phy auto-detects and renders
                        natively, no plugin needed.

Large, never-edited arrays (templates, pc_features, whitening matrices, channel
geometry) are hardlinked rather than copied between the two folders -- same content,
avoids ~11GB of duplication per run; falls back to a real copy if hardlinking fails
(e.g. cross-volume).
"""
import os
import shutil
import numpy as np
import pandas as pd

from .session import SessionPaths

CHILD_CLUSTER_OFFSET = 1000

IMMUTABLE_FILES = [
    "spike_templates.npy", "templates.npy", "templates_ind.npy",
    "pc_features.npy", "pc_feature_ind.npy",
    "whitening_mat.npy", "whitening_mat_inv.npy", "whitening_mat_dat.npy",
    "channel_map.npy", "channel_positions.npy", "channel_shanks.npy",
    "similar_templates.npy", "amplitudes.npy", "spike_positions.npy",
]
SMALL_COPY_FILES = ["params.py", "ops.npy"]


def _link_or_copy(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_kilosort_original(paths: SessionPaths, overwrite=False):
    paths.ensure_output_dirs()
    out_dir = paths.kilosort_original_dir
    src_dir = paths.ks4_output_dir

    spike_templates = np.load(os.path.join(src_dir, "spike_templates.npy")).ravel()
    spike_times = np.load(os.path.join(src_dir, "spike_times.npy")).ravel()

    dst_spike_times = os.path.join(out_dir, "spike_times.npy")
    dst_spike_clusters = os.path.join(out_dir, "spike_clusters.npy")
    if overwrite or not os.path.exists(dst_spike_clusters):
        np.save(dst_spike_times, spike_times)
        np.save(dst_spike_clusters, spike_templates.astype(np.int32))  # original clustering == spike_templates

    for fname in IMMUTABLE_FILES:
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            _link_or_copy(src, os.path.join(out_dir, fname))
    for fname in SMALL_COPY_FILES:
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fname))

    return dict(spike_times=spike_times, spike_clusters=spike_templates,
                templates=np.load(os.path.join(out_dir, "templates.npy")))


def init_agent_processed(paths: SessionPaths, overwrite=False):
    paths.ensure_output_dirs()
    out_dir = paths.agent_processed_dir
    src_dir = paths.kilosort_original_dir

    dst_spike_clusters = os.path.join(out_dir, "spike_clusters.npy")
    if overwrite or not os.path.exists(dst_spike_clusters):
        shutil.copy2(os.path.join(src_dir, "spike_times.npy"), os.path.join(out_dir, "spike_times.npy"))
        shutil.copy2(os.path.join(src_dir, "spike_clusters.npy"), dst_spike_clusters)

    for fname in IMMUTABLE_FILES + SMALL_COPY_FILES:
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            _link_or_copy(src, os.path.join(out_dir, fname))


def write_cluster_tsv(agent_processed_dir, label, id_value_map):
    """Phy auto-detects any cluster_<label>.tsv (columns cluster_id, <label>) as a
    sortable/filterable column -- Section 9.1, no plugin needed."""
    df = pd.DataFrame(sorted(id_value_map.items()), columns=["cluster_id", label])
    df.to_csv(os.path.join(agent_processed_dir, f"cluster_{label}.tsv"), sep="\t", index=False)


def append_recovered_spikes(agent_processed_dir, recoveries: dict):
    """recoveries: {unit_id: recovered_spike_times_array}. Writes recovered spikes as
    separate child clusters (unit 15 -> 1015), never merged silently into the parent
    cluster (Section 9.2) -- reversible-by-construction in phy's merge view."""
    spike_times = np.load(os.path.join(agent_processed_dir, "spike_times.npy")).ravel()
    spike_clusters = np.load(os.path.join(agent_processed_dir, "spike_clusters.npy")).ravel()

    new_times, new_clusters = [], []
    child_id_map = {}
    for unit_id, times in recoveries.items():
        if len(times) == 0:
            continue
        child_id = int(unit_id) + CHILD_CLUSTER_OFFSET
        child_id_map[int(unit_id)] = child_id
        new_times.append(np.asarray(times, dtype=spike_times.dtype))
        new_clusters.append(np.full(len(times), child_id, dtype=spike_clusters.dtype))

    if new_times:
        spike_times = np.concatenate([spike_times] + new_times)
        spike_clusters = np.concatenate([spike_clusters] + new_clusters)
        order = np.argsort(spike_times)
        spike_times, spike_clusters = spike_times[order], spike_clusters[order]

    np.save(os.path.join(agent_processed_dir, "spike_times.npy"), spike_times)
    np.save(os.path.join(agent_processed_dir, "spike_clusters.npy"), spike_clusters)
    return child_id_map


def write_cluster_group(agent_processed_dir, group_map: dict):
    df = pd.DataFrame(sorted(group_map.items()), columns=["cluster_id", "group"])
    df.to_csv(os.path.join(agent_processed_dir, "cluster_group.tsv"), sep="\t", index=False)

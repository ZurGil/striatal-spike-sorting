"""
Run striatal_agent's run_pipeline() for ONE batch of units, as a genuinely
fresh process (required -- see WORKFLOW.md 1e: the pipeline reliably crashes
around unit ~20-24 regardless of requested count or free system RAM, so each
batch must be its own process invocation, not a loop within one process).

RECONSTRUCTED 2026-09-17 -- the original batch-orchestration scripts
(plan_batches_v2.py / run_batch.py / run_all_batches.sh / consolidate_batches.py,
used successfully for 20260911_100049's 47 batches) were written during an
earlier conversation and never saved to scripts/ -- lost when that
conversation's scratchpad was cleaned up. This is a fresh rebuild of the same
approach, now actually saved.

Important: `run_pipeline()` writes `agent_processed/agent_report.json` and
`agent_processed/cluster_<label>.tsv` at FIXED filenames every time it's
called -- each batch's call OVERWRITES these with just that batch's own
units, so this script immediately archives its own batch's output to
`agent_processed/batch_reports/` before the next batch process can stomp on
it. `consolidate_batches_session.py` merges all the archived batches back
into one final combined state afterward.

`build_kilosort_original()`/`init_agent_processed()` (called internally by
run_pipeline) are idempotent (overwrite=False checks if already built), so
it's safe to call this once per batch without redoing that setup work each
time.

Usage: python run_pipeline_batch_session.py <session_id> <rec_root> <batch_num> <start_idx> <end_idx>
  <start_idx>/<end_idx>: Python-style half-open slice indices into the
  sorted list of all unit ids (0-based, end exclusive).
"""
import os, sys, json, shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from striatal_agent.pipeline import run_pipeline
from striatal_agent.config import AgentConfig
from striatal_agent.session import discover_session

SESSION_ID = sys.argv[1]
REC_ROOT = sys.argv[2]
BATCH_NUM = int(sys.argv[3])
START_IDX = int(sys.argv[4])
END_IDX = int(sys.argv[5])


def main():
    paths = discover_session(REC_ROOT)
    spike_templates = np.load(os.path.join(paths.ks4_output_dir, "spike_templates.npy")).ravel()
    all_unit_ids = sorted(np.unique(spike_templates).tolist())
    batch_units = all_unit_ids[START_IDX:END_IDX]
    print(f"batch {BATCH_NUM}: units[{START_IDX}:{END_IDX}] = {batch_units[0]}..{batch_units[-1]} ({len(batch_units)} units)")

    cfg = AgentConfig(skip_cross_unit_acg=True)
    report = run_pipeline(REC_ROOT, units=batch_units, cfg=cfg, verbose=True)

    archive_dir = os.path.join(paths.agent_processed_dir, "batch_reports")
    os.makedirs(archive_dir, exist_ok=True)
    with open(os.path.join(archive_dir, f"batch_{BATCH_NUM:03d}_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    for fname in os.listdir(paths.agent_processed_dir):
        if fname.startswith("cluster_") and fname.endswith(".tsv"):
            shutil.copy2(os.path.join(paths.agent_processed_dir, fname),
                         os.path.join(archive_dir, f"batch_{BATCH_NUM:03d}_{fname}"))

    print(f"batch {BATCH_NUM} done, archived to {archive_dir}")


if __name__ == "__main__":
    main()

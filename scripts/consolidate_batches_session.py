"""
Merge all archived per-batch outputs (from run_pipeline_batch_session.py's
`agent_processed/batch_reports/`) into one final, consistent
`agent_processed/agent_report.json` and `cluster_<label>.tsv` set, covering
every unit across every batch.

RECONSTRUCTED 2026-09-17, replaces the lost consolidate_batches.py from the
earlier (unsaved) run for 20260911_100049 -- see run_pipeline_batch_session.py's
docstring for why archiving-then-consolidating is necessary at all (each
batch's run_pipeline() call overwrites agent_report.json/cluster_*.tsv with
just its own units).

Note: `merge_records` (cross-batch merge-candidate scoring) is expected to be
INCOMPLETE across batch boundaries -- plain sequential batching, not spatial-
overlap batching (that approach was tried and abandoned for 20260911_100049,
see WORKFLOW.md 1e -- ballooned batch sizes too much). This is accepted as
advisory-only and does not affect any unit's own score/label.

Usage: python consolidate_batches_session.py <session_id> <rec_root>
"""
import os, sys, json, glob
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from striatal_agent.session import discover_session
from striatal_agent.audit import session_summary

SESSION_ID = sys.argv[1]
REC_ROOT = sys.argv[2]


def main():
    paths = discover_session(REC_ROOT)
    archive_dir = os.path.join(paths.agent_processed_dir, "batch_reports")

    report_files = sorted(glob.glob(os.path.join(archive_dir, "batch_*_report.json")))
    print(f"found {len(report_files)} batch reports")

    all_per_unit = []
    all_merge_records = []
    all_child_id_map = {}
    cfg_dict = None
    for rf in report_files:
        with open(rf) as f:
            r = json.load(f)
        all_per_unit.extend(r["per_unit_records"])
        all_merge_records.extend(r.get("merge_records", []))
        all_child_id_map.update(r.get("child_cluster_id_map", {}))
        cfg_dict = r.get("config", cfg_dict)

    seen = set()
    dedup_records = []
    for rec in all_per_unit:
        uid = rec["unit_id"]
        if uid in seen:
            print(f"WARNING: duplicate unit_id {uid} across batches, keeping first occurrence")
            continue
        seen.add(uid)
        dedup_records.append(rec)
    print(f"{len(dedup_records)} unique unit records ({len(all_per_unit) - len(dedup_records)} duplicates dropped)")

    summary = session_summary(dedup_records)
    final_report = dict(session_id=SESSION_ID, per_unit_records=dedup_records,
                         merge_records=all_merge_records, summary=summary,
                         child_cluster_id_map=all_child_id_map, config=cfg_dict)
    final_report_path = os.path.join(paths.agent_processed_dir, "agent_report.json")
    with open(final_report_path, "w") as f:
        json.dump(final_report, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"wrote consolidated report: {final_report_path}")
    print(f"summary: {summary}")

    label_files = {}
    for tf in glob.glob(os.path.join(archive_dir, "batch_*_cluster_*.tsv")):
        bn = os.path.basename(tf)
        label = bn.split("_cluster_", 1)[1][:-4]  # strip "batch_NNN_cluster_" prefix and ".tsv" suffix
        label_files.setdefault(label, []).append(tf)

    for label, files in label_files.items():
        dfs = [pd.read_csv(f, sep="\t") for f in files]
        combined = pd.concat(dfs, ignore_index=True).drop_duplicates("cluster_id", keep="first")
        combined = combined.sort_values("cluster_id")
        out_path = os.path.join(paths.agent_processed_dir, f"cluster_{label}.tsv")
        combined.to_csv(out_path, sep="\t", index=False)
        print(f"wrote {out_path}: {len(combined)} rows")


if __name__ == "__main__":
    main()

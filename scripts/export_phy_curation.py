"""
Stage C: export Stage B's verdicts (manual, from the review tool's shared db) plus
the automatic classification (as a placeholder for not-yet-reviewed units) as a
phy-compatible cluster_group.tsv.

Written directly into kilosort_original/, alongside the existing params.py --
purely additive (no file there is modified or removed), so the "kilosort_original/
is a forever baseline" invariant holds: this only adds a new curation file, the
same kind of file phy itself would create on first manual save inside its own GUI.

Priority: a manual verdict (from outputs/verdicts/*.json, pulled from the review
tool's db) always overrides the automatic label. Units with neither a manual verdict
nor a resolvable automatic label are left out of the file entirely, which phy
displays as "unsorted".
"""
import os, sys, json, glob

OUT = r"D:\Gil\spike_sorting_agent\outputs"
# Exporting to agent_processed/, not kilosort_original/: phy auto-loads every cluster_*.tsv
# in whatever directory params.py lives in, and kilosort_original/ already carries ~26
# bombcell metric TSVs -- fine to have on disk, but not useful clutter in phy's cluster
# table. agent_processed/ already has its own full copy of the KS4 data arrays (from an
# earlier pipeline stage), so no file copying is needed; its old cluster_*.tsv files (a
# now-superseded automated tiering pass, pre-dating this session's corrected_classification.py)
# were moved into agent_processed/_previous_agent_metrics/ rather than deleted.
KO_DIR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\agent_processed"
VERDICTS_DIR = r"C:\Users\Adam\AppData\Local\Temp\claude\D--Gil-spike-sorting-agent\8186c809-76fa-41ff-bbca-59d44125e5fc\scratchpad\verdicts\verdicts"
OUT_TSV = os.path.join(KO_DIR, "cluster_group.tsv")
OUT_STATUS_TSV = os.path.join(KO_DIR, "cluster_review_status.tsv")
OUT_VIOLATION_TSV = os.path.join(KO_DIR, "cluster_violation_ratio.tsv")

AUTO_LABEL_TO_GROUP = {"GOOD": "good", "MUA": "mua", "NOISE": "noise"}


def main():
    with open(os.path.join(OUT, "pipeline_review_data.json")) as f:
        units = {u["unit_id"]: u for u in json.load(f)}

    manual_verdicts = {}
    for path in glob.glob(os.path.join(VERDICTS_DIR, "*.json")):
        uid = int(os.path.splitext(os.path.basename(path))[0])
        with open(path) as f:
            manual_verdicts[uid] = json.load(f)["verdict"]

    rows = []
    n_manual, n_auto, n_unsorted = 0, 0, 0
    for uid in sorted(units.keys()):
        if uid in manual_verdicts:
            group = manual_verdicts[uid]
            status = "manual"
            n_manual += 1
        else:
            core_label = units[uid]["label"].replace("+NON-SOMA", "")
            group = AUTO_LABEL_TO_GROUP.get(core_label)
            if group is None:
                n_unsorted += 1
                continue
            status = "auto"
            n_auto += 1
        rows.append((uid, group, status))

    with open(OUT_TSV, "w", newline="") as f:
        f.write("cluster_id\tgroup\n")
        for uid, group, _ in rows:
            f.write(f"{uid}\t{group}\n")

    with open(OUT_STATUS_TSV, "w", newline="") as f:
        f.write("cluster_id\treview_status\n")
        for uid, _, status in rows:
            f.write(f"{uid}\t{status}\n")

    with open(OUT_VIOLATION_TSV, "w", newline="") as f:
        f.write("cluster_id\tviolation_ratio\n")
        for uid in sorted(units.keys()):
            vr = units[uid].get("violation_ratio")
            if vr is not None:
                f.write(f"{uid}\t{vr}\n")

    print(f"wrote {OUT_TSV}")
    print(f"wrote {OUT_STATUS_TSV}")
    print(f"wrote {OUT_VIOLATION_TSV}")
    print(f"{len(rows)}/{len(units)} units labeled "
          f"({n_manual} manual verdicts, {n_auto} auto-classification placeholders, "
          f"{n_unsorted} left as unsorted -- no resolvable label)")
    from collections import Counter
    print("group breakdown:", Counter(g for _, g, _ in rows))
    print("review_status breakdown:", Counter(s for _, _, s in rows))


if __name__ == "__main__":
    main()

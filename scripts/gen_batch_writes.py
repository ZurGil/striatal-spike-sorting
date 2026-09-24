import json, os

OUT_DIR = r"D:\Gil\spike_sorting_agent\outputs\raw_traces_20260911_100049_split"
uids = sorted(int(f.replace('.json', '')) for f in os.listdir(OUT_DIR))
BATCH = 8
batches = [uids[i:i + BATCH] for i in range(0, len(uids), BATCH)]
lines = []
for bi, b in enumerate(batches):
    writes = [
        {"op": "set", "collection": "raw_traces", "doc_id": str(u),
         "file_path": os.path.join(OUT_DIR, f"{u}.json")}
        for u in b
    ]
    lines.append(f"{bi}\t{json.dumps(writes)}")

with open(r"D:\Gil\spike_sorting_agent\outputs\batch_writes.txt", "w") as f:
    f.write("\n".join(lines))
print(len(batches), "batches written")

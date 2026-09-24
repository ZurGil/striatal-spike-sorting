"""
Precompute real raw-data spike traces for all 401 units, for the interactive
review tool's optional "raw spikes" panel (toggle-able, per user request:
"view about 4-10 traces spikes for a unit... to verify if specific unit has
spike shape in the raw data"). 8 real spikes per unit, spread evenly across
that unit's full spike train, each as a +/-15ms raw filtered trace (gain-
corrected to uV) plus its exact 2ms waveform snippet -- same format already
validated and liked in the one-off pages built earlier this session.

Output is written locally first (not directly to the artifact db, which
needs the Artifact tool from the main conversation), one JSON file per unit
batch, to be pushed via write_db afterward.
"""
import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from striatal_agent.config import AgentConfig
from striatal_agent.session import discover_session
from striatal_agent.raw_io import RawReader, highpass_filter

SESSION_ROOT = r"D:\Gil\Shamir\20260901_085606.rec"
ko = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\kilosort_original"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
OUT_PATH = os.path.join(OUT, "raw_trace_snippets_401.json")
PROGRESS_PATH = os.path.join(OUT, "raw_trace_snippets_401_progress.txt")

FS = 30000.0
GAIN_TO_UV = 0.018311105685598315
NT, NT0MIN = 60, 20
PAD_MS = 15
N_SPIKES = 8


def extract_unit(reader, cfg, chan, st):
    if len(st) < N_SPIKES:
        picks = st
    else:
        picks_idx = np.linspace(0, len(st) - 1, N_SPIKES).astype(int)
        picks = st[picks_idx]
    pad = int(PAD_MS / 1000 * FS)
    margin = 100
    traces = []
    for t0 in picks:
        t0 = int(t0)
        block, s2, e2 = reader.read_window(t0 - pad - margin, t0 + pad + margin, chans=[chan])
        filt = highpass_filter(block, FS, cfg.highpass_hz, axis=0)[:, 0] * GAIN_TO_UV
        idx_t0 = t0 - s2

        trace_full = filt[idx_t0 - pad: idx_t0 + pad]
        trace = [round(float(x), 1) for x in trace_full] if len(trace_full) == 2 * pad else None

        wf = filt[idx_t0 - NT0MIN: idx_t0 - NT0MIN + NT]
        waveform = [round(float(x), 1) for x in wf] if len(wf) == NT else None

        if trace is not None and waveform is not None:
            traces.append(dict(t_sec=round(t0 / FS, 3), trace=trace, waveform=waveform))
    return traces


def main():
    t0 = time.time()
    cfg = AgentConfig()
    paths = discover_session(SESSION_ROOT)
    spike_times_all = np.load(os.path.join(ko, "spike_times.npy")).ravel()
    spike_clusters_all = np.load(os.path.join(ko, "spike_clusters.npy")).ravel()
    reader = RawReader(paths.probe_dat, cfg.n_chan)

    with open(os.path.join(OUT, "pipeline_review_data.json")) as f:
        units = json.load(f)

    results = {}
    for i, u in enumerate(units):
        uid = u["unit_id"]
        chan = u["peak_channel"]
        st = np.sort(spike_times_all[spike_clusters_all == uid])
        if len(st) == 0:
            results[uid] = None
            continue
        try:
            traces = extract_unit(reader, cfg, chan, st)
        except Exception as e:
            traces = None
            print(f"unit {uid} failed: {e}")
        results[uid] = dict(chan=int(chan), pad_ms=PAD_MS, gain_applied=True, traces=traces)

        if (i + 1) % 20 == 0 or (i + 1) == len(units):
            elapsed = time.time() - t0
            with open(PROGRESS_PATH, "w") as f:
                f.write(f"{i+1}/{len(units)} units done, {elapsed:.0f}s elapsed, "
                        f"~{elapsed/(i+1)*(len(units)-(i+1)):.0f}s remaining\n")
            with open(OUT_PATH, "w") as f:
                json.dump(results, f)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f)
    with open(PROGRESS_PATH, "w") as f:
        f.write(f"DONE: {len(units)}/{len(units)} units, {time.time()-t0:.0f}s total\n")
    print(f"DONE: {len(units)} units in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

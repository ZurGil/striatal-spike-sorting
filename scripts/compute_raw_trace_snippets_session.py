"""
Session-parameterized version of compute_raw_trace_snippets.py (original hardcoded to
20260901_085606 -- leave that one alone). Precomputes N_SPIKES real raw-data spike
traces per unit for the review tool's "raw spikes" panel. Output is written locally
first, then must be pushed into the published artifact's `raw_traces` database
collection (one doc per unit_id, via the Artifact tool's write_db action -- see
WORKFLOW.md 1f-pre) since a script has no way to reach the artifact db directly.

2026-09-16 update (per user request): each trace window now also records every
OTHER spike detected within it -- not just the one it was centered on -- so the
review tool can mark them with vertical lines instead of only the centered event.
"Detected within it" means: any spike (from this unit or any other unit sharing
this unit's peak channel, i.e. the same set already surfaced as `co_located_units`)
whose recorded spike time falls inside [t0-pad, t0+pad). Units on OTHER channels are
deliberately excluded -- a coincidental spike on a physically distant channel isn't
something this trace's voltage could actually show, so marking it would be
misleading rather than useful for the "does this deflection correspond to a real,
sorted spike" QC check the user wants. N_SPIKES was doubled from 8 to 16 (the review
tool now shows the first 8 by default with a "show more" button for the rest) --
doubling only doubles the number of small raw-data window reads, it does not
materially change runtime (was ~110s/466 units at N_SPIKES=8, expect roughly ~220s
at N_SPIKES=16 -- still cheap).

Usage: python compute_raw_trace_snippets_session.py <session_id> <rec_root> <gain_to_uV>
"""
import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from striatal_agent.config import AgentConfig
from striatal_agent.session import discover_session
from striatal_agent.raw_io import RawReader, highpass_filter

SESSION_ID = sys.argv[1]
REC_ROOT = sys.argv[2]
GAIN_TO_UV = float(sys.argv[3])

SESSION_ROOT = REC_ROOT
KO_DIR = rf"{REC_ROOT}\{SESSION_ID}.kilosort\kilosort4\kilosort_original"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
REVIEW_DATA_JSON = rf"{OUT}\pipeline_review_data_{SESSION_ID}.json"
OUT_PATH = rf"{OUT}\raw_trace_snippets_{SESSION_ID}.json"
PROGRESS_PATH = rf"{OUT}\raw_trace_snippets_{SESSION_ID}_progress.txt"

FS = 30000.0
NT, NT0MIN = 60, 20
PAD_MS = 15
N_SPIKES = 16


def spikes_in_window(t0, pad, candidate_uids, spike_times_by_unit):
    """All (unit_id, offset_ms) pairs for any candidate unit's spike time inside
    [t0-pad, t0+pad), offset measured from t0 -- same convention as the trace's own
    x-axis ("ms from spike time"), so the review tool can place a vertical line
    directly from this number with no further conversion."""
    hits = []
    for u2 in candidate_uids:
        cst = spike_times_by_unit.get(u2)
        if cst is None or len(cst) == 0:
            continue
        lo = np.searchsorted(cst, t0 - pad, side="left")
        hi = np.searchsorted(cst, t0 + pad, side="left")
        for st2 in cst[lo:hi]:
            hits.append(dict(unit_id=int(u2), offset_ms=round(float((st2 - t0) / FS * 1000.0), 2)))
    hits.sort(key=lambda h: h["offset_ms"])
    return hits


def extract_unit(reader, cfg, chan, uid, st, candidate_uids, spike_times_by_unit):
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
            spikes = spikes_in_window(t0, pad, candidate_uids, spike_times_by_unit)
            traces.append(dict(t_sec=round(t0 / FS, 3), trace=trace, waveform=waveform, spikes=spikes))
    return traces


def main():
    t0 = time.time()
    cfg = AgentConfig()
    paths = discover_session(SESSION_ROOT)
    spike_times_all = np.load(os.path.join(KO_DIR, "spike_times.npy")).ravel()
    spike_clusters_all = np.load(os.path.join(KO_DIR, "spike_clusters.npy")).ravel()
    reader = RawReader(paths.probe_dat, cfg.n_chan)

    with open(REVIEW_DATA_JSON) as f:
        units = json.load(f)

    chan_to_units = {}
    for u in units:
        chan_to_units.setdefault(u["peak_channel"], []).append(u["unit_id"])

    # Built once and reused both as each unit's own spike train and as the source
    # for neighbor lookups -- same per-unit boolean mask cost that was already being
    # paid once per unit before this change, just cached instead of only used once.
    spike_times_by_unit = {}
    for u in units:
        uid2 = u["unit_id"]
        spike_times_by_unit[uid2] = np.sort(spike_times_all[spike_clusters_all == uid2])

    results = {}
    for i, u in enumerate(units):
        uid = u["unit_id"]
        chan = u["peak_channel"]
        st = spike_times_by_unit[uid]
        if len(st) == 0:
            results[uid] = None
            continue
        candidate_uids = chan_to_units.get(chan, [uid])
        try:
            traces = extract_unit(reader, cfg, chan, uid, st, candidate_uids, spike_times_by_unit)
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

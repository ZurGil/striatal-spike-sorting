"""Measure how long Kilosort4 takes on CPU, so the remote plan rests on a number.

The RIS compute allocation is expected to be CPU-mostly, with GPU availability
uncertain. Kilosort4 runs on CPU, but the slowdown against a GPU is the single
fact that decides whether remote sorting is a throughput win or a dead end --
and it is not something to estimate from general knowledge, because it depends
on the core count, the thread count, and which KS4 stage dominates.

So: run the same short fixture the project already has a GPU result for, on CPU,
with a bounded thread count, and report seconds-of-wall-clock per
second-of-recording plus the peak memory. Those two numbers size the LSF request
(`-n` cores and `rusage[mem=]`) and predict a real session's runtime.

Runs on this Windows box and unchanged on a Linux node, which is the point --
the number that matters is the node's, not this PC's.

Usage:
    python benchmark_kilosort_cpu.py --data hybrid_small/hybrid_small.bin \
        --probe hybrid_small/probe.json --device cpu --threads 8 --out <dir>

Writes a JSON summary to <dir>/benchmark_summary.json. Read-only with respect to
the project: it only writes inside --out.
"""
import argparse
import json
import os
import threading
import time


def peak_rss_watcher(stop, out, interval=2.0):
    """Sample this process's resident memory until told to stop.

    KS4 does not report its own peak, and the LSF memory request has to come
    from somewhere other than a guess.
    """
    try:
        import psutil
    except ImportError:
        out["peak_rss_gb"] = None
        out["peak_rss_note"] = "psutil not installed -- memory not measured"
        return
    p = psutil.Process()
    peak = 0.0
    while not stop.is_set():
        try:
            # children too: KS4 spawns workers for data loading
            tot = p.memory_info().rss
            for c in p.children(recursive=True):
                try:
                    tot += c.memory_info().rss
                except Exception:
                    pass
            peak = max(peak, tot / 1e9)
        except Exception:
            pass
        stop.wait(interval)
    out["peak_rss_gb"] = round(peak, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="flat int16 binary, channels interleaved")
    ap.add_argument("--probe", required=True, help="probe JSON (make_probe_map.py output)")
    ap.add_argument("--out", required=True, help="results directory (created)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=8,
                    help="torch intra-op threads; mirror the cores requested from LSF")
    ap.add_argument("--n-chan", type=int, default=384)
    ap.add_argument("--fs", type=float, default=30000.0)
    ap.add_argument("--th-universal", type=float, default=9.0)
    ap.add_argument("--th-learned", type=float, default=8.0)
    args = ap.parse_args()

    # Thread caps must be set before torch starts, and the BLAS vars before numpy.
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(args.threads)

    import numpy as np
    import torch
    from kilosort import run_kilosort
    from kilosort.io import load_probe

    torch.set_num_threads(args.threads)

    nbytes = os.path.getsize(args.data)
    nsamp = nbytes // (args.n_chan * 2)
    rec_seconds = nsamp / args.fs

    os.makedirs(args.out, exist_ok=True)

    info = {
        "device_requested": args.device,
        "threads": args.threads,
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "data": os.path.abspath(args.data),
        "data_bytes": nbytes,
        "n_chan": args.n_chan,
        "recording_seconds": round(rec_seconds, 2),
        "cpu_count": os.cpu_count(),
    }
    try:
        import kilosort
        info["kilosort_version"] = kilosort.__version__
    except Exception:
        pass
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda asked for but torch.cuda.is_available() is False")
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["gpu_vram_gb"] = round(p.total_memory / 1e9, 1)

    print("benchmarking Kilosort4")
    for k in ("kilosort_version", "torch_version", "device_requested", "threads",
              "cpu_count", "recording_seconds", "gpu"):
        if k in info:
            print("  %-20s %s" % (k, info[k]))

    probe = load_probe(args.probe)
    settings = {
        "n_chan_bin": args.n_chan,
        "fs": args.fs,
        "Th_universal": args.th_universal,
        "Th_learned": args.th_learned,
    }

    stop = threading.Event()
    mem = {}
    t = threading.Thread(target=peak_rss_watcher, args=(stop, mem), daemon=True)
    t.start()

    t0 = time.time()
    try:
        run_kilosort(
            settings=settings,
            probe=probe,
            filename=args.data,
            data_dtype="int16",
            results_dir=args.out,
            device=torch.device(args.device),
            do_CAR=True,
            invert_sign=False,
            save_preprocessed_copy=False,
        )
    finally:
        wall = time.time() - t0
        stop.set()
        t.join(timeout=10)

    info.update(mem)
    info["wall_seconds"] = round(wall, 1)
    info["realtime_factor"] = round(wall / rec_seconds, 3)

    # What this predicts for the real thing. Session lengths in this project run
    # 3.0-4.8 h; 4.5 h is a representative long one.
    for label, hours in (("3.0 h session", 3.0), ("4.5 h session", 4.5)):
        info["predicted_" + label.split()[0].replace(".", "p") + "h_hours"] = \
            round(wall / rec_seconds * hours, 2)

    try:
        n_spikes = len(np.load(os.path.join(args.out, "spike_times.npy")))
        n_units = len(set(np.load(os.path.join(args.out, "spike_clusters.npy")).tolist()))
        info["n_spikes"] = int(n_spikes)
        info["n_units"] = int(n_units)
    except Exception as e:
        info["output_check"] = "could not read results: %s" % e

    print("\nresult")
    print("  wall clock            %.1f s for %.1f s of recording" % (wall, rec_seconds))
    print("  realtime factor       %.3f x  (wall / recording)" % info["realtime_factor"])
    if info.get("peak_rss_gb"):
        print("  peak memory           %.2f GB" % info["peak_rss_gb"])
    for label, hours in (("3.0", 3.0), ("4.5", 4.5)):
        print("  predicts a %s h session at %.2f h" % (label, wall / rec_seconds * hours))
    if "n_units" in info:
        print("  found                 %d units, %d spikes" % (info["n_units"], info["n_spikes"]))

    with open(os.path.join(args.out, "benchmark_summary.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("\nsummary -> %s" % os.path.join(args.out, "benchmark_summary.json"))


if __name__ == "__main__":
    main()

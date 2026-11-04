"""Run Kilosort4 on one session, with paths and thresholds as arguments.

The existing runners (`run_kilosort4_20260910.py`, `run_kilosort4_20260916.py`)
hardcode Windows paths and one session's thresholds each. That is fine for a
hand-run sort on this PC and useless on a compute node, where the same image has
to sort whichever session the job was handed. This is the same call with the
session-specific parts lifted into arguments, and nothing else changed:
`do_CAR=True`, `invert_sign=False`, `save_preprocessed_copy=False`, `int16`.

Two project-specific hazards are checked rather than trusted, because both
produce output that looks completely normal and is wrong:

  1. **The thresholds are per session.** 20260916 used Th_universal=9.0 /
     Th_learned=8.0; 20260910 and 20260901 used 9.2 / 7.0. They are required
     arguments here -- there is no default, because a default is exactly how the
     wrong pair gets used silently.
  2. **There are two channel maps**, differing by a silent 820 um depth shift
     (map A through 2026-10-23, map B from 2026-10-26). This script cannot know
     which one is correct for a session, but it prints and records the probe's
     y-coordinate range so the log shows which map was actually used, and it
     refuses to run if the probe's channel count disagrees with --n-chan.

Usage (inside the RIS container, or anywhere):
    python run_kilosort4_session.py \
        --data   /scratch1/.../20261021_111842.probe1.dat \
        --probe  /scratch1/.../20261021_111842_custom_neuropixels_map.json \
        --out    /scratch1/.../kilosort4 \
        --th-universal 9.2 --th-learned 7.0 \
        --device cpu --threads 8
"""
import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="<session>.probe1.dat (flat int16)")
    ap.add_argument("--probe", required=True, help="probe JSON from make_probe_map.py")
    ap.add_argument("--out", required=True, help="results directory (created)")
    # No defaults on the thresholds -- see the module docstring.
    ap.add_argument("--th-universal", type=float, required=True,
                    help="Th_universal for THIS session (9.0 on 20260916, 9.2 on 20260910/0901)")
    ap.add_argument("--th-learned", type=float, required=True,
                    help="Th_learned for THIS session (8.0 on 20260916, 7.0 on 20260910/0901)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=8,
                    help="torch threads; mirror the cores requested from LSF")
    ap.add_argument("--n-chan", type=int, default=384)
    ap.add_argument("--fs", type=float, default=30000.0)
    args = ap.parse_args()

    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(args.threads)

    import torch
    import kilosort
    from kilosort import run_kilosort
    from kilosort.io import load_probe

    torch.set_num_threads(args.threads)

    if args.device == "cuda" and not torch.cuda.is_available():
        # Falling back to CPU silently would turn a 5 h job into a multi-day one
        # without anyone noticing until the queue backed up.
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False. "
                         "Refusing to fall back to CPU silently -- pass --device cpu if "
                         "that is what you meant.")

    nbytes = os.path.getsize(args.data)
    if nbytes % (args.n_chan * 2):
        raise SystemExit("%s is %d bytes, not a whole number of %d-channel int16 samples. "
                         "Wrong --n-chan, or a truncated copy."
                         % (args.data, nbytes, args.n_chan))
    nsamp = nbytes // (args.n_chan * 2)
    rec_hours = nsamp / args.fs / 3600.0

    with open(args.probe) as f:
        pj = json.load(f)
    probe_nchan = int(pj.get("n_chan", len(pj.get("chanMap", []))))
    if probe_nchan != args.n_chan:
        raise SystemExit("probe JSON describes %d channels but --n-chan is %d. This is the "
                         "wrong map for this recording -- refusing to sort."
                         % (probe_nchan, args.n_chan))
    yc = pj.get("yc") or []
    xc = pj.get("xc") or []
    geom = {
        "probe_n_chan": probe_nchan,
        "yc_min": min(yc) if yc else None,
        "yc_max": max(yc) if yc else None,
        "xc_min": min(xc) if xc else None,
        "xc_max": max(xc) if xc else None,
    }

    print("Kilosort4 %s | torch %s | device %s | threads %d"
          % (kilosort.__version__, torch.__version__, args.device, torch.get_num_threads()))
    print("data   %s" % args.data)
    print("       %d bytes = %d samples = %.2f h at %g Hz"
          % (nbytes, nsamp, rec_hours, args.fs))
    print("probe  %s" % args.probe)
    print("       %d channels | yc %s..%s um | xc %s..%s um"
          % (probe_nchan, geom["yc_min"], geom["yc_max"], geom["xc_min"], geom["xc_max"]))
    print("       ^ the yc range identifies WHICH channel map this is. Map A and map B")
    print("         differ by 820 um of depth; check it against the session's date.")
    print("thresholds  Th_universal=%g  Th_learned=%g" % (args.th_universal, args.th_learned))

    os.makedirs(args.out, exist_ok=True)
    probe = load_probe(args.probe)
    settings = {
        "n_chan_bin": args.n_chan,
        "fs": args.fs,
        "Th_universal": args.th_universal,
        "Th_learned": args.th_learned,
    }

    t0 = time.time()
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
    wall = time.time() - t0

    # A provenance file beside the results. Without it, a sorted session on a
    # share is indistinguishable from one sorted with different settings.
    prov = {
        "kilosort_version": kilosort.__version__,
        "torch_version": torch.__version__,
        "device": args.device,
        "threads": torch.get_num_threads(),
        "data": os.path.abspath(args.data),
        "data_bytes": nbytes,
        "recording_hours": round(rec_hours, 3),
        "probe": os.path.abspath(args.probe),
        "settings": settings,
        "do_CAR": True,
        "invert_sign": False,
        "save_preprocessed_copy": False,
        "data_dtype": "int16",
        "wall_seconds": round(wall, 1),
        "wall_hours": round(wall / 3600.0, 3),
        "realtime_factor": round(wall / (nsamp / args.fs), 3),
        "probe_geometry": geom,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(args.out, "run_provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)

    print("\ndone in %.2f h (%.3f x realtime)" % (wall / 3600.0, prov["realtime_factor"]))
    print("provenance -> %s" % os.path.join(args.out, "run_provenance.json"))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""An independent, per-channel threshold sorter, built to be compared against Kilosort.

    python scripts/threshold_sort_validate.py noise   <sid> [options]
    python scripts/threshold_sort_validate.py detect  <sid> [options]
    python scripts/threshold_sort_validate.py cluster <sid> [options]
    python scripts/threshold_sort_validate.py compare <sid> [options]

WHY THIS EXISTS
---------------
Kilosort is a template-matching sorter, and template matching can split one neuron
into several templates. Nothing inside Kilosort's own output can tell you that it
did -- every diagnostic we have (the merge panel included) is computed from the
clustering it produced. So this is a second opinion that shares none of its
machinery: no templates, no drift model, no spatial clustering, no whitening across
channels. One channel at a time, a high-pass filter, a threshold, and the same four
features the review tool shows a human.

If one threshold unit's spikes are split across two Kilosort clusters, that is
evidence of an oversplit that does not come from Kilosort's own bookkeeping.

WHAT IT IS NOT
--------------
Not a better sorter. It cannot separate two neurons that land on the same channel
with similar amplitude, it double-counts every spike that appears on several
channels, and it says nothing about anything buried in the noise. It is deliberately
blunt: whatever it finds, it finds for reasons you can write on one line.

THE FOUR STAGES
---------------
noise    Per channel and per section, a robust noise level from the filtered data
         (sigma = median|x| / 0.6745, measured on sampled snippets, never on the
         whole section -- a section containing a big artefact would inflate it).
         The session is cut into --sections equal parts because the noise floor
         drifts over 4.7 hours; one threshold for the whole session is either too
         deaf at the start or too trigger-happy at the end.

detect   One pass over probe1.dat. Band-pass 300-6000 Hz per channel, find every
         negative crossing of -K sigma, take the trough, enforce a 0.5 ms dead time,
         cut a 60-sample window (20 before, 40 after, baseline from the first 5
         samples -- the same window the merge builder uses), and compute:
             trough      |min| in uV
             p2t         max - min in uV          <- the review tool's peak-to-trough
             energy      sum of squares, uV^2     <- the review tool's total energy
             width_ms    trough to the following peak
         It also measures what the NOISE does on that same channel and section, by
         cutting windows at random positions at least 2 ms away from any detected
         event. That gives an explicit null: "this is what a window containing
         nothing looks like here". Waveforms are written to disk as float16, so the
         clustering stage never has to touch the network again.

cluster  Per channel and per section: PCA on that channel's own waveforms, a
         Gaussian mixture over PC1-3 with k chosen by BIC, and then a gate that each
         component passes or fails for a stated reason. A component is a RELIABLE
         unit only if all of:
             median trough     >= --reliable_k sigma
             5th pct of trough >= --floor_k sigma      (the whole cluster is clear,
                                                        not just its middle)
             5th pct of p2t    >  the noise null's 99.9th percentile of p2t
             ISI < 1.5 ms      <= --max_refr of its spikes   (a refractory dip)
             n spikes          >= --min_spikes in the section
         Everything else is written out too, with the gate it failed. A unit that is
         not reliable has not been judged a non-neuron; it has been judged
         not-clearly-separable-from-noise by one blunt instrument.

compare  Matches each reliable unit against the Kilosort clusters whose peak channel
         is within --near_um of this channel, pairing spikes within --tol_ms. Then,
         for each threshold unit, how its spikes distribute over Kilosort clusters.
         A threshold unit whose spikes are taken by two or more Kilosort clusters,
         each with a real share, is the oversplit signature.

OUTPUT -- outside the repo, under D:\\Gil\\Popper, one folder per session:
    D:\\Gil\\Popper\\threshold_validation\\<sid>\\
        config.json                 every parameter, and the input file's size
        noise.npz                   sigma and threshold per (section, channel)
        sec##\\events.npz           channel, sample, trough, p2t, energy, width
        sec##\\waves.f16            the 60-sample windows, float16, in order
        sec##\\noise_null.npz       the null's percentiles per channel
        units.csv                   every component, reliable or not, with its reason
        units_reliable.csv          just the ones that passed
        compare_kilosort.csv        the matching, per threshold unit
        oversplit_candidates.csv    Kilosort clusters that share one threshold unit
        log_<stage>.txt

The raw data is read from Z and never written to. Nothing outside the session's own
output folder is touched.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

FS = 30000.0
NCH = 384
UV_PER_BIT = 0.018311105685598315      # overridden from merge_candidates if present
WF_PRE, WF_POST = 20, 40               # 60-sample window, as in build_merge_candidates
BASE_N = 5                             # baseline from the first 5 samples, likewise
DEAD_SAMP = 15                         # 0.5 ms dead time between troughs
OUT_ROOT = r"D:\Gil\Popper\threshold_validation"


def session_paths(sid):
    ks = os.path.join(r"Z:\Gil\Popper", sid + ".rec", sid + ".kilosort")
    return {
        "probe": os.path.join(ks, sid + ".probe1.dat"),
        "ks4": os.path.join(ks, "kilosort4"),
        "out": os.path.join(OUT_ROOT, sid),
    }


class Log(object):
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
        print(line)
        sys.stdout.flush()
        self.f.write(line + "\n")
        self.f.flush()


def n_samples_of(probe):
    n = os.path.getsize(probe)
    if n % (NCH * 2):
        raise SystemExit("%s is not a whole number of %d-channel int16 samples" % (probe, NCH))
    return n // (NCH * 2)


def open_probe(probe, n):
    return np.memmap(probe, dtype=np.int16, mode="r", shape=(n, NCH))


def make_sos(lo, hi, order):
    from scipy.signal import butter
    return butter(order, [lo / (FS / 2.0), hi / (FS / 2.0)], btype="bandpass", output="sos")


def section_bounds(n, k):
    edges = np.linspace(0, n, k + 1).astype(np.int64)
    return list(zip(edges[:-1], edges[1:]))


def filt_block(mm, sos, t0, t1, pad, uv):
    """Band-passed float32 (samples, NCH) for [t0,t1) plus pad either side.

    Returns (filtered, offset) where filtered[offset:offset+(t1-t0)] is the core.
    """
    from scipy.signal import sosfiltfilt
    a = max(0, t0 - pad)
    b = min(mm.shape[0], t1 + pad)
    raw = np.asarray(mm[a:b, :], dtype=np.float32)
    raw *= uv
    out = sosfiltfilt(sos, raw, axis=0).astype(np.float32, copy=False)
    return out, t0 - a


# ----------------------------------------------------------------------------- noise
def stage_noise(args):
    p = session_paths(args.sid)
    os.makedirs(p["out"], exist_ok=True)
    log = Log(os.path.join(p["out"], "log_noise.txt"))
    n = n_samples_of(p["probe"])
    mm = open_probe(p["probe"], n)
    sos = make_sos(args.hp, args.lp, args.order)
    secs = section_bounds(n, args.sections)
    log("session %s: %d samples, %.2f h, %d sections of %.1f s"
        % (args.sid, n, n / FS / 3600, len(secs), (secs[0][1] - secs[0][0]) / FS))
    log("noise from %d snippets of %.1f s per section, sigma = median|x| / 0.6745"
        % (args.snips, args.snip_s))

    sig = np.zeros((len(secs), NCH), dtype=np.float32)
    rng = np.random.default_rng(args.seed)
    w = int(args.snip_s * FS)
    for si, (a, b) in enumerate(secs):
        starts = np.sort(rng.integers(a, max(a + 1, b - w), size=args.snips))
        acc = np.empty((args.snips, NCH), dtype=np.float32)
        for k, s in enumerate(starts):
            f, off = filt_block(mm, sos, int(s), int(s) + w, int(0.05 * FS), args.uv)
            core = f[off:off + w]
            acc[k] = np.median(np.abs(core), axis=0) / 0.6745
        sig[si] = np.median(acc, axis=0)
        log("  section %2d/%d  sigma median %.2f uV  min %.2f  max %.2f"
            % (si + 1, len(secs), np.median(sig[si]), sig[si].min(), sig[si].max()))

    thr = -args.detect_k * sig
    np.savez(os.path.join(p["out"], "noise.npz"), sigma_uv=sig, thr_uv=thr,
             sections=np.array(secs, dtype=np.int64), detect_k=args.detect_k)
    log("wrote noise.npz   sigma over the whole probe: median %.2f uV, 5th %.2f, 95th %.2f"
        % (np.median(sig), np.percentile(sig, 5), np.percentile(sig, 95)))
    drift = sig.max(axis=0) / np.maximum(sig.min(axis=0), 1e-6)
    log("per-channel sigma drift across sections: median x%.2f, worst x%.2f  "
        "(this is why the threshold is per section)" % (np.median(drift), drift.max()))
    save_config(args, p, n, "noise")
    return 0


def save_config(args, p, n, stage):
    cfg_path = os.path.join(p["out"], "config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    cfg[stage] = {k: (v if not isinstance(v, np.generic) else v.item())
                  for k, v in vars(args).items() if k != "func"}
    cfg["session_id"] = args.sid
    cfg["n_samples"] = int(n)
    cfg["probe_dat"] = p["probe"]
    cfg["probe_bytes"] = os.path.getsize(p["probe"])
    cfg["fs_hz"] = FS
    cfg["n_channels"] = NCH
    cfg["window"] = {"pre": WF_PRE, "post": WF_POST, "baseline_n": BASE_N}
    cfg["dead_time_samples"] = DEAD_SAMP
    cfg["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ---------------------------------------------------------------------------- detect
def troughs_on(x, thr):
    """Sample indices of the trough of every run of x below thr, dead-time filtered."""
    below = x < thr
    if not below.any():
        return np.empty(0, dtype=np.int64)
    # Pad with False either side and difference: starts and ends then always come out
    # the same length, and a run touching either end of the block is handled by the
    # same arithmetic as one in the middle. Deriving them from the transitions alone
    # does not -- a run that never rises back above the threshold yields a start with
    # no matching end, and truncating to the shorter of the two silently drops it.
    pad = np.concatenate(([False], below, [False]))
    d = np.diff(pad.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)        # exclusive
    out = np.empty(starts.size, dtype=np.int64)
    for i in range(starts.size):
        s, e = starts[i], ends[i]
        out[i] = s + int(np.argmin(x[s:e]))
    if out.size < 2:
        return out
    keep = [0]
    for i in range(1, out.size):
        if out[i] - out[keep[-1]] >= DEAD_SAMP:
            keep.append(i)
        elif x[out[i]] < x[out[keep[-1]]]:
            keep[-1] = i
    return out[np.array(keep)]


def features_of(W):
    """(n,60) baseline-subtracted uV -> trough, p2t, energy, width_ms."""
    mn = W.min(axis=1)
    imn = W.argmin(axis=1)
    mx = W.max(axis=1)
    trough = -mn
    p2t = mx - mn
    energy = np.einsum("ij,ij->i", W, W)
    width = np.empty(W.shape[0], dtype=np.float32)
    for i in range(W.shape[0]):
        tail = W[i, imn[i]:]
        width[i] = (int(np.argmax(tail)) / FS * 1000.0) if tail.size > 1 else 0.0
    return (trough.astype(np.float32), p2t.astype(np.float32),
            energy.astype(np.float32), width)


def cut_windows(f, off, idx):
    """Baseline-subtracted windows around idx (core-relative) from padded f."""
    a = idx + off - WF_PRE
    ok = (a >= 0) & (a + WF_PRE + WF_POST <= f.shape[0])
    a = a[ok]
    if a.size == 0:
        return np.empty((0, WF_PRE + WF_POST), dtype=np.float32), ok
    rows = a[:, None] + np.arange(WF_PRE + WF_POST)[None, :]
    W = f[rows]
    W = W - W[:, :BASE_N].mean(axis=1, keepdims=True)
    return W.astype(np.float32, copy=False), ok


def stage_detect(args):
    p = session_paths(args.sid)
    os.makedirs(p["out"], exist_ok=True)
    log = Log(os.path.join(p["out"], "log_detect.txt"))
    nz = np.load(os.path.join(p["out"], "noise.npz"))
    sig, secs = nz["sigma_uv"], nz["sections"]
    n = n_samples_of(p["probe"])
    mm = open_probe(p["probe"], n)
    sos = make_sos(args.hp, args.lp, args.order)
    chans = parse_channels(args.channels)
    sec_sel = parse_channels(args.only_sections) if args.only_sections else range(len(secs))
    log("detecting on %d channels, %d sections, threshold -%.1f sigma, dead time %.2f ms"
        % (len(chans), len(list(sec_sel)), args.detect_k, DEAD_SAMP / FS * 1000))
    blk = int(args.block_s * FS)
    pad = int(args.pad_s * FS)
    rng = np.random.default_rng(args.seed + 1)
    t_start = time.time()

    for si in sec_sel:
        a, b = int(secs[si][0]), int(secs[si][1])
        if args.limit_s:
            # A shortened section, for proving the chain end to end before committing
            # to a four-hour pass. events.npz records the real t0/t1 it covered, so
            # nothing downstream mistakes a trial run for the whole section.
            b = min(b, a + int(args.limit_s * FS))
        sd = os.path.join(p["out"], "sec%02d" % si)
        os.makedirs(sd, exist_ok=True)
        thr_ch = (-args.detect_k * sig[si]).astype(np.float32)
        ev_ch, ev_t, ev_tr, ev_p2, ev_en, ev_wd = [], [], [], [], [], []
        wf_file = open(os.path.join(sd, "waves.f16"), "wb")
        n_wf = 0
        null_p2t = [[] for _ in range(NCH)]
        null_en = [[] for _ in range(NCH)]

        for t0 in range(a, b, blk):
            t1 = min(t0 + blk, b)
            f, off = filt_block(mm, sos, t0, t1, pad, args.uv)
            core_n = t1 - t0
            for ch in chans:
                x = f[off:off + core_n, ch]
                idx = troughs_on(x, thr_ch[ch])
                if idx.size:
                    W, ok = cut_windows(f[:, ch], off, idx)
                    if W.shape[0]:
                        tr, p2, en, wd = features_of(W)
                        ev_ch.append(np.full(W.shape[0], ch, dtype=np.int16))
                        ev_t.append((idx[ok] + t0).astype(np.int64))
                        ev_tr.append(tr); ev_p2.append(p2); ev_en.append(en); ev_wd.append(wd)
                        W.astype(np.float16).tofile(wf_file)
                        n_wf += W.shape[0]
                # the null: windows with nothing in them, on this same channel
                cand = rng.integers(WF_PRE + 1, max(WF_PRE + 2, core_n - WF_POST - 1),
                                    size=args.null_per_block)
                if idx.size:
                    far = np.min(np.abs(cand[:, None] - idx[None, :]), axis=1) > int(0.002 * FS)
                    cand = cand[far]
                if cand.size:
                    Wn, okn = cut_windows(f[:, ch], off, cand)
                    if Wn.shape[0]:
                        _, p2n, enn, _ = features_of(Wn)
                        null_p2t[ch].append(p2n); null_en[ch].append(enn)
            log("  sec %02d  %6.1f-%6.1f s   events so far %d   %.1f min elapsed"
                % (si, t0 / FS, t1 / FS, n_wf, (time.time() - t_start) / 60))
        wf_file.close()

        if ev_ch:
            np.savez(os.path.join(sd, "events.npz"),
                     channel=np.concatenate(ev_ch), sample=np.concatenate(ev_t),
                     trough_uv=np.concatenate(ev_tr), p2t_uv=np.concatenate(ev_p2),
                     energy=np.concatenate(ev_en), width_ms=np.concatenate(ev_wd),
                     sigma_uv=sig[si], t0=a, t1=b)
        else:
            np.savez(os.path.join(sd, "events.npz"),
                     channel=np.empty(0, np.int16), sample=np.empty(0, np.int64),
                     trough_uv=np.empty(0, np.float32), p2t_uv=np.empty(0, np.float32),
                     energy=np.empty(0, np.float32), width_ms=np.empty(0, np.float32),
                     sigma_uv=sig[si], t0=a, t1=b)
        q = np.array([50, 90, 99, 99.9])
        np_p2t = np.full((NCH, q.size), np.nan, dtype=np.float32)
        np_en = np.full((NCH, q.size), np.nan, dtype=np.float32)
        np_n = np.zeros(NCH, dtype=np.int64)
        for ch in chans:
            if null_p2t[ch]:
                v = np.concatenate(null_p2t[ch]); np_p2t[ch] = np.percentile(v, q); np_n[ch] = v.size
            if null_en[ch]:
                v = np.concatenate(null_en[ch]); np_en[ch] = np.percentile(v, q)
        np.savez(os.path.join(sd, "noise_null.npz"), q=q, p2t=np_p2t, energy=np_en,
                 n_windows=np_n)
        tot = sum(len(c) for c in ev_ch)
        log("sec %02d done: %d events over %.1f s = %.1f Hz per channel; waves.f16 %.2f GiB"
            % (si, tot, (b - a) / FS,
               tot / ((b - a) / FS) / max(1, len(chans)),
               os.path.getsize(os.path.join(sd, "waves.f16")) / 1024.0 ** 3))
    save_config(args, p, n, "detect")
    log("all requested sections done in %.1f min" % ((time.time() - t_start) / 60))
    return 0


# --------------------------------------------------------------------------- cluster
def load_section(out, si):
    sd = os.path.join(out, "sec%02d" % si)
    ev = np.load(os.path.join(sd, "events.npz"))
    nn = np.load(os.path.join(sd, "noise_null.npz"))
    n = ev["sample"].size
    wf = np.memmap(os.path.join(sd, "waves.f16"), dtype=np.float16, mode="r",
                   shape=(n, WF_PRE + WF_POST)) if n else None
    return sd, ev, nn, wf


def gmm_labels(X, kmax, seed, min_per):
    """Label X by the BIC-best Gaussian mixture, k in 1..kmax. Returns (labels, k, bic)."""
    from sklearn.mixture import GaussianMixture
    best = None
    for k in range(1, kmax + 1):
        if X.shape[0] < max(min_per * k, 10 * k):
            break
        g = GaussianMixture(n_components=k, covariance_type="full", random_state=seed,
                            n_init=1, reg_covar=1e-4, max_iter=200)
        try:
            g.fit(X)
        except Exception:
            continue
        b = g.bic(X)
        if best is None or b < best[0]:
            best = (b, k, g)
    if best is None:
        return np.zeros(X.shape[0], dtype=np.int16), 1, float("nan")
    return best[2].predict(X).astype(np.int16), best[1], float(best[0])


def stage_cluster(args):
    p = session_paths(args.sid)
    log = Log(os.path.join(p["out"], "log_cluster.txt"))
    from sklearn.decomposition import PCA
    nz = np.load(os.path.join(p["out"], "noise.npz"))
    sig, secs = nz["sigma_uv"], nz["sections"]
    chans = parse_channels(args.channels)
    sec_sel = (parse_channels(args.only_sections) if args.only_sections
               else [i for i in range(len(secs))
                     if os.path.exists(os.path.join(p["out"], "sec%02d" % i, "events.npz"))])
    log("clustering %d channels over %d sections; PCA on <=%d waveforms, GMM k<=%d"
        % (len(chans), len(sec_sel), args.pca_max, args.kmax))
    rng = np.random.default_rng(args.seed + 2)
    rows = []
    refr = int(round(args.refr_ms / 1000.0 * FS))

    for si in sec_sel:
        sd, ev, nn, wf = load_section(p["out"], si)
        if wf is None:
            log("  sec %02d: no events" % si)
            continue
        ch_all = ev["channel"]
        order = np.argsort(ch_all, kind="stable")
        labels = np.full(ch_all.size, -1, dtype=np.int16)
        qs = [float(x) for x in nn["q"]]
        if args.null_q not in qs:
            raise SystemExit("--null_q %g was not measured; detect stored %s"
                             % (args.null_q, qs))
        qi = qs.index(args.null_q)
        sec_s = (int(ev["t1"]) - int(ev["t0"])) / FS
        for ch in chans:
            sel = order[np.searchsorted(ch_all[order], ch, "left"):
                        np.searchsorted(ch_all[order], ch, "right")]
            s = float(sig[si, ch])
            if sel.size < args.min_spikes:
                rows.append(dict(section=si, channel=ch, comp=-1, n_spikes=int(sel.size),
                                 sigma_uv=s, reliable=0,
                                 reason="only %d events in the section" % sel.size))
                continue
            sub = sel if sel.size <= args.pca_max else rng.choice(sel, args.pca_max, False)
            W = np.asarray(wf[np.sort(sub)], dtype=np.float32)
            pca = PCA(n_components=min(3, W.shape[1], W.shape[0]), random_state=args.seed)
            pca.fit(W)
            P = pca.transform(np.asarray(wf[sel], dtype=np.float32))
            Z = (P - P.mean(axis=0)) / np.maximum(P.std(axis=0), 1e-9)
            lab, k, bic = gmm_labels(Z, args.kmax, args.seed, args.min_spikes)
            labels[sel] = lab
            null_p2t = float(nn["p2t"][ch, qi])
            t = ev["sample"][sel]
            for c in range(k):
                m = lab == c
                nsp = int(m.sum())
                tr, p2 = ev["trough_uv"][sel][m], ev["p2t_uv"][sel][m]
                tt = np.sort(t[m])
                isi = np.diff(tt)
                frac_refr = float((isi < refr).mean()) if isi.size else 1.0
                med_tr = float(np.median(tr)) if nsp else 0.0
                med_p2 = float(np.median(p2)) if nsp else 0.0
                # The FLOOR is about the bulk of a cluster, not its deepest tail.
                # Detection at --detect_k sigma truncates every cluster's small end at
                # exactly that value, so a 5th-percentile floor set above it can only
                # be passed by a cluster with no small spikes at all -- it failed 66 of
                # the first 69 components measured, which is a property of the detector,
                # not of the neurons. The 25th percentile asks the question actually
                # meant: is most of this cluster clear of the noise?
                lo_tr = float(np.percentile(tr, args.floor_pct)) if nsp else 0.0
                lo_p2 = float(np.percentile(p2, args.floor_pct)) if nsp else 0.0
                why = []
                if nsp < args.min_spikes:
                    why.append("%d spikes < %d" % (nsp, args.min_spikes))
                if med_tr < args.reliable_k * s:
                    why.append("median trough %.1f uV < %.1f (%.1f sigma)"
                               % (med_tr, args.reliable_k * s, args.reliable_k))
                if lo_tr < args.floor_k * s:
                    why.append("%dth pct trough %.1f uV < %.1f (%.1f sigma)"
                               % (args.floor_pct, lo_tr, args.floor_k * s, args.floor_k))
                if not (med_p2 > null_p2t):
                    why.append("median p2t %.1f uV does not clear the noise null's "
                               "%.4gth pct %.1f uV" % (med_p2, args.null_q, null_p2t))
                if frac_refr > args.max_refr:
                    why.append("%.1f%% of ISIs under %.1f ms > %.1f%%"
                               % (100 * frac_refr, args.refr_ms, 100 * args.max_refr))
                rows.append(dict(
                    section=si, channel=ch, comp=c, n_spikes=nsp, sigma_uv=s,
                    rate_hz=nsp / sec_s, median_trough_uv=med_tr,
                    trough_sigma=med_tr / s if s > 0 else 0.0,
                    floor_trough_uv=lo_tr, median_p2t_uv=float(np.median(p2)) if nsp else 0.0,
                    floor_p2t_uv=lo_p2, noise_null_p2t=null_p2t,
                    median_energy=float(np.median(ev["energy"][sel][m])) if nsp else 0.0,
                    median_width_ms=float(np.median(ev["width_ms"][sel][m])) if nsp else 0.0,
                    frac_isi_refr=frac_refr, gmm_k=k, gmm_bic=bic,
                    reliable=0 if why else 1, reason="; ".join(why) if why else "passed"))
        np.savez(os.path.join(sd, "clusters.npz"), label=labels)
        nrel = sum(1 for r in rows if r.get("section") == si and r["reliable"])
        log("  sec %02d: %d components, %d reliable" % (si, sum(
            1 for r in rows if r.get("section") == si and r["comp"] >= 0), nrel))

    import csv as _csv
    keys = ["section", "channel", "comp", "n_spikes", "rate_hz", "sigma_uv",
            "median_trough_uv", "trough_sigma", "floor_trough_uv", "median_p2t_uv",
            "floor_p2t_uv", "noise_null_p2t", "median_energy", "median_width_ms",
            "frac_isi_refr", "gmm_k", "gmm_bic", "reliable", "reason"]
    for name, keep in (("units.csv", lambda r: True),
                       ("units_reliable.csv", lambda r: r["reliable"] == 1)):
        with open(os.path.join(p["out"], name), "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                if keep(r):
                    w.writerow(r)
    nrel = sum(1 for r in rows if r["reliable"] == 1)
    log("wrote units.csv (%d rows) and units_reliable.csv (%d rows)" % (len(rows), nrel))
    n = n_samples_of(p["probe"]) if os.path.exists(p["probe"]) else 0
    save_config(args, p, n, "cluster")
    return 0


# --------------------------------------------------------------------------- compare
def stage_compare(args):
    p = session_paths(args.sid)
    log = Log(os.path.join(p["out"], "log_compare.txt"))
    import csv as _csv
    ks = p["ks4"]
    st = np.load(os.path.join(ks, "spike_times.npy")).ravel().astype(np.int64)
    sc = np.load(os.path.join(ks, "spike_clusters.npy")).ravel().astype(np.int64)
    pos = np.load(os.path.join(ks, "channel_positions.npy"))
    o = np.argsort(st, kind="stable")
    st, sc = st[o], sc[o]
    log("Kilosort: %d spikes, %d clusters" % (st.size, np.unique(sc).size))

    peak = {}
    cc = os.path.join(r"D:\Gil\spike_sorting_agent\outputs",
                      "corrected_classification_%s.csv" % args.sid)
    lab = {}
    with open(cc, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            peak[int(row["unit_id"])] = int(float(row["peak_channel"]))
            lab[int(row["unit_id"])] = row["label"]
    log("peak channel known for %d Kilosort clusters (from %s)" % (len(peak), os.path.basename(cc)))

    # Kilosort's own view of how alike two co-located clusters are, for telling an
    # oversplit (two clusters that look the same) from this sorter lumping two real
    # neurons together (two clusters that do not).
    sim = {}
    mc = os.path.join(r"D:\Gil\spike_sorting_agent\outputs",
                      "merge_candidates_%s.csv" % args.sid)
    mcj = os.path.join(r"D:\Gil\spike_sorting_agent\outputs",
                       "merge_candidates_%s.json" % args.sid)
    if os.path.exists(mcj):
        with open(mcj, encoding="utf-8") as f:
            d = json.load(f)
        for pr in d.get("pairs", []):
            sim[(pr["unit_a"], pr["unit_b"])] = (pr.get("template_similarity"),
                                                 pr.get("waveform_corr"))
        log("loaded %d co-located Kilosort pairs with similarity" % len(sim))

    units = []
    with open(os.path.join(p["out"], "units_reliable.csv"), encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            units.append(row)
    log("%d reliable threshold units to match, tol %.2f ms, neighbourhood %.0f um"
        % (len(units), args.tol_ms, args.near_um))

    tol = int(round(args.tol_ms / 1000.0 * FS))
    secs = np.load(os.path.join(p["out"], "noise.npz"))["sections"]
    out_rows, over_rows = [], []
    cache = {}
    for u in units:
        si, ch, comp = int(u["section"]), int(u["channel"]), int(u["comp"])
        if si not in cache:
            sd, ev, nn, wf = load_section(p["out"], si)
            cache = {si: (ev, np.load(os.path.join(sd, "clusters.npz"))["label"])}
        ev, labels = cache[si]
        m = (ev["channel"] == ch) & (labels == comp)
        T = np.sort(ev["sample"][m])
        a, b = int(secs[si][0]), int(secs[si][1])
        lo, hi = np.searchsorted(st, [a, b])
        kt, kc = st[lo:hi], sc[lo:hi]
        near = [c for c in np.unique(kc)
                if c in peak and np.hypot(*(pos[peak[c]] - pos[ch])) <= args.near_um]
        shares = []
        anym = np.zeros(T.size, dtype=bool)
        for c in near:
            ct = kt[kc == c]
            if ct.size == 0:
                continue
            j = np.searchsorted(ct, T)
            left = np.clip(j - 1, 0, ct.size - 1)
            right = np.clip(j, 0, ct.size - 1)
            d = np.minimum(np.abs(T - ct[left]), np.abs(T - ct[right]))
            m = d <= tol
            if m.any():
                anym |= m
                shares.append((float(m.mean()), int(c), int(ct.size)))
        # A share out of ALL my spikes is diluted by everything Kilosort never detected
        # -- at a 4 sigma threshold that is a large minority, and it makes every share
        # look small. The oversplit question is only about spikes Kilosort DID detect:
        # of those, how many clusters did it spread them over? So the test runs on the
        # conditional share, and both numbers are reported.
        any_share = float(anym.mean()) if T.size else 0.0
        cond = [(f / any_share if any_share > 0 else 0.0, c, n) for f, c, n in shares]
        shares.sort(reverse=True)
        cond.sort(reverse=True)
        big = [s for s in cond if s[0] >= args.share]
        best = cond[0] if cond else (0.0, -1, 0)
        out_rows.append(dict(
            section=si, channel=ch, comp=comp, n_spikes=int(T.size),
            trough_sigma=float(u["trough_sigma"]), rate_hz=float(u["rate_hz"]),
            n_ks_near=len(near),
            any_share=round(any_share, 4),
            best_ks=best[1], best_share_of_matched=round(best[0], 4),
            best_share_of_all=round(shares[0][0], 4) if shares else 0.0,
            best_ks_label=lab.get(best[1], ""),
            n_ks_sharing=len(big),
            ks_sharing="|".join("%d:%.2f" % (c, f) for f, c, _ in big),
            total_share=round(sum(f for f, _, _ in big), 4)))
        if len(big) >= 2:
            for i in range(len(big)):
                for j2 in range(i + 1, len(big)):
                    # Sort the id pair AND its shares together. Canonicalising the pair
                    # key while reporting the shares in the original order silently
                    # swaps them whenever the sort reorders the pair.
                    (ca, fa), (cb, fb) = sorted(((big[i][1], big[i][0]),
                                                 (big[j2][1], big[j2][0])))
                    ts, wc = sim.get((ca, cb), (None, None))
                    over_rows.append(dict(
                        section=si, channel=ch, comp=comp,
                        n_spikes=int(T.size), ks_a=ca, ks_b=cb,
                        share_a=round(fa, 4), share_b=round(fb, 4),
                        ks_a_label=lab.get(ca, ""), ks_b_label=lab.get(cb, ""),
                        template_similarity=ts, waveform_corr=wc,
                        reading=("the two Kilosort clusters look alike -- oversplit"
                                 if (ts or 0) >= 0.5 else
                                 "the two Kilosort clusters do NOT look alike -- more "
                                 "likely this sorter lumped two neurons on one channel")))

    for name, rws in (("compare_kilosort.csv", out_rows),
                      ("oversplit_candidates.csv", over_rows)):
        if not rws:
            log("nothing to write for %s" % name)
            continue
        with open(os.path.join(p["out"], name), "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(rws[0].keys()))
            w.writeheader()
            w.writerows(rws)
        log("wrote %s (%d rows)" % (name, len(rws)))
    if out_rows:
        sh = np.array([r["best_share_of_matched"] for r in out_rows])
        log("best-match share across threshold units: median %.2f, %d of %d below 0.5"
            % (np.median(sh), int((sh < 0.5).sum()), sh.size))
        log("%d threshold units are shared by 2+ Kilosort clusters at >=%.0f%% each"
            % (sum(1 for r in out_rows if r["n_ks_sharing"] >= 2), 100 * args.share))
    return 0


def parse_channels(spec):
    if spec in (None, "", "all"):
        return list(range(NCH))
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    def common(q):
        q.add_argument("sid")
        q.add_argument("--sections", type=int, default=20,
                       help="equal parts the session is cut into (default 20)")
        q.add_argument("--hp", type=float, default=300.0)
        q.add_argument("--lp", type=float, default=6000.0)
        q.add_argument("--order", type=int, default=3)
        q.add_argument("--detect_k", type=float, default=5.0,
                       help="detection threshold in sigma (default 5)")
        q.add_argument("--uv", type=float, default=UV_PER_BIT)
        q.add_argument("--seed", type=int, default=7)

    n = sub.add_parser("noise"); common(n)
    n.add_argument("--snips", type=int, default=12)
    n.add_argument("--snip_s", type=float, default=1.0)
    n.set_defaults(func=stage_noise)

    d = sub.add_parser("detect"); common(d)
    d.add_argument("--channels", default="all")
    d.add_argument("--only_sections", default=None)
    d.add_argument("--block_s", type=float, default=20.0)
    d.add_argument("--pad_s", type=float, default=0.1)
    d.add_argument("--null_per_block", type=int, default=120)
    d.add_argument("--limit_s", type=float, default=0.0,
                   help="only the first N seconds of each section (0 = the whole thing); "
                        "for trial runs")
    d.set_defaults(func=stage_detect)

    c = sub.add_parser("cluster"); common(c)
    c.add_argument("--channels", default="all")
    c.add_argument("--only_sections", default=None)
    c.add_argument("--pca_max", type=int, default=20000,
                   help="waveforms the PCA is fitted on per channel and section")
    c.add_argument("--kmax", type=int, default=4, help="largest mixture tried")
    c.add_argument("--min_spikes", type=int, default=100,
                   help="a component below this is not judged (default 100 per section)")
    c.add_argument("--reliable_k", type=float, default=6.0,
                   help="median trough must reach this many sigma (default 6)")
    c.add_argument("--floor_k", type=float, default=5.0,
                   help="the floor percentile of trough must reach this many sigma")
    c.add_argument("--floor_pct", type=float, default=25.0,
                   help="which percentile the floor test uses (default 25; a value at "
                        "or below the detection threshold makes the test vacuous)")
    c.add_argument("--null_q", type=float, default=99.0,
                   help="percentile of the noise null the median p2t must clear; must "
                        "be one the detect stage measured (50, 90, 99, 99.9)")
    c.add_argument("--refr_ms", type=float, default=1.5)
    c.add_argument("--max_refr", type=float, default=0.02,
                   help="largest allowed fraction of ISIs inside the refractory window")
    c.set_defaults(func=stage_cluster)

    m = sub.add_parser("compare"); common(m)
    m.add_argument("--tol_ms", type=float, default=0.5,
                   help="two spikes count as the same event within this (default 0.5 ms)")
    m.add_argument("--near_um", type=float, default=60.0,
                   help="a Kilosort cluster is a candidate if its peak channel is this "
                        "close to the threshold unit's channel (default 60 um)")
    m.add_argument("--share", type=float, default=0.2,
                   help="a Kilosort cluster 'shares' a threshold unit if it accounts "
                        "for this fraction of its spikes (default 0.2)")
    m.set_defaults(func=stage_compare)
    return ap


def main():
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

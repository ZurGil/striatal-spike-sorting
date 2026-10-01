"""
Flag-gated patches for Kilosort4. Nothing here changes Kilosort unless
`enable()` is called, and the installed package is never modified.

HOW IT WORKS. `enable("subsample_align")` replaces `kilosort.spikedetect.run`
with the version below, which is a vendored copy of the original with ONE
inserted block. If the flag is off, the inserted block is skipped and the
original line runs, so disabling the patch reproduces stock Kilosort exactly.
`disable()` puts the original function back. Nothing is written to the
kilosort package directory, so uninstalling this is deleting an import.

------------------------------------------------------------------------------
WHAT `subsample_align` CHANGES, AND WHY THAT EXACT SPOT

Kilosort clusters spikes on `tF`, the projection of each spike's snippet onto
a learned 6-component basis `wPCA`. In `spikedetect.run` that happens here:

    xsub  = X[iC[:,xy[:,:1]], xy[:,1:2] + tarange]   # <- INTEGER positions
    xfeat = xsub @ ops['wPCA'].T

`xy[:,1]` is an integer sample index, so every snippet carries up to half a
sample of residual timing error, and that error goes straight into the
features the clustering runs on. Measured on this project's own data
(RESEARCH_LOG 5v): half a sample of jitter moves a unit's feature vector 18%
of the distance separating genuinely different units, and about 11.8% of
unit pairs within 40um sit closer together than that. Our wavelet estimator
measures the offset to ~0.035 samples RMS against known ground truth (5x),
an eightfold improvement on leaving it alone.

Note that `align_U` in template_matching.py is NOT the right place, even
though it is the obviously integer-shifting function. It aligns templates to
each other AFTER they are built, and a template is already an average of
jittered spikes -- realigning the average cannot un-blur it. The blur has to
be prevented where the snippets are measured, which is here.

THE IMPLEMENTATION TRICK. Resampling every snippet per spike would be slow
inside the batch loop. It is not necessary, because the features are linear
projections:

    <shift(x, -d), w>  =  <x, shift(w, +d)>

Aligning a snippet by -d and projecting onto the basis is identical to
projecting the raw snippet onto a basis shifted by +d. So we precompute the
basis at a grid of sub-sample offsets ONCE, and each spike simply uses the
pre-shifted basis nearest its measured offset. No snippet is ever resampled.
With N_BINS=41 the offset grid is spaced 0.025 samples, giving a
quantization error around 0.007 samples RMS -- five times smaller than the
estimator's own 0.035, so it adds nothing material.

WHICH FREQUENCY. The per-unit frequency selection used elsewhere needs a
unit's template, and at detection time no units exist yet. So f0 is chosen
once from `wPCA[0]` (effectively the mean spike shape across the recording)
using the same timing criterion, argmax f*|W(f)|, established in 5o. The
reference phase is taken from the same waveform. For clustering what matters
is that every spike is aligned to the SAME phase, not that the phase is
absolutely correct -- consistency is what removes jitter from the feature
space, and a shared constant offset is harmless.
"""
import numpy as np
import torch
from torch.nn.functional import conv1d
from scipy.interpolate import interp1d
from tqdm import tqdm

import kilosort
from kilosort import spikedetect
from kilosort.spikedetect import (extract_wPCA_wTEMP, get_waves, template_centers,
                                   nearest_chans, yweighted, template_match)
from kilosort.utils import log_performance

_ORIGINAL_RUN = spikedetect.run
_ENABLED = None
N_BINS = 41
MAX_SHIFT = 0.5
N_CYCLES = 3.0
F_LO, F_HI, N_GRID = 300.0, 8000.0, 60

AVAILABLE = ("subsample_align", "amplitude_normalize", "align_and_amp_norm")

# ---------------------------------------------------------------------------
# WHY amplitude_normalize EXISTS, AND THE CORRECTION BEHIND IT
#
# RESEARCH_LOG 5z measured that Kilosort's fragments of a single injected
# neuron are separated by AMPLITUDE about 370x more strongly than by timing,
# and I attributed that to burst-driven bimodal amplitude. A population check
# (5aa) refuted BOTH halves of that story:
#   * real amplitude distributions are NOT bimodal -- median bimod_score
#     0.000 across 443 units, only 0.7% reach the 0.6 split threshold
#   * bursting does not drive amplitude here -- amplitude vs log(ISI)
#     correlates -0.083 at the median, the wrong sign, with a real drop in
#     only 14.7% of units
# and a second check ruled out the obvious artifact: drawing snippets from
# the whole recording inflated amplitude spread by only 1.05x, so the test
# data's amplitude variation is realistic.
#
# The F-ratio in 5z is also partly CIRCULAR: if a split happens along the
# amplitude direction for any reason, the resulting fragments must differ in
# amplitude. It identifies the AXIS of the cut, not its cause.
#
# So the live hypothesis is no longer "bimodal amplitude causes splits". It
# is that amplitude is simply the LARGEST-VARIANCE direction within a unit's
# spikes, and Kilosort seeds 200 clusters per spatial region with kmeans++
# (clustering_qr.cluster, nclust=200, not user-settable) -- an over-seeded
# k-means will place its boundaries along the direction of greatest spread,
# which cuts a perfectly unimodal distribution into slices. Note also that
# swarmsplitter's FIRST criterion, tstat[kk,0] < 0.2, keeps a split without
# ever consulting bimodality.
#
# This patch is still the right experiment under the new hypothesis: removing
# amplitude from the features collapses that direction, so there is nothing
# for an over-seeded k-means to cut along. Whether that helps is exactly what
# needs measuring -- and a real risk is that it does not, because the
# over-seeding would then just cut along whatever direction is next largest.
#
# KNOWN RISK IN THE IMPLEMENTATION: tF feeds both the clustering AND the
# template construction (Wall). Normalizing removes scale information that
# downstream template matching may rely on, so this could degrade things for
# reasons unrelated to the hypothesis. That is a reason to measure it, not a
# reason to assume the outcome.
# ---------------------------------------------------------------------------
EPS = 1e-6


def enable(config):
    """Turn on one patch. Raises on an unknown name rather than silently
    running vanilla, which would make a comparison meaningless."""
    global _ENABLED
    if config not in AVAILABLE:
        raise ValueError(f"unknown patch {config!r}; available: {AVAILABLE}")
    _ENABLED = config
    spikedetect.run = _patched_run
    kilosort.spikedetect.run = _patched_run
    return config


def disable():
    global _ENABLED
    _ENABLED = None
    spikedetect.run = _ORIGINAL_RUN
    kilosort.spikedetect.run = _ORIGINAL_RUN


def is_enabled():
    return _ENABLED


# --------------------------------------------------------------------------
# the sub-sample machinery
# --------------------------------------------------------------------------

def _morlet(f0, fs, nt, center, n_cycles=N_CYCLES):
    """Complex probe on the snippet's own sample grid, centred on `center`."""
    t = (np.arange(nt) - center) / fs
    half_width_s = n_cycles / f0 / 2.0
    sigma = half_width_s / 2.5
    win = np.exp(-0.5 * (t / sigma) ** 2)
    return np.exp(1j * 2 * np.pi * f0 * t) * win


def _select_f0(waveform, fs, center):
    """argmax f*|W(f)| -- the timing criterion (RESEARCH_LOG 5o). Choosing by
    |W| alone is the bug that made alignment actively harmful, because the
    same f0 also sets the phase-to-time conversion fs/(2*pi*f0)."""
    nt = len(waveform)
    grid = np.logspace(np.log10(F_LO), np.log10(F_HI), N_GRID)
    best, best_s = grid[0], -np.inf
    for f in grid:
        psi = _morlet(f, fs, nt, center)
        s = abs(np.dot(waveform, np.conj(psi))) * f
        if s > best_s:
            best_s, best = s, f
    return float(best)


def _build_shifted_bases(wPCA, offsets):
    """wPCA re-sampled at each sub-sample offset.

    Uses <shift(x,-d), w> == <x, shift(w,+d)>, so shifting the BASIS is
    equivalent to aligning every snippet, at a fraction of the cost.
    """
    n_pc, nt = wPCA.shape
    x = np.arange(nt)
    out = np.empty((len(offsets), n_pc, nt), dtype=np.float32)
    for j, d in enumerate(offsets):
        for p in range(n_pc):
            ip = interp1d(x, wPCA[p], kind="cubic", bounds_error=False,
                          fill_value=0.0)
            out[j, p] = ip(x - d)       # +d shift of the basis
    return out


def _prepare(ops, device):
    """Everything that does not depend on the data, computed once."""
    wPCA = ops["wPCA"].cpu().numpy() if torch.is_tensor(ops["wPCA"]) else np.asarray(ops["wPCA"])
    wPCA = np.asarray(wPCA, dtype=np.float64)
    nt = ops["nt"]
    center = nt // 2                 # spikedetect snippets are centred here,
                                     # NOT at nt0min -- see xy[:,1:2]+tarange
    fs = float(ops["fs"])
    ref_wave = wPCA[0]
    f0 = _select_f0(ref_wave, fs, center)
    psi = _morlet(f0, fs, nt, center)
    ref_phase = float(np.angle(np.dot(ref_wave, np.conj(psi))))

    offsets = np.linspace(-MAX_SHIFT, MAX_SHIFT, N_BINS)
    bases = _build_shifted_bases(wPCA, offsets)

    print(f"[subsample_align] f0 = {f0:.0f} Hz (cycle {fs/f0:.1f} samples), "
          f"{N_BINS} offset bins spaced {offsets[1]-offsets[0]:.4f} samples")
    return dict(
        f0=f0, omega=2 * np.pi * f0 / fs, ref_phase=ref_phase,
        psi_re=torch.from_numpy(np.real(psi)).float().to(device),
        psi_im=torch.from_numpy(np.imag(psi)).float().to(device),
        bases=torch.from_numpy(bases).float().to(device),
        offsets=torch.from_numpy(offsets).float().to(device),
        n_bins=N_BINS)


def _normalize_amplitude(xsub):
    """Divide each spike's snippet by its own magnitude across the whole
    channel group, so only SHAPE survives into the features.

    The norm is taken over channels and time together, not per channel: a
    spike is one event with one amplitude, and normalizing each channel
    separately would destroy the spatial footprint (pillar 1c), which is the
    one feature this project has measured to be decisive.
    """
    # xsub is (nC, nsp, nt); norm per spike across channels and time
    n = torch.sqrt((xsub ** 2).sum(dim=(0, 2))).clamp_min(EPS)   # (nsp,)
    return xsub / n.view(1, -1, 1)


def _features_aligned(xsub, prep):
    """Sub-sample-aligned PC features for a batch of snippets.

    xsub : (nC, nsp, nt) -- channel 0 is the nearest channel to the template,
           which is what the offset is measured on (one spike, one time).
    """
    peak = xsub[0]                                        # (nsp, nt)
    # W = sum x(t) conj(psi(t)) = (x.psi_re) - i (x.psi_im)
    wr = peak @ prep["psi_re"]
    wi = -(peak @ prep["psi_im"])
    phase = torch.atan2(wi, wr) - prep["ref_phase"]
    phase = torch.atan2(torch.sin(phase), torch.cos(phase))   # rewrap
    delta = -phase / prep["omega"]
    delta = torch.clamp(delta, -MAX_SHIFT, MAX_SHIFT)

    step = (2 * MAX_SHIFT) / (prep["n_bins"] - 1)
    idx = torch.round((delta + MAX_SHIFT) / step).long().clamp_(0, prep["n_bins"] - 1)
    B = prep["bases"][idx]                                # (nsp, n_pc, nt)
    return torch.einsum("cst,spt->csp", xsub, B), delta


# --------------------------------------------------------------------------
# vendored copy of spikedetect.run, with ONE inserted block
# --------------------------------------------------------------------------

def _patched_run(ops, bfile, device=torch.device('cuda'), progress_bar=None,
                 clear_cache=False):
    if _ENABLED is None:
        return _ORIGINAL_RUN(ops, bfile, device=device,
                             progress_bar=progress_bar, clear_cache=clear_cache)

    sig = ops['settings']['min_template_size']
    nsizes = ops['settings']['template_sizes']

    if ops['settings']['templates_from_data']:
        ops['wPCA'], ops['wTEMP'] = extract_wPCA_wTEMP(
            ops, bfile, nt=ops['nt'], twav_min=ops['nt0min'],
            Th_single_ch=ops['settings']['Th_single_ch'], nskip=25,
            device=device)
    else:
        ops['wPCA'], ops['wTEMP'] = get_waves(ops, device=device)

    ops = template_centers(ops)
    [ys, xs] = np.meshgrid(ops['yup'], ops['xup'])
    ys, xs = ys.flatten(), xs.flatten()
    xc, yc = ops['xc'], ops['yc']

    nC = ops['settings']['nearest_chans']
    nC2 = ops['settings']['nearest_templates']
    iC, ds = nearest_chans(ys, yc, xs, xc, nC, device=device)

    igood = ds[0, :] <= ops['max_channel_distance'] ** 2
    iC = iC[:, igood]
    ds = ds[:, igood]
    ys = ys[igood]
    xs = xs[igood]
    ops['ycup'], ops['xcup'] = ys, xs

    iC2, _ = nearest_chans(ys, ys, xs, xs, nC2, device=device)

    ds_torch = torch.from_numpy(ds).to(device).float()
    template_sizes = sig * (1 + torch.arange(nsizes, device=device))
    weigh = torch.exp(-ds_torch.unsqueeze(-1) / template_sizes ** 2)
    weigh = torch.permute(weigh, (2, 0, 1)).contiguous()
    weigh = weigh / (weigh ** 2).sum(1).unsqueeze(1) ** .5

    st = np.zeros((10 ** 6, 6), 'float64')
    tF = np.zeros((10 ** 6, nC, ops['settings']['n_pcs']), 'float32')

    # ===================== INSERTED: sub-sample alignment setup ============
    prep = _prepare(ops, device)
    shift_log = []
    # =======================================================================

    k = 0
    nt = ops['nt']
    tarange = torch.arange(-(nt // 2), nt // 2 + 1, device=device)
    prog = tqdm(np.arange(bfile.n_batches), miniters=200 if progress_bar else None,
                mininterval=60 if progress_bar else None)
    try:
        for ibatch in prog:
            X = bfile.padded_batch_to_torch(ibatch, ops)
            xy, imax, amp, adist = template_match(X, ops, iC, iC2, weigh, device=device)
            yct = yweighted(yc, iC, adist, xy, device=device)
            nsp = len(xy)

            if k + nsp > st.shape[0]:
                st = np.concatenate((st, np.zeros_like(st)), 0)
                tF = np.concatenate((tF, np.zeros_like(tF)), 0)

            xsub = X[iC[:, xy[:, :1]], xy[:, 1:2] + tarange]

            # ============== INSERTED: the actual change =====================
            # original line was:  xfeat = xsub @ ops['wPCA'].T
            if _ENABLED in ("amplitude_normalize", "align_and_amp_norm"):
                xsub_f = _normalize_amplitude(xsub)
            else:
                xsub_f = xsub
            if _ENABLED in ("subsample_align", "align_and_amp_norm"):
                xfeat, delta = _features_aligned(xsub_f, prep)
                if nsp > 0 and ibatch % 20 == 0:
                    shift_log.append(delta.abs().median().item())
            else:
                xfeat = xsub_f @ ops['wPCA'].T
            # ================================================================

            tF[k:k + nsp] = xfeat.transpose(0, 1).cpu().numpy()

            st[k:k + nsp, 0] = ((xy[:, 1].cpu().numpy() - nt) / ops['fs']
                                + ibatch * (ops['batch_size'] / ops['fs']))
            st[k:k + nsp, 1] = yct.cpu().numpy()
            st[k:k + nsp, 2] = amp.cpu().numpy()
            st[k:k + nsp, 3] = imax.cpu().numpy()
            st[k:k + nsp, 4] = ibatch
            st[k:k + nsp, 5] = xy[:, 0].cpu().numpy()

            k = k + nsp

            if progress_bar is not None:
                progress_bar.emit(int((ibatch + 1) / bfile.n_batches * 100))
    except:
        raise

    log_performance(None, 'debug', f'Batch {ibatch}') if False else None

    if shift_log:
        print(f"[subsample_align] median |correction| across batches: "
              f"{np.median(shift_log):.4f} samples "
              f"(range {np.min(shift_log):.4f}-{np.max(shift_log):.4f})")

    st = st[:k]
    tF = tF[:k]
    ops['iC'] = iC
    ops['iC2'] = iC2
    ops['weigh'] = weigh
    return st, tF, ops

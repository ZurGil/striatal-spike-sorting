# -*- coding: utf-8 -*-
"""Second pass over the review tool's merge panel, answering four objections.

Applied on top of patch_review_tool_merge_panel.py (which put the panel there in the
first place). Run against a freshly-read copy of the published page; the two scripts
in order reproduce the live artifact.

WHAT CHANGED AND WHY
--------------------
1. THE DIP IS MEASURED AGAINST ITS OWN SHOULDERS, NOT THE FAR FLANKS.
   Gil's objection: a ratio normalised by the far flanks is the same construction as
   the Kilosort contamination metric that misbehaves. He is right, and it is not
   academic -- on this session 30 pairs classed "one neuron" by the >=10 ms flank
   ratio become 13 when measured against the 2-5 ms shoulders and tested against
   Poisson chance, and 59 pairs have shoulders disagreeing with flanks by >=1.5x.
   Pair 233+236, which the old metric ranked as the textbook merge (0.31), is 0.606
   against its own shoulders with p=0.06: not a dip at all.
   Both baselines are now shown, with the disagreement called out, because when they
   differ the correlogram has broad structure and neither number stands alone.

2. FIRING RATES SIT ABOVE THE MERGED ACG.
   With a large rate asymmetry the merged train is almost entirely the fast unit's
   spikes, so the merged violation PERCENTAGE barely moves however wrong the pairing
   is. The rates, the ratio, and -- the number that is actually invariant to the
   asymmetry -- the COUNT of refractory violations the interleaving creates are all
   stated. Above 5:1 the panel says outright that the ACG shape is not evidence.

3. BOTH UNITS' WAVEFORMS ACROSS CHANNELS.
   The data always had 10 channels per unit with their real micron positions; only
   the single shared channel was ever drawn. Now there is a footprint: every channel
   either unit covers, each trace at its true position on the probe, one microvolt
   scale throughout. One neuron has one footprint.

4. "worth a look" IS STATED IN FULL where the panel defines it, rather than left to
   a tooltip.
"""
import io
import os
import sys

SRC = (sys.argv[1] if len(sys.argv) > 1 else
        r"C:\Users\Adam\.claude\projects\D--Gil-spike-sorting-agent"
        r"\8186c809-76fa-41ff-bbca-59d44125e5fc\tool-results"
        r"\artifact-74dec064-1792157641-801f.html")
DST = (sys.argv[2] if len(sys.argv) > 2 else
        r"C:\Users\Adam\AppData\Local\Temp\claude\D--Gil-spike-sorting-agent"
        r"\8186c809-76fa-41ff-bbca-59d44125e5fc\scratchpad\review_tool_v2.html")

# --------------------------------------------------------------------- new code

HELPERS = u"""
/* --- the dip, measured against its own shoulders --------------------------- */
/* A refractory ratio normalised by the FAR flanks (>=10 ms) is the same construction
   as the contamination metrics that misbehave: any CCG with broad structure --
   bursting, shared drive, slow co-modulation -- has flanks that are not a valid
   baseline for the refractory window, and the ratio then reports a dip that is not
   there. Measured instead against the 2-5 ms shoulders either side of the window,
   and tested against Poisson chance at the count actually observed, 30 pairs in this
   session that looked like one neuron become 13. Both baselines are reported: when
   they disagree the correlogram has structure and neither number stands alone. */
function poissonCdf(k, lam){                 /* P(X <= k) for a Poisson mean lam */
  if(!(lam > 0)) return null;
  let term = Math.exp(-lam), s = term;
  for(let i = 1; i <= k; i++){ term *= lam / i; s += term; }
  return Math.min(1, s);
}
function dipStats(p){
  const h = p.ccg_hist, n = h.length, bin = p.ccg_bin_ms, maxLag = p.ccg_max_lag_ms;
  const refr = (MERGE && MERGE.refractory_ms) || 1.5;
  let nRefr = 0, nBins = 0, sNear = 0, cNear = 0, sFar = 0, cFar = 0;
  for(let i = 0; i < n; i++){
    const lag = Math.abs(-maxLag + (i + 0.5) * bin);
    if(lag <= refr){ nRefr += h[i]; nBins++; }
    else if(lag >= SHOULDER_LO && lag <= SHOULDER_HI){ sNear += h[i]; cNear++; }
    if(lag >= FLANK_LO){ sFar += h[i]; cFar++; }
  }
  const near = cNear ? sNear / cNear : 0, far = cFar ? sFar / cFar : 0;
  const expNear = near * nBins, expFar = far * nBins;
  return {
    nRefr: nRefr, nBins: nBins, near: near, far: far,
    expNear: expNear, expFar: expFar,
    ratio: expNear > 0 ? nRefr / expNear : null,
    ratioFar: expFar > 0 ? nRefr / expFar : null,
    depthPct: expNear > 0 ? 100 * (1 - nRefr / expNear) : null,
    p: expNear > 0 ? poissonCdf(nRefr, expNear) : null,
    tilt: (near > 0 && far > 0) ? Math.max(near, far) / Math.min(near, far) : null,
  };
}
function pFmt(v){ return (v === null || v === undefined) ? '--' : (v < 0.001 ? '&lt;0.001' : v.toFixed(3)); }
function nFmt(v){ return (v === null || v === undefined) ? '--' : Number(v).toLocaleString('en-US'); }

/* Both units' mean waveform on every channel either of them covers, each trace drawn
   at the channel's real position on the probe. ONE NEURON HAS ONE FOOTPRINT: same
   column, same depth, falling off the same way in both directions. Two neurons that
   happen to share a peak channel separate here when they separate nowhere else --
   which is why the single-channel overlay was never enough on its own. */
function mFootprint(wa, wb){
  const byCh = {};
  wa.channels.forEach(function(c, i){
    byCh[c] = byCh[c] || {ch: c, um: wa.channel_um[i]}; byCh[c].a = wa.uv[i]; });
  wb.channels.forEach(function(c, i){
    byCh[c] = byCh[c] || {ch: c, um: wb.channel_um[i]}; byCh[c].b = wb.uv[i]; });
  const cells = Object.keys(byCh).map(function(k){ return byCh[k]; });
  if(!cells.length) return '<div class="no-data" style="padding:20px 0">no waveform channels</div>';
  const xs = Array.from(new Set(cells.map(function(c){ return c.um[0]; }))).sort(function(u, v){ return u - v; });
  const ys = Array.from(new Set(cells.map(function(c){ return c.um[1]; }))).sort(function(u, v){ return v - u; });
  let lo = Infinity, hi = -Infinity;
  cells.forEach(function(c){ ['a', 'b'].forEach(function(k){
    if(c[k]) c[k].forEach(function(v){ lo = Math.min(lo, v); hi = Math.max(hi, v); }); }); });
  if(!(hi > lo)){ lo = -1; hi = 1; }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const L = 42, R = 10, T = 16, B = 28;
  const cw = Math.max(46, Math.min(104, 440 / xs.length));
  const ch = Math.max(26, Math.min(54, 300 / ys.length));
  const W = L + R + xs.length * cw, H = T + B + ys.length * ch;
  const X = function(xi, i, n){ return L + xi * cw + 3 + (n > 1 ? (i / (n - 1)) * (cw - 9) : 0); };
  const Y = function(yi, v){ return T + yi * ch + (ch - 3) - ((v - lo) / (hi - lo)) * (ch - 6); };
  let s = '<svg viewBox="0 0 ' + W.toFixed(0) + ' ' + H.toFixed(0) +
          '" role="img" aria-label="mean waveform of both units on every nearby channel">';
  ys.forEach(function(y, yi){
    s += '<text x="' + (L - 5) + '" y="' + (T + yi * ch + ch / 2 + 3).toFixed(1) +
         '" text-anchor="end" font-size="8" fill="var(--text-3)" font-family="monospace">' + y + '</text>';
  });
  cells.forEach(function(c){
    const xi = xs.indexOf(c.um[0]), yi = ys.indexOf(c.um[1]);
    const z = Y(yi, 0);
    s += '<line x1="' + (L + xi * cw + 3).toFixed(1) + '" x2="' + (L + xi * cw + cw - 6).toFixed(1) +
         '" y1="' + z.toFixed(1) + '" y2="' + z.toFixed(1) + '" stroke="var(--border)" stroke-width="0.5"/>';
    ['a', 'b'].forEach(function(k){
      const w = c[k];
      if(!w) return;
      let d = '';
      for(let i = 0; i < w.length; i++){
        d += (i ? 'L' : 'M') + X(xi, i, w.length).toFixed(1) + ' ' + Y(yi, w[i]).toFixed(1);
      }
      s += '<path class="mc-' + k + '" d="' + d + '" style="stroke-width:1.3"/>';
    });
    s += '<text x="' + (L + xi * cw + 3).toFixed(1) + '" y="' + (T + yi * ch + 8).toFixed(1) +
         '" font-size="7" fill="var(--text-3)" font-family="monospace">' + c.ch + '</text>';
  });
  s += '<text x="' + (L - 5) + '" y="' + (T - 5) +
       '" text-anchor="end" font-size="8" fill="var(--text-3)" font-family="monospace">&micro;m</text>';
  s += '<text x="' + L + '" y="' + (H - 15).toFixed(1) +
       '" font-size="8" fill="var(--text-3)" font-family="monospace">' +
       Math.round(lo) + ' to ' + Math.round(hi) + ' &micro;V, one scale on every channel</text>';
  s += '<text x="' + L + '" y="' + (H - 4).toFixed(1) +
       '" font-size="8" fill="var(--text-3)" font-family="monospace">' +
       'each box is one channel at its real position on the probe &middot; depth in &micro;m at left</text>';
  s += '</svg>';
  return s;
}

"""

OLD_DIPCLASS = u"""function dipClass(p){
  const exp = p.ccg.expected_refractory || 0, d = p.ccg.dip_ratio;
  if(d === null || exp < EXP_USABLE) return ['thin', 'too few coincident spikes to read a dip'];
  if(d >= DIP_SYNC) return ['two', 'a PEAK at zero lag: synchrony or one spike detected twice, not one neuron'];
  if(d >= DIP_TWO) return ['two', 'flat through the refractory window: two neurons'];
  if(d <= DIP_ONE) return ['one', 'a real refractory dip: consistent with ONE neuron'];
  return ['partial', 'a partial dip: contaminated, or one neuron plus a neighbour'];
}"""

NEW_DIPCLASS = u"""function dipClass(p){
  const d = dipStats(p);
  if(d.ratio === null || d.expNear < EXP_USABLE)
    return ['thin', 'too few coincident spikes beside the refractory window to read a dip at all'];
  if(d.ratio >= DIP_SYNC)
    return ['two', 'a PEAK at zero lag: synchrony or one spike detected twice, not one neuron'];
  if(d.ratio >= DIP_TWO)
    return ['two', 'flat through the refractory window against its own shoulders: two neurons'];
  if(d.ratio <= DIP_ONE)
    return (d.p !== null && d.p <= DIP_P)
      ? ['one', 'a real refractory dip against its own shoulders, beyond chance at this count: consistent with ONE neuron']
      : ['partial', 'the shape of a dip, but not beyond Poisson chance at this spike count -- not yet evidence'];
  return ['partial', 'a partial dip: contaminated, or one neuron plus a neighbour'];
}"""

OLD_CONSTS = u"const DIP_ONE = 0.45, DIP_TWO = 0.8, DIP_SYNC = 1.5;"
NEW_CONSTS = (u"const DIP_ONE = 0.45, DIP_TWO = 0.8, DIP_SYNC = 1.5;\n"
              u"/* The baseline the refractory window is judged against: the shoulders immediately\n"
              u"   beside it, NOT the far flanks. FLANK_LO is kept only to show the two side by\n"
              u"   side, so a disagreement between them is visible rather than silently decisive.\n"
              u"   DIP_P: a dip must also be unlikely under Poisson chance at the observed count. */\n"
              u"const SHOULDER_LO = 2.0, SHOULDER_HI = 5.0, FLANK_LO = 10.0, DIP_P = 0.01;")

OLD_CCG_CHART = u"""    '<div class="mc-chart"><h5>cross-correlogram #' + p.unit_a + ' vs #' + p.unit_b + '</h5>' +
    mCorrelogram(p.ccg_hist, p.ccg_bin_ms, p.ccg_max_lag_ms, p.ccg.flank_per_bin) +
    '<div class="mc-legend">' + p.ccg.n_refractory + ' observed vs ' +
    mFmt(p.ccg.expected_refractory, 1) + ' expected &mdash; ' + say + '</div></div>' +"""

NEW_CCG_CHART = u"""    '<div class="mc-chart"><h5>cross-correlogram #' + p.unit_a + ' vs #' + p.unit_b + '</h5>' +
    mCorrelogram(p.ccg_hist, p.ccg_bin_ms, p.ccg_max_lag_ms, ds.near) +
    '<div class="mc-legend"><b>' + ds.nRefr + '</b> spikes observed within &plusmn;' +
    mFmt(MERGE.refractory_ms, 1) + ' ms against <b>' + mFmt(ds.expNear, 1) +
    '</b> predicted by its own ' + SHOULDER_LO + '&ndash;' + SHOULDER_HI + ' ms shoulders (' +
    mFmt(ds.near, 2) + '/bin) &rarr; ratio <b>' + mFmt(ds.ratio) + '</b>, dip ' +
    mFmt(ds.depthPct, 0) + '% deep, Poisson p ' + pFmt(ds.p) +
    '<br>for comparison the far flanks (&ge;' + FLANK_LO + ' ms) give ' + mFmt(ds.far, 2) +
    '/bin &rarr; ratio ' + mFmt(ds.ratioFar) +
    (ds.tilt !== null && ds.tilt >= 1.5
      ? ' &mdash; <b>shoulders and flanks disagree ' + mFmt(ds.tilt, 1) + '&times;: this correlogram ' +
        'has broad structure, so the far-flank ratio is not a valid baseline here</b>' : '') +
    '<br>' + say + '</div></div>' +"""

OLD_ACG_CHART = u"""    '<div class="mc-chart"><h5>ACG if merged &middot; ' + mFmt(p.merged_refractory.violation_pct, 3) + '% violations</h5>' +
    mCorrelogram(p.merged_acg_hist, p.ccg_bin_ms, p.ccg_max_lag_ms, null) +
    '<div class="mc-legend">apart: ' + mFmt(p.refractory_a.violation_pct, 3) + '% / ' +
    mFmt(p.refractory_b.violation_pct, 3) + '% &mdash; if merging fills the gap in, it is not one neuron</div></div>' +"""

NEW_ACG_CHART = u"""    '<div class="mc-chart"><h5>ACG if merged &middot; ' + mFmt(p.merged_refractory.violation_pct, 3) + '% violations</h5>' +
    '<div class="mc-legend" style="margin:0 0 5px">#' + p.unit_a + ' <b>' + mFmt(a.mean_rate_hz, 2) +
    ' Hz</b> (' + nFmt(p.refractory_a.n) + ' spikes) &middot; #' + p.unit_b + ' <b>' +
    mFmt(b.mean_rate_hz, 2) + ' Hz</b> (' + nFmt(p.refractory_b.n) + ' spikes)' +
    (rateRatio === null ? '' :
      (rateRatio >= RATE_LOPSIDED
        ? ' &mdash; <b>' + mFmt(rateRatio, 1) + ':1. This ACG is almost entirely #' + fastUnit +
          ', so neither its shape nor its violation % carries much weight for this merge</b>'
        : ' &middot; ' + mFmt(rateRatio, 1) + ':1 rate ratio')) +
    '</div>' +
    mCorrelogram(p.merged_acg_hist, p.ccg_bin_ms, p.ccg_max_lag_ms, null) +
    '<div class="mc-legend">apart: ' + mFmt(p.refractory_a.violation_pct, 3) + '% / ' +
    mFmt(p.refractory_b.violation_pct, 3) + '% &middot; merging creates <b>~' + nFmt(newViol) +
    '</b> refractory violations neither unit had on its own &mdash; a <i>count</i>, which an ' +
    'uneven rate cannot dilute the way it dilutes the percentage</div></div>' +"""

OLD_WF_CHART = u"""    '<div class="mc-chart"><h5>mean raw waveforms &middot; ' + chTxt + '</h5>' +
    mWaveforms(a.waveform, b.waveform, ia, ib) +
    '<div class="mc-legend">#' + p.unit_a + ' solid, #' + p.unit_b + ' dashed &middot; amplitude ratio ' +
    mFmt(p.amplitude_ratio) + ' &middot; KS template similarity ' + mFmt(p.template_similarity, 3) + '</div></div>' +"""

NEW_WF_CHART = u"""    '<div class="mc-chart"><h5>mean raw waveforms &middot; ' + chTxt + '</h5>' +
    mWaveforms(a.waveform, b.waveform, ia, ib) +
    '<div class="mc-legend">#' + p.unit_a + ' solid, #' + p.unit_b + ' dashed &middot; amplitude ratio ' +
    mFmt(p.amplitude_ratio) + ' &middot; KS template similarity ' + mFmt(p.template_similarity, 3) + '</div></div>' +
    '<div class="mc-chart mc-wide"><h5>both waveforms across the channels &mdash; the footprint</h5>' +
    mFootprint(a.waveform, b.waveform) +
    '<div class="mc-legend">#' + p.unit_a + ' solid, #' + p.unit_b + ' dashed, in &micro;V read from ' +
    'probe1.dat &middot; waveform correlation on the shared channels ' + mFmt(p.waveform_corr) +
    ' &middot; peak channels ' + a.peak_channel + ' and ' + b.peak_channel + ', ' +
    mFmt(Math.abs(a.depth_um - b.depth_um), 0) + ' &micro;m apart in depth. One neuron has ONE ' +
    'footprint: same column, same depth, decaying the same way in both directions.</div></div>' +"""

OLD_BODY_HEAD = u"""function mergeBody(p, uid, other, say){
  const a = MERGE.units[String(p.unit_a)], b = MERGE.units[String(p.unit_b)];"""

NEW_BODY_HEAD = u"""function mergeBody(p, uid, other, say){
  const a = MERGE.units[String(p.unit_a)], b = MERGE.units[String(p.unit_b)];
  const ds = dipStats(p);
  /* Rate asymmetry, stated rather than left for you to notice: when one unit fires
     far faster than the other, the merged train is mostly that unit, so the merged
     ACG and its violation PERCENTAGE are both nearly unchanged by the merge however
     wrong it is. The count of violations the interleaving creates is not. */
  const rHi = Math.max(a.mean_rate_hz, b.mean_rate_hz);
  const rLo = Math.min(a.mean_rate_hz, b.mean_rate_hz);
  const rateRatio = rLo > 0 ? rHi / rLo : null;
  const fastUnit = a.mean_rate_hz >= b.mean_rate_hz ? p.unit_a : p.unit_b;
  const vOf = function(r){ return ((r.violation_pct || 0) / 100) * Math.max(0, (r.n || 0) - 1); };
  const newViol = Math.max(0, Math.round(
    vOf(p.merged_refractory) - vOf(p.refractory_a) - vOf(p.refractory_b)));"""

OLD_TAG = u"""      '<span class="tag ' + cls + '">' + (p.ccg.dip_ratio === null ? 'no dip' : 'dip ' + mFmt(p.ccg.dip_ratio)) + '</span>' +"""
NEW_TAG = u"""      '<span class="tag ' + cls + '">' + (dsRow.ratio === null ? 'no dip' : 'dip ' + mFmt(dsRow.ratio)) + '</span>' +"""

OLD_ROW_HEAD = u"""    const [cls, say] = dipClass(p);"""
NEW_ROW_HEAD = u"""    const [cls, say] = dipClass(p);
    const dsRow = dipStats(p);"""

OLD_WHY = u"""    'cross-correlogram: one neuron cannot fire twice inside its refractory period, so if two ' +
    'clusters are one neuron the &plusmn;1.5 ms window is empty. <b>dip</b> is what is observed ' +
    'there over what the correlogram&rsquo;s own flanks predict, and it is only a measurement ' +
    'when enough coincident spikes exist &mdash; which is why pairs that cannot be read say so ' +
    'rather than showing a flattering 0.00. Deciding here changes nothing by itself: export, ' +
    'run write_unit_merges.py, then re-run the pipeline.</div>';"""

NEW_WHY = u"""    'cross-correlogram: one neuron cannot fire twice inside its refractory period, so if two ' +
    'clusters are one neuron the &plusmn;1.5 ms window is empty. <b>dip</b> is what is observed ' +
    'there over what the <b>' + SHOULDER_LO + '&ndash;' + SHOULDER_HI + ' ms shoulders beside it</b> ' +
    'predict &mdash; deliberately not the far flanks, which any broad correlogram structure ' +
    'makes an invalid baseline. A pair is counted as <b>worth a look</b> when either the dip is ' +
    '&le;' + DIP_ONE + ' against those shoulders AND unlikely under Poisson chance (p&le;' + DIP_P +
    ') at the count observed, or the two units trade places across the session (rate correlation ' +
    '&le;&minus;0.1 with simultaneous-activity overlap &le;0.35), which is a drift pair the ' +
    'correlogram cannot see. Everything else stays one click away rather than crowding those out. ' +
    'Deciding here changes nothing by itself: export, run write_unit_merges.py, then re-run the ' +
    'pipeline.</div>';"""

OLD_STAMP = u"""    dip_ratio: p.ccg.dip_ratio, waveform_corr: p.waveform_corr,"""
NEW_STAMP = u"""    /* both metrics stored: the shoulder ratio the decision was shown, and the old
       far-flank one, so a verdict can be re-read against either */
    dip_ratio: dipStats(p).ratio, dip_ratio_far_flank: p.ccg.dip_ratio,
    dip_poisson_p: dipStats(p).p, waveform_corr: p.waveform_corr,"""

OLD_EMPTY = u"""      'evidence a merge needs: no readable refractory dip, no drift-pair signature. ' +"""
NEW_EMPTY = u"""      'evidence a merge needs: no refractory dip that survives its own shoulders and a ' +
      'Poisson test, no drift-pair signature. ' +"""

SUBS = [
    (OLD_CONSTS, NEW_CONSTS, "dip constants + shoulder/Poisson thresholds"),
    (u"function dipClass(p){", HELPERS.lstrip(u"\n") + u"function dipClass(p){",
     "poissonCdf / dipStats / mFootprint helpers"),
    (OLD_DIPCLASS, NEW_DIPCLASS, "dipClass on shoulders + Poisson"),
    (OLD_ROW_HEAD, NEW_ROW_HEAD, "row-level dipStats"),
    (OLD_TAG, NEW_TAG, "collapsed-row dip tag uses the shoulder ratio"),
    (OLD_BODY_HEAD, NEW_BODY_HEAD, "mergeBody: dipStats, rate ratio, new-violation count"),
    (OLD_CCG_CHART, NEW_CCG_CHART, "CCG legend: shoulders, Poisson p, flank disagreement"),
    (OLD_ACG_CHART, NEW_ACG_CHART, "merged ACG: firing rates above it + violation count"),
    (OLD_WF_CHART, NEW_WF_CHART, "across-channel footprint chart"),
    (OLD_WHY, NEW_WHY, "panel blurb states 'worth a look' in full"),
    (OLD_EMPTY, NEW_EMPTY, "empty-list wording"),
    (OLD_STAMP, NEW_STAMP, "verdict stamps both dip metrics"),
]


def main():
    s = io.open(SRC, encoding="utf-8").read()
    print("read  %s (%d bytes)" % (SRC, len(s)))
    for old, new, label in SUBS:
        n = s.count(old)
        if n != 1:
            raise SystemExit("ANCHOR %r matched %d times (need exactly 1)" % (label, n))
        s = s.replace(old, new)
        print("  ok  %s" % label)
    d = os.path.dirname(DST)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    io.open(DST, "w", encoding="utf-8").write(s)
    print("wrote %s (%d bytes)" % (DST, os.path.getsize(DST)))
    for must in ("mFootprint", "poissonCdf", "SHOULDER_LO", "dip_ratio_far_flank"):
        print("  contains %-20s %s" % (must, s.count(must)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

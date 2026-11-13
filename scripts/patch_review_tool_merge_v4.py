# -*- coding: utf-8 -*-
"""Fourth pass over the merge panel: selection is rebuilt around Gil's rule.

Applied after v1 (the panel), v2 (the shoulder metric) and v3 (twins, navigation).

WHY
---
v1-v3 selected a pair with `isWorthLook = dip OR drift OR twin`. Measured on
20261112_082028 with every co-located pair analysed (2728 of them, after the
--max_pairs cap was lifted), that rule showed 254 pairs of which only 22 rested on
the correlogram at all. Worse, the drift route does not merely ignore the
correlogram, it overrides it: 154 of its 220 pairs have correlograms that
positively say two neurons.

Gil's objection, and it is the right one: drift is not evidence that two clusters
are one neuron. It is an explanation for why the correlogram cannot speak -- two
units that never fire in the same period cannot produce coincidences. The same
holds for the footprint: if the probe moved, one neuron's two clusters will sit on
different channels and decay differently, so drift also excuses a footprint
mismatch. An explanation for missing evidence is not itself evidence, so it cannot
be an entry route.

And measured, the two never collide anyway: NOT ONE of the 22 dip pairs has a drift
signature (the most negative rate_corr among them is -0.095), exactly as the physics
predicts.

THE RULE NOW
------------
    shape agrees  =  template similarity >= SIM_FLOOR
                     AND ( footprint agrees  OR  strict drift )

    TIER A  shape agrees AND the correlogram shows a real dip      -> confirmed
    TIER B  shape agrees AND the correlogram cannot settle it      -> needs a human
    rejected: everything else, each with the reason it failed

Template similarity is a REJECTOR, never a selector. High similarity proves nothing
-- the 99th-percentile pair in this session is 0.993 with waveform r 0.976 and a
flat correlogram -- but low similarity does reject, because two clusters of one
neuron must look alike unless the probe moved. It is also the measure Kilosort
itself fits, so distrusting it at the merge stage would make everything mergeable.

Waveform correlation is DISPLAYED, never gated on. Measured, it only disagrees with
template similarity in the middle of its range, where it carries no information: of
the 22 dip pairs, 359+361 has waveform r 0.224 with similarity 0.759 (91st pctl)
while 143+145 has waveform r 0.193 with similarity 0.129 (13th pctl) -- the same
near-zero correlation, opposite verdicts. A near-zero r is the transition between
high positive and high negative, not a finding. Where similarity is high the raw
waveform agrees anyway: all five pairs at similarity >= 0.8 have waveform r between
0.64 and 0.97.

THE FOOTPRINT SCORE (new, computed in the page)
-----------------------------------------------
One neuron has one footprint: same centre, same fall-off in both directions. Both
numbers come from the per-unit mean waveform the payload already carries (10
channels with real micron positions, 60 samples each), so nothing is recomputed
from raw data:

    offset_um   distance between the two units' amplitude-weighted centroids
    profile_r   correlation of their per-channel amplitude profiles, each
                normalised to its own maximum, over the channels they share

Measured on this session: profile_r has median -0.123 across all 2728 pairs but
0.607 across the 22 dip pairs, so it separates. Note what it does to the four
pairs whose mean waveforms are INVERTED (waveform r -0.77 to -0.91): their
footprints match almost perfectly (profile_r 0.72-0.98, offsets 2.9-14.2 um). Same
spatial extent, opposite polarity -- which is a dipole signature, possibly one
neuron split across opposite-sign features. Two of the four fail the similarity
floor and drop out; 309+312 and 180+183 survive into Tier A with their negative
waveform r on show, which is the right place for them: flagged, not silently
dropped.

THRESHOLDS ARE TUNABLE IN THE PANEL
-----------------------------------
Every number above was measured, not guessed, but none of them is settled, so the
panel exposes all five as inputs with the live counts beside them. Defaults, and
what they give on this session:

    SIM_FLOOR      0.50   (72nd percentile of all pairs)
    FP_PROFILE_MIN 0.30
    FP_OFFSET_MAX  35 um
    DRIFT_R       -0.20   (was -0.10: 1% shared variance, far too loose for a
                           rule that now excuses two separate rejections)
    DRIFT_OVERLAP  0.15   (was 0.35)

    -> Tier A 13, Tier B 202, shown 215, rejected 2513

Nine of the 22 dips are rejected, each for a stated reason: 143+145, 277+283,
139+152, 228+242 on similarity and footprint both; 438+439 and 284+286 on
similarity alone; 267+270, 348+350 and 420+423 on footprint with no drift to
excuse it.
"""
import io
import os
import sys

SRC = (sys.argv[1] if len(sys.argv) > 1 else
       r"D:\Gil\spike_sorting_agent\outputs\unit_review_tool_20261112_082028.html")
DST = (sys.argv[2] if len(sys.argv) > 2 else
       r"D:\Gil\spike_sorting_agent\outputs\unit_review_tool_20261112_082028.html")


# --------------------------------------------------------------- 1. the helpers
OLD_CONSTS = u"const SIM_TWIN = 0.95, WFR_TWIN = 0.85;"

NEW_CONSTS = u"""const SIM_TWIN = 0.95, WFR_TWIN = 0.85;

/* --- the footprint ---------------------------------------------------------
   One neuron has ONE footprint: the same centre on the probe and the same
   fall-off in both directions. Both numbers below come from the mean waveform
   already in the payload -- 10 channels per unit at their real micron positions
   -- so nothing is recomputed from raw data.

     offset_um   distance between the two amplitude-weighted centroids
     profile_r   correlation of the per-channel amplitude profiles, each
                 normalised to its own maximum, over the shared channels

   Measured on this session: profile_r runs at median -0.12 across all pairs and
   +0.61 across the pairs with a real dip, so it separates. It is deliberately
   built from AMPLITUDE per channel, not from waveform shape, so a pair whose
   mean waveforms are inverted still scores high if it occupies the same place --
   which is what the four inverted pairs here do. */
const _fpCache = {};
function _ampProfile(w){
  const out = [];
  for(let i = 0; i < w.uv.length; i++){
    const tr = w.uv[i];
    let lo = tr[0], hi = tr[0];
    for(let k = 1; k < tr.length; k++){
      if(tr[k] < lo) lo = tr[k];
      if(tr[k] > hi) hi = tr[k];
    }
    out.push(hi - lo);
  }
  return out;
}
function _pearson(a, b){
  const n = a.length;
  if(n < 3) return null;
  let ma = 0, mb = 0;
  for(let i = 0; i < n; i++){ ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let sa = 0, sb = 0, sab = 0;
  for(let i = 0; i < n; i++){
    const da = a[i] - ma, db = b[i] - mb;
    sa += da * da; sb += db * db; sab += da * db;
  }
  if(!(sa > 0) || !(sb > 0)) return null;
  return sab / Math.sqrt(sa * sb);
}
function footprintStats(p){
  if(_fpCache[p.pair_id]) return _fpCache[p.pair_id];
  const res = {offset: null, profile_r: null};
  const ua = MERGE.units[String(p.unit_a)], ub = MERGE.units[String(p.unit_b)];
  if(ua && ub && ua.waveform && ub.waveform){
    const wa = ua.waveform, wb = ub.waveform;
    const aa = _ampProfile(wa), ab = _ampProfile(wb);
    let sa = 0, sb = 0;
    for(let i = 0; i < aa.length; i++) sa += aa[i];
    for(let i = 0; i < ab.length; i++) sb += ab[i];
    if(sa > 0 && sb > 0){
      let ax = 0, ay = 0, bx = 0, by = 0;
      for(let i = 0; i < aa.length; i++){
        ax += wa.channel_um[i][0] * aa[i]; ay += wa.channel_um[i][1] * aa[i];
      }
      for(let i = 0; i < ab.length; i++){
        bx += wb.channel_um[i][0] * ab[i]; by += wb.channel_um[i][1] * ab[i];
      }
      res.offset = Math.hypot(ax / sa - bx / sb, ay / sa - by / sb);
      const shared = wa.channels.filter(c => wb.channels.indexOf(c) >= 0);
      if(shared.length >= 4){
        const va = shared.map(c => aa[wa.channels.indexOf(c)]);
        const vb = shared.map(c => ab[wb.channels.indexOf(c)]);
        const mxa = Math.max.apply(null, va), mxb = Math.max.apply(null, vb);
        if(mxa > 0 && mxb > 0)
          res.profile_r = _pearson(va.map(v => v / mxa), vb.map(v => v / mxb));
      }
    }
  }
  _fpCache[p.pair_id] = res;
  return res;
}

/* --- the selection rule, and it is tunable --------------------------------
   Template similarity REJECTS, it never selects: a high value proves nothing
   (this session's most similar pair, 0.993 with waveform r 0.976, has a flat
   correlogram), but a low value does reject, because two clusters of one neuron
   have to look alike unless the probe moved. Drift is not an entry route -- it is
   the EXCUSE for evidence that cannot be gathered, so it only waives the
   footprint test. Waveform r is shown and never gated on.

   Every threshold was measured rather than guessed, and none is settled, so all
   five are inputs on the panel with live counts beside them. */
const SEL_DEFAULT = {sim: 0.50, fpr: 0.30, fpoff: 35, driftR: -0.20, driftOv: 0.15};
let SEL = Object.assign({}, SEL_DEFAULT);
try {
  const raw = localStorage.getItem('mergeSel');
  if(raw){
    const o = JSON.parse(raw);
    ['sim','fpr','fpoff','driftR','driftOv'].forEach(function(k){
      if(typeof o[k] === 'number' && isFinite(o[k])) SEL[k] = o[k];
    });
  }
} catch(e) {}
function selSave(){
  try { localStorage.setItem('mergeSel', JSON.stringify(SEL)); } catch(e) {}
}

function simAgrees(p){ return (p.template_similarity || 0) >= SEL.sim; }
function isDriftStrict(p){
  return p.rate_corr !== null && p.rate_corr <= SEL.driftR &&
         p.active_overlap !== null && p.active_overlap <= SEL.driftOv;
}
function fpAgrees(p){
  const f = footprintStats(p);
  return f.profile_r !== null && f.profile_r >= SEL.fpr &&
         f.offset !== null && f.offset <= SEL.fpoff;
}
function shapeAgrees(p){ return simAgrees(p) && (fpAgrees(p) || isDriftStrict(p)); }

/* Tier A: the shape matches AND the correlogram shows a real dip.
   Tier B: the shape matches and the correlogram cannot settle it. */
function tierOf(p){
  if(!shapeAgrees(p)) return null;
  const c = dipClass(p)[0];
  if(c === 'one') return 'A';
  if(c === 'thin' || c === 'partial') return 'B';
  return null;                       /* flat or peaked: the correlogram said no */
}
function rejectReason(p){
  const c = dipClass(p)[0];
  if(!simAgrees(p))
    return 'template similarity ' + mFmt(p.template_similarity) + ' below ' + SEL.sim;
  if(!fpAgrees(p) && !isDriftStrict(p)){
    const f = footprintStats(p);
    return 'footprint mismatch (profile r ' + mFmt(f.profile_r) + ', centres ' +
           mFmt(f.offset, 0) + ' &micro;m apart) with no drift to explain it';
  }
  if(c === 'peak') return 'a peak at zero lag: synchrony, not one neuron';
  if(c === 'flat') return 'correlogram flat through the refractory window: two neurons';
  return 'does not meet the rule';
}
/* Rank Tier A by how much the dip rests on real counts, not by its depth alone:
   a dip of 0.43 measured against 27992 expected spikes is far better evidence
   than 0.125 measured against 8. */
function dipConfidence(p){
  const d = dipStats(p);
  if(d.ratio === null || !(d.expNear > 0)) return -1;
  return (1 - Math.min(d.ratio, 1)) * Math.log10(Math.max(d.expNear, 1.0001));
}"""


# ------------------------------------------- 2. isWorthLook rides on the tiers
OLD_WORTH = (u"function isWorthLook(p){ return dipClass(p)[0] === 'one' || "
             u"isDrift(p) || isTwin(p); }")
NEW_WORTH = (u"function isWorthLook(p){ return tierOf(p) !== null; }")


# ------------------------------------------------- 3. the bucketing + the head
OLD_BUCKETS_HEAD = u"""const all = (mergeByUnit[uid] || []).slice().sort((x, y) => {
    const dx = x.ccg.dip_ratio === null ? 9 : x.ccg.dip_ratio;
    const dy = y.ccg.dip_ratio === null ? 9 : y.ccg.dip_ratio;
    return dx - dy;
  });
  const strong = all.filter(isWorthLook);
  const weak = all.filter(p => !isWorthLook(p) && !mergeVerdicts[p.pair_id]);
  const decidedWeak = all.filter(p => !isWorthLook(p) && mergeVerdicts[p.pair_id]);
  const showAll = mergeShowAll[uid];
  /* decided pairs always stay visible, however weak -- a decision you made is not
     something the page should hide from you */
  const list = showAll ? all : strong.concat(decidedWeak);"""

NEW_BUCKETS_HEAD = u"""const all = (mergeByUnit[uid] || []).slice();
  /* Tier A ranked by how much the dip rests on real counts; Tier B has no dip to
     rank by, so it falls back to template similarity. */
  const tierA = all.filter(p => tierOf(p) === 'A')
                   .sort((x, y) => dipConfidence(y) - dipConfidence(x));
  const tierB = all.filter(p => tierOf(p) === 'B')
                   .sort((x, y) => (y.template_similarity || 0) - (x.template_similarity || 0));
  const strong = tierA.concat(tierB);
  const weak = all.filter(p => tierOf(p) === null && !mergeVerdicts[p.pair_id])
                  .sort((x, y) => (y.template_similarity || 0) - (x.template_similarity || 0));
  const decidedWeak = all.filter(p => tierOf(p) === null && mergeVerdicts[p.pair_id]);
  const showAll = mergeShowAll[uid];
  /* decided pairs always stay visible, however weak -- a decision you made is not
     something the page should hide from you */
  const list = showAll ? tierA.concat(tierB, decidedWeak, weak)
                       : tierA.concat(tierB, decidedWeak);
  /* section headers, emitted before the first row of each group */
  const sectionOf = {};
  if(tierA.length) sectionOf[tierA[0].pair_id] =
    '<div class="msec"><b>Confirmed</b> &mdash; shape agrees and the correlogram shows a ' +
    'real refractory dip. ' + tierA.length + ' pair' + (tierA.length === 1 ? '' : 's') +
    ', strongest counts first.</div>';
  if(tierB.length) sectionOf[tierB[0].pair_id] =
    '<div class="msec"><b>Shape agrees, correlogram cannot settle it</b> &mdash; too few ' +
    'coincident spikes for a verdict, so these need your eye. ' + tierB.length +
    ' pair' + (tierB.length === 1 ? '' : 's') + ', most similar templates first.</div>';
  if(decidedWeak.length) sectionOf[decidedWeak[0].pair_id] =
    '<div class="msec"><b>Decided by you</b> &mdash; kept visible whatever the rule says ' +
    'about them.</div>';
  if(showAll && weak.length) sectionOf[weak[0].pair_id] =
    '<div class="msec"><b>Rejected</b> &mdash; each row says which test it failed.</div>';"""


OLD_HEADLINE = u"""let h = '<div class="merge-head"><div class="panel-label">Possible merges &middot; ' +
    (strong.length ? strong.length + ' worth a look' : 'none convincing') +
    ' of ' + all.length + ' neighbour' + (all.length === 1 ? '' : 's') + ' for unit #' + uid + '</div>' +
    '<button class="merge-export" id="mergeExportBtn">export ' + nDecided + ' merge' +
    (nDecided === 1 ? '' : 's') + '</button></div>';"""

NEW_HEADLINE = u"""let h = '<div class="merge-head"><div class="panel-label">Possible merges &middot; ' +
    (tierA.length ? tierA.length + ' confirmed' : 'none confirmed') +
    (tierB.length ? ', ' + tierB.length + ' undecided' : '') +
    ' of ' + all.length + ' neighbour' + (all.length === 1 ? '' : 's') + ' for unit #' + uid + '</div>' +
    '<button class="merge-export" id="mergeExportBtn">export ' + nDecided + ' merge' +
    (nDecided === 1 ? '' : 's') + '</button></div>';

  /* the rule itself, open to the hand. Counts are over every co-located pair in
     the session, not just this unit's, so moving a threshold shows its real cost. */
  let gA = 0, gB = 0, gRej = 0;
  MERGE.pairs.forEach(function(q){
    const t = tierOf(q);
    if(t === 'A') gA++; else if(t === 'B') gB++; else gRej++;
  });
  h += '<div class="msel">' +
    '<div class="msel-row"><span class="msel-lab">template similarity &ge;</span>' +
    '<input class="msel-in" id="selSim" type="number" step="0.05" min="0" max="1" value="' + SEL.sim + '">' +
    '<span class="msel-note">rejects a pair outright &mdash; it never promotes one</span></div>' +
    '<div class="msel-row"><span class="msel-lab">footprint profile r &ge;</span>' +
    '<input class="msel-in" id="selFpr" type="number" step="0.05" min="-1" max="1" value="' + SEL.fpr + '">' +
    '<span class="msel-lab">centres &le;</span>' +
    '<input class="msel-in" id="selFpoff" type="number" step="5" min="0" max="60" value="' + SEL.fpoff + '">' +
    '<span class="msel-note">&micro;m apart</span></div>' +
    '<div class="msel-row"><span class="msel-lab">drift: rate r &le;</span>' +
    '<input class="msel-in" id="selDriftR" type="number" step="0.05" min="-1" max="0" value="' + SEL.driftR + '">' +
    '<span class="msel-lab">overlap &le;</span>' +
    '<input class="msel-in" id="selDriftOv" type="number" step="0.05" min="0" max="1" value="' + SEL.driftOv + '">' +
    '<span class="msel-note">waives the footprint test only, never the similarity floor</span></div>' +
    '<div class="msel-tot">across all ' + MERGE.pairs.length + ' co-located pairs in this session: ' +
    '<b>' + gA + '</b> confirmed &middot; <b>' + gB + '</b> undecided &middot; ' +
    gRej + ' rejected &nbsp;<button class="msel-reset" id="selReset">reset</button></div>' +
    '</div>';"""


OLD_EMPTY = u"""if(!list.length){
    h += '<div class="merge-why">' + all.length + ' cluster' + (all.length === 1 ? ' is' : 's are') +
      ' within ' + MERGE.max_dist_um + ' &micro;m of this one, but none of them shows the ' +
      'evidence a merge needs: no refractory dip that survives its own shoulders and a ' +
      'Poisson test, no drift-pair signature. ' +
      'That is a result, not an empty list &mdash; on a dense probe most neighbours are ' +
      'simply other neurons.</div>';
  }"""

NEW_EMPTY = u"""if(!list.length){
    h += '<div class="merge-why">' + all.length + ' cluster' + (all.length === 1 ? ' is' : 's are') +
      ' within ' + MERGE.max_dist_um + ' &micro;m of this one, and every one of them fails ' +
      'the rule &mdash; the reasons are on each row below the button. That is a result, not an ' +
      'empty list: on a dense probe most neighbours are simply other neurons.</div>';
  }"""


OLD_BLURB = u"""'Everything else stays one click away rather than crowding those out. ' +
    'Deciding here changes nothing by itself: export, run write_unit_merges.py, then re-run the ' +
    'pipeline.</div>';"""

NEW_BLURB = u"""'Everything else stays one click away rather than crowding those out. ' +
    'Deciding here changes nothing by itself: export, run write_unit_merges.py, then re-run the ' +
    'pipeline.</div>' +
    '<div class="merge-why"><b>How a pair gets here.</b> The shape has to agree first: ' +
    'KS template similarity &ge;<b>' + SEL.sim + '</b> and a matching footprint &mdash; one ' +
    'neuron has one footprint, so the two amplitude centroids must sit within <b>' + SEL.fpoff +
    ' &micro;m</b> and the per-channel amplitude profiles must correlate at &ge;<b>' + SEL.fpr +
    '</b>. Then the correlogram decides: a dip confirms, flat or peaked rejects, too few ' +
    'coincidences leaves it to you. Similarity is a <b>rejector only</b> &mdash; the most ' +
    'similar pair in this session (0.993, waveform r 0.976) has a flat correlogram and is ' +
    'certainly two neurons &mdash; but a low value does reject, since two clusters of one neuron ' +
    'must look alike unless the probe moved. <b>Drift is not a route in.</b> It explains away ' +
    'evidence that could not be gathered: units that never fire in the same period cannot ' +
    'produce coincidences, and a probe that moved leaves one neuron with two different ' +
    'footprints. So it waives the footprint test, never the similarity floor. Measured here, ' +
    'not one of the pairs with a readable dip has a drift signature at all. Waveform r is ' +
    'shown on every row and gated on nowhere: near zero it is the crossing point between ' +
    'high positive and high negative and carries no information, and where similarity is high ' +
    'it agrees anyway.</div>';"""


# --------------------------------------- 4. emit the section header before a row
OLD_FOREACH = u"""list.forEach(p => {
    const other = partnerOf(p, uid);"""
NEW_FOREACH = u"""list.forEach(p => {
    if(sectionOf[p.pair_id]) h += sectionOf[p.pair_id];
    const other = partnerOf(p, uid);"""


# ----------------------------------------------------- 5. the show-all wording
OLD_SHOWALL = u"""      (showAll ? 'hide the ' + weak.length + ' unconvincing neighbour' + (weak.length === 1 ? '' : 's')
               : 'show the other ' + weak.length + ' neighbour' + (weak.length === 1 ? '' : 's') +
                 ' (flat correlograms)') + '</button>';"""
NEW_SHOWALL = u"""      (showAll ? 'hide the ' + weak.length + ' rejected neighbour' + (weak.length === 1 ? '' : 's')
               : 'show the ' + weak.length + ' rejected neighbour' + (weak.length === 1 ? '' : 's') +
                 ', with the reason each failed') + '</button>';"""


# -------------------------------------------------- 6. wire the threshold inputs
OLD_HANDLER = u"const allBtn = container.querySelector('#mergeShowAllBtn');"
NEW_HANDLER = u"""[['selSim','sim'],['selFpr','fpr'],['selFpoff','fpoff'],
   ['selDriftR','driftR'],['selDriftOv','driftOv']].forEach(function(kv){
    const el = container.querySelector('#' + kv[0]);
    if(!el) return;
    el.addEventListener('change', function(){
      const v = parseFloat(el.value);
      if(isFinite(v)){ SEL[kv[1]] = v; selSave(); renderDetail(uid); }
    });
  });
  const selRst = container.querySelector('#selReset');
  if(selRst) selRst.addEventListener('click', function(){
    SEL = Object.assign({}, SEL_DEFAULT); selSave(); renderDetail(uid);
  });
  const allBtn = container.querySelector('#mergeShowAllBtn');"""


# ------------------------------------------------- 7. the reason on a weak row
OLD_ROWTAGS = u"""      (isDrift(p) ? '<span class="tag drift">drift pair?</span>' : '') +
      (isTwin(p) ? '<span class="tag twin">twin templates</span>' : '') +"""
NEW_ROWTAGS = u"""      (isDriftStrict(p) ? '<span class="tag drift">drift pair</span>' : '') +
      (tierOf(p) === null ? '<span class="tag rej">rejected</span>' : '') +"""


# --------------------------------- 8. a rejected row says which test it failed
OLD_ROWTAIL = (u"""'<div class="mc-body">' + (open ? mergeBody(p, uid, other, say) : '')"""
               u""" + '</div></div>';""")
NEW_ROWTAIL = (u"""'<div class="mc-body">' + (open ? mergeBody(p, uid, other, say) : '')"""
               u""" + '</div>' +
      (tierOf(p) === null ? '<div class="mrej">rejected &mdash; ' + rejectReason(p) +
        '</div>' : '') + '</div>';""")


# -------------------- 9. the expanded body describes drift at the NEW threshold
OLD_BODY_DRIFT = (u"""(isDrift(p) ? ' &mdash; <b>they trade places: a drift pair, """
                  u"""which the correlogram cannot see</b>' : '') +""")
NEW_BODY_DRIFT = (u"""(isDriftStrict(p)
      ? ' &mdash; <b>they trade places: a drift pair, so neither the correlogram nor ' +
        'the footprint can be held against them</b>'
      : (isDrift(p)
          ? ' &mdash; they lean apart, but not far enough to count as drift at rate r &le;' +
            SEL.driftR + ' and overlap &le;' + SEL.driftOv
          : '')) +""")


# ------------------------------------------------------------------- 10. styles
OLD_CSS = u".tag.twin{"
NEW_CSS = u""".msec{font-size:10.5px;color:var(--text-2);background:var(--surface-2);
  border-left:2px solid var(--accent);padding:6px 9px;margin:14px 0 8px;line-height:1.5;}
.msec b{color:var(--text);}
.msel{border:1px solid var(--border);border-radius:5px;padding:8px 10px;margin:0 0 10px;
  background:var(--surface);}
.msel-row{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-bottom:5px;}
.msel-lab{font-size:10.5px;color:var(--text-2);}
.msel-note{font-size:10px;color:var(--text-3);}
.msel-in{width:68px;font:inherit;font-size:11px;padding:2px 5px;border-radius:4px;
  border:1px solid var(--border);background:var(--surface-2);color:var(--text);}
.msel-tot{font-size:10.5px;color:var(--text-2);border-top:1px solid var(--border);
  padding-top:6px;margin-top:3px;}
.msel-tot b{color:var(--text);}
.msel-reset{font:inherit;font-size:10px;padding:1px 7px;border-radius:4px;cursor:pointer;
  border:1px solid var(--border);background:var(--surface-2);color:var(--text-2);}
.tag.rej{background:var(--surface-2);color:var(--text-3);}
.mrej{font-size:10px;color:var(--text-3);padding:0 9px 7px;line-height:1.45;}
.tag.twin{"""


SUBS = [
    (OLD_CONSTS, NEW_CONSTS, "footprint score + the tunable selection rule"),
    (OLD_WORTH, NEW_WORTH, "isWorthLook rides on the tiers"),
    (OLD_BUCKETS_HEAD, NEW_BUCKETS_HEAD, "two ranked tiers instead of one OR'd list"),
    (OLD_HEADLINE, NEW_HEADLINE, "headline counts + the threshold controls"),
    (OLD_EMPTY, NEW_EMPTY, "empty-state wording"),
    (OLD_BLURB, NEW_BLURB, "the blurb states the new rule in full"),
    (OLD_FOREACH, NEW_FOREACH, "section headers between the tiers"),
    (OLD_SHOWALL, NEW_SHOWALL, "show-all button names the reasons"),
    (OLD_HANDLER, NEW_HANDLER, "threshold inputs wired to a re-render"),
    (OLD_ROWTAGS, NEW_ROWTAGS, "row tags: strict drift, rejected"),
    (OLD_ROWTAIL, NEW_ROWTAIL, "a rejected row states which test it failed"),
    (OLD_BODY_DRIFT, NEW_BODY_DRIFT, "body text uses the strict drift threshold"),
    (OLD_CSS, NEW_CSS, "styles for the tier headers and the rule controls"),
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
    for must in ("footprintStats", "tierOf", "dipConfidence", "SEL_DEFAULT",
                 "sectionOf", "msel-row", "rejectReason"):
        if must not in s:
            raise SystemExit("expected %r in the result and it is missing" % must)
    d = os.path.dirname(DST)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    io.open(DST, "w", encoding="utf-8").write(s)
    print("wrote %s (%d bytes)" % (DST, os.path.getsize(DST)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

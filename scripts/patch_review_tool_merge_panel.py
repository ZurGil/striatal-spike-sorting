# -*- coding: utf-8 -*-
"""Splice a per-unit "Possible merges" section into the review tool.

Built against published version 1792083808-c26b of
https://claude.ai/code/artifact/74dec064-c86a-46a3-8c03-f112ee4ecb05 and published as
version 1792157641-801f. To change the panel: read the CURRENT published page with the
Artifact tool (it saves the full HTML locally), point SRC at that file, edit the CSS/JS
below, re-run, and republish the result to the same URL together with
outputs/merge_candidates_<session>.json as the file data/merge_candidates.json.

The seven anchors at the bottom are deliberately exact strings: if the review tool is
edited by hand, a moved anchor fails loudly here rather than silently patching the
wrong place.

Reads the saved published page, inserts CSS + JS + the panel, writes a new file ready
to republish to the same artifact URL. Everything is additive: no existing markup,
style rule or function is changed except the four anchors patched below.
"""
import io
import os

SRC = (r"C:\Users\Adam\.claude\projects\D--Gil-spike-sorting-agent"
       r"\8186c809-76fa-41ff-bbca-59d44125e5fc\tool-results"
       r"\artifact-74dec064-1792083808-c26b.html")
OUT = (r"C:\Users\Adam\AppData\Local\Temp\claude\D--Gil-spike-sorting-agent"
       r"\8186c809-76fa-41ff-bbca-59d44125e5fc\scratchpad\review_tool_merged.html")

CSS = """
/* --- possible merges (per unit) -------------------------------------------- */
.merge-panel{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:12px;margin-bottom:16px;}
.merge-head{display:flex;align-items:baseline;justify-content:space-between;gap:10px;
  flex-wrap:wrap;margin-bottom:8px;}
.merge-why{font-size:10.5px;color:var(--text-3);margin-bottom:10px;max-width:92ch;line-height:1.5;}
.mc{border:1px solid var(--border);border-radius:8px;margin-bottom:8px;overflow:hidden;}
.mc-head{display:flex;align-items:center;gap:8px;padding:8px 10px;cursor:pointer;
  font-family:'IBM Plex Mono',monospace;font-size:11px;}
.mc-head:hover{background:var(--surface-2);}
.mc-head .pid{font-weight:700;font-size:12px;width:52px;flex-shrink:0;}
.mc-head .tag{font-size:9px;padding:1px 6px;border-radius:8px;font-weight:600;}
.mc-head .spacer{flex:1;}
.mc-head .nums{color:var(--text-3);font-size:10px;display:flex;gap:10px;flex-wrap:wrap;}
.mc-head .nums b{color:var(--text-2);font-weight:600;}
.tag.one{background:var(--good-soft);color:var(--good);}
.tag.two{background:var(--bad-soft);color:var(--bad);}
.tag.partial{background:var(--warn-soft);color:var(--warn);}
.tag.thin{background:var(--surface-2);color:var(--text-3);}
.tag.drift{background:var(--accent-soft);color:var(--accent);}
.tag.decided{background:var(--accent-soft);color:var(--accent);}
.mc-body{display:none;padding:10px;border-top:1px solid var(--border);}
.mc.open .mc-body{display:block;}
.mc-charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:10px;}
.mc-chart h5{font-size:9.5px;color:var(--text-3);text-transform:uppercase;letter-spacing:.04em;
  margin:0 0 4px;font-weight:600;}
.mc-chart svg{width:100%;height:auto;display:block;}
.mc-wide{grid-column:1/-1;}
.mc-verdict{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-top:10px;}
.mbtn{padding:6px 12px;border-radius:7px;border:1.5px solid var(--border);background:var(--surface);
  color:var(--text-2);font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:600;cursor:pointer;}
.mbtn:hover{border-color:var(--text-3);}
.mbtn.on-merge{background:var(--good-soft);color:var(--good);border-color:var(--good);}
.mbtn.on-sep{background:var(--bad-soft);color:var(--bad);border-color:var(--bad);}
.mbtn.on-unsure{background:var(--warn-soft);color:var(--warn);border-color:var(--warn);}
.mbtn.on-label{background:var(--accent-soft);color:var(--accent);border-color:var(--accent);}
.mc-note{flex:1;min-width:160px;background:var(--surface);border:1px solid var(--border);
  color:var(--text);border-radius:6px;padding:5px 7px;font-size:11px;font-family:'Archivo',sans-serif;}
.mc-sep{font-size:10px;color:var(--text-3);font-family:'IBM Plex Mono',monospace;}
.merge-export{font-size:10px;padding:3px 10px;border-radius:6px;border:1px solid var(--accent);
  background:var(--accent-soft);color:var(--accent);cursor:pointer;font-weight:600;
  font-family:'IBM Plex Mono',monospace;}
.merge-dot{font-size:9px;font-family:'IBM Plex Mono',monospace;color:var(--accent);
  background:var(--accent-soft);border-radius:8px;padding:1px 5px;flex-shrink:0;line-height:1.4;}
.mc-legend{font-size:10px;color:var(--text-3);font-family:'IBM Plex Mono',monospace;margin-top:6px;}
.mc-a{stroke:var(--accent);fill:none;stroke-width:2;}
.mc-b{stroke:var(--violet);fill:none;stroke-width:2;stroke-dasharray:5 2.5;}
"""

JS = """
/* ============================================================================
   POSSIBLE MERGES -- per-unit merge candidates, shown while reviewing a unit.

   Data: data/merge_candidates.json, built by scripts/build_merge_candidates.py
   (every pair whose peak channels are within 60 um, with the cross-correlogram,
   the combined train's ACG, and raw mean waveforms in microvolts). Fetched lazily
   the first time a unit is opened, so the page still loads at its old speed.

   What the panel is for: when you are looking at a unit, it tells you whether any
   other cluster might be the SAME NEURON, and shows the evidence for each. It does
   not merge anything. Decisions are saved per pair in the shared database and
   exported as the input for scripts/write_unit_merges.py, which is what actually
   reaches the dataset -- after you re-run sync_pipeline for the session.

   The evidence, in the order it should be weighed:
     1. the cross-correlogram. One neuron cannot fire twice inside its refractory
        period, so if these two clusters are one neuron the +/-1.5 ms window is
        EMPTY. dip_ratio is what is observed there over what the correlogram's own
        flanks predict, and it is always shown with the expected count: a ratio of
        0.00 measured where 0.7 counts were expected is not evidence of anything.
     2. the merged ACG -- the nearest thing to seeing the merged unit before it
        exists. If merging fills in the refractory gap, it is not one neuron.
     3. waveform and footprint, in microvolts on one shared scale.
     4. the session timelines, which are the ONLY evidence for a drift pair: those
        two units are hardly ever present at the same time, so their correlogram is
        empty by construction.
   ========================================================================= */
const MERGE_DATA_URL = 'data/merge_candidates.json';
const MERGED_ID_BASE = 10000;          /* same floor as sync_pipeline.unit_merges */
const DIP_ONE = 0.45, DIP_TWO = 0.8, DIP_SYNC = 1.5;
const EXP_STRONG = 20, EXP_USABLE = 5;
let MERGE = null;                      /* the loaded candidate document */
let mergeByUnit = {};                  /* unit_id -> [pair, ...] */
let mergeVerdicts = {};                /* pair_id -> {verdict, quality_label, note} */
let mergeLoadState = 'idle';
let mergeOpen = {};                    /* pair_id -> expanded? */

function mergeCount(uid){ return (mergeByUnit[uid] || []).length; }

/* On a Neuropixels probe nearly every unit has SOME cluster within 60 um, so "has a
   candidate" marks almost every row and tells you nothing. What deserves attention is
   a candidate the evidence actually supports: a readable refractory dip, or a
   drift-pair signature the correlogram cannot see. That is what the list marker counts
   and what this panel puts first; the rest stay one click away rather than crowding
   the real ones out. */
function isWorthLook(p){ return dipClass(p)[0] === 'one' || isDrift(p); }
function mergeStrongCount(uid){ return (mergeByUnit[uid] || []).filter(isWorthLook).length; }
let mergeShowAll = {};   /* unit_id -> show the unconvincing neighbours too */

function loadMergeData(){
  if(mergeLoadState !== 'idle') return;
  mergeLoadState = 'loading';
  fetch(MERGE_DATA_URL).then(r => r.json()).then(doc => {
    MERGE = doc;
    mergeByUnit = {};
    doc.pairs.forEach(p => {
      (mergeByUnit[p.unit_a] = mergeByUnit[p.unit_a] || []).push(p);
      (mergeByUnit[p.unit_b] = mergeByUnit[p.unit_b] || []).push(p);
    });
    mergeLoadState = 'ready';
    if(db){
      db.collection('merge_verdicts').onSnapshot(snap => {
        snap.docs.forEach(d => { const v = d.data(); if(v && v.pair_id) mergeVerdicts[v.pair_id] = v; });
        if(selected != null) renderDetail(selected);
      }, e => console.error('merge_verdicts', e));
    }
    renderList();
    if(selected != null) renderDetail(selected);
  }).catch(e => {
    mergeLoadState = 'failed';
    console.error('merge candidates', e);
    if(selected != null) renderDetail(selected);
  });
}

function mFmt(v, d){ return (v === null || v === undefined || Number.isNaN(v)) ? '--' : Number(v).toFixed(d === undefined ? 2 : d); }

function dipClass(p){
  const exp = p.ccg.expected_refractory || 0, d = p.ccg.dip_ratio;
  if(d === null || exp < EXP_USABLE) return ['thin', 'too few coincident spikes to read a dip'];
  if(d >= DIP_SYNC) return ['two', 'a PEAK at zero lag: synchrony or one spike detected twice, not one neuron'];
  if(d >= DIP_TWO) return ['two', 'flat through the refractory window: two neurons'];
  if(d <= DIP_ONE) return ['one', 'a real refractory dip: consistent with ONE neuron'];
  return ['partial', 'a partial dip: contaminated, or one neuron plus a neighbour'];
}
function isDrift(p){
  return p.rate_corr !== null && p.rate_corr <= -0.1 &&
         p.active_overlap !== null && p.active_overlap <= 0.35;
}
function partnerOf(p, uid){ return p.unit_a === uid ? p.unit_b : p.unit_a; }

/* --- small charts, drawn from one scale, text in theme colours -------------- */
function mCorrelogram(hist, binMs, maxLagMs, flank){
  const W = 250, H = 132, L = 26, R = 6, T = 8, B = 24;
  const n = hist.length, iw = W - L - R, ih = H - T - B;
  let max = 1; for(let i = 0; i < n; i++) if(hist[i] > max) max = hist[i];
  const bw = iw / n, refr = 1.5;
  const lagOf = i => -maxLagMs + (i + 0.5) * binMs;
  let s = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="correlogram">';
  const rx0 = L + ((-refr + maxLagMs) / (2 * maxLagMs)) * iw;
  const rx1 = L + ((refr + maxLagMs) / (2 * maxLagMs)) * iw;
  s += '<rect x="' + rx0 + '" y="' + T + '" width="' + (rx1 - rx0) + '" height="' + ih +
       '" fill="var(--accent-soft)"/>';
  for(let i = 0; i < n; i++){
    const h = (hist[i] / max) * ih;
    const inR = Math.abs(lagOf(i)) <= refr;
    s += '<rect x="' + (L + i * bw + 0.2).toFixed(1) + '" y="' + (T + ih - h).toFixed(1) +
         '" width="' + Math.max(0.6, bw - 0.4).toFixed(2) + '" height="' + h.toFixed(1) +
         '" fill="' + (inR ? 'var(--accent)' : 'var(--text-3)') + '"/>';
  }
  if(flank){
    const fy = T + ih - (flank / max) * ih;
    if(fy > T) s += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + fy.toFixed(1) + '" y2="' +
      fy.toFixed(1) + '" stroke="var(--text-3)" stroke-width="1" stroke-dasharray="3 3"/>';
  }
  s += '<text x="' + (W - R) + '" y="' + (T + 7) + '" text-anchor="end" font-size="8" fill="var(--text-3)" font-family="monospace">' + max + '</text>';
  [[-maxLagMs, L], [0, L + iw / 2], [maxLagMs, W - R]].forEach(t => {
    s += '<text x="' + t[1] + '" y="' + (T + ih + 10) + '" text-anchor="middle" font-size="8" fill="var(--text-3)" font-family="monospace">' + t[0] + '</text>';
  });
  s += '<text x="' + (L + iw / 2) + '" y="' + (H - 2) + '" text-anchor="middle" font-size="8" fill="var(--text-3)" font-family="monospace">lag (ms)</text>';
  s += '</svg>';
  return s;
}

function mWaveforms(wa, wb, chIdxA, chIdxB){
  const W = 250, H = 132, L = 30, R = 6, T = 10, B = 24;
  const a = wa.uv[chIdxA], b = wb.uv[chIdxB];
  const n = a.length, iw = W - L - R, ih = H - T - B;
  let lo = Infinity, hi = -Infinity;
  for(let i = 0; i < n; i++){
    lo = Math.min(lo, a[i], b[i]); hi = Math.max(hi, a[i], b[i]);
  }
  const pad = (hi - lo) * 0.1 || 1; lo -= pad; hi += pad;
  const X = i => L + (i / (n - 1)) * iw;
  const Y = v => T + ih - ((v - lo) / (hi - lo)) * ih;
  const path = w => w.map((v, i) => (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(v).toFixed(1)).join(' ');
  let s = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="mean waveforms">';
  s += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + Y(0).toFixed(1) + '" y2="' + Y(0).toFixed(1) + '" stroke="var(--border)"/>';
  s += '<path class="mc-a" d="' + path(a) + '"/><path class="mc-b" d="' + path(b) + '"/>';
  s += '<text x="' + (L - 3) + '" y="' + (T + 7) + '" text-anchor="end" font-size="8" fill="var(--text-3)" font-family="monospace">' + Math.round(hi) + '</text>';
  s += '<text x="' + (L - 3) + '" y="' + (T + ih) + '" text-anchor="end" font-size="8" fill="var(--text-3)" font-family="monospace">' + Math.round(lo) + '</text>';
  s += '<text x="' + (L + iw / 2) + '" y="' + (H - 2) + '" text-anchor="middle" font-size="8" fill="var(--text-3)" font-family="monospace">microvolts, one shared scale</text>';
  s += '</svg>';
  return s;
}

/* Both units' rate across the session, from the page's own TIMELINE data. */
function mTimelines(uidA, uidB){
  const ca = decodeCounts(uidA), cb = decodeCounts(uidB);
  if(!ca || !cb) return '<div class="no-data" style="padding:20px 0">no timeline for one of these units</div>';
  const W = 520, H = 110, L = 30, R = 6, T = 8, B = 22;
  const binS = TL.bin_s, n = Math.max(ca.length, cb.length);
  const iw = W - L - R, ih = H - T - B;
  let max = 0.001;
  for(let i = 0; i < n; i++){ max = Math.max(max, (ca[i] || 0) / binS, (cb[i] || 0) / binS); }
  const X = i => L + (i / Math.max(1, n - 1)) * iw;
  const Y = v => T + ih - (v / max) * ih;
  const path = c => { let d = ''; for(let i = 0; i < c.length; i++){ d += (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(c[i] / binS).toFixed(1); } return d; };
  let s = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="firing rate across the session">';
  s += '<line x1="' + L + '" x2="' + L + '" y1="' + T + '" y2="' + (T + ih) + '" stroke="var(--border)"/>';
  s += '<path class="mc-a" d="' + path(ca) + '"/><path class="mc-b" d="' + path(cb) + '"/>';
  s += '<text x="' + (L - 3) + '" y="' + (T + 7) + '" text-anchor="end" font-size="8" fill="var(--text-3)" font-family="monospace">' + max.toFixed(0) + '</text>';
  s += '<text x="' + (L - 3) + '" y="' + (T + ih) + '" text-anchor="end" font-size="8" fill="var(--text-3)" font-family="monospace">0</text>';
  s += '<text x="' + (L + iw / 2) + '" y="' + (H - 2) + '" text-anchor="middle" font-size="8" fill="var(--text-3)" font-family="monospace">minutes into the recording (Hz)</text>';
  s += '</svg>';
  return s;
}

/* --- decisions ------------------------------------------------------------- */
function setMergeVerdict(pairId, patch){
  const p = MERGE.pairs.find(x => x.pair_id === pairId);
  const prev = mergeVerdicts[pairId] || {};
  const next = {
    pair_id: pairId, unit_a: p.unit_a, unit_b: p.unit_b,
    verdict: patch.verdict !== undefined ? patch.verdict : (prev.verdict || 'unsure'),
    quality_label: patch.quality_label !== undefined ? patch.quality_label : (prev.quality_label || ''),
    note: patch.note !== undefined ? patch.note : (prev.note || ''),
    decided_at: Date.now(),
    /* stamped so a verdict can never be read against a different sorting */
    spike_clusters_sha1: MERGE.clustering.spike_clusters_sha1,
    dip_ratio: p.ccg.dip_ratio, waveform_corr: p.waveform_corr,
  };
  mergeVerdicts[pairId] = next;
  if(db) db.collection('merge_verdicts').doc(pairId).set(next).catch(e => console.error(e));
  renderList();
  if(selected != null) renderDetail(selected);
}

/* Accepted pairs chain transitively: the pipeline refuses a unit that parents two
   different merges, so 38+163 plus 163+204 has to become one merge of three. */
function mergeGroups(){
  const parent = {};
  const find = x => { while(parent[x] !== x){ parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  const accepted = Object.keys(mergeVerdicts).filter(id => mergeVerdicts[id].verdict === 'merge');
  accepted.forEach(id => {
    const v = mergeVerdicts[id];
    [v.unit_a, v.unit_b].forEach(u => { if(!(u in parent)) parent[u] = u; });
    const ra = find(v.unit_a), rb = find(v.unit_b);
    if(ra !== rb) parent[ra] = rb;
  });
  const groups = {};
  Object.keys(parent).forEach(u => { const r = find(Number(u)); (groups[r] = groups[r] || []).push(Number(u)); });
  return Object.keys(groups).map(r => {
    const members = groups[r].sort((a, b) => a - b);
    const ids = accepted.filter(id => members.indexOf(mergeVerdicts[id].unit_a) >= 0 &&
                                      members.indexOf(mergeVerdicts[id].unit_b) >= 0);
    const labels = ids.map(id => mergeVerdicts[id].quality_label).filter(Boolean);
    const notes = ids.map(id => mergeVerdicts[id].note).filter(Boolean);
    return { parents: members, pairIds: ids, label: labels.length ? labels[labels.length - 1] : '', note: notes.join('; ') };
  }).sort((a, b) => a.parents[0] - b.parents[0]);
}

function mergeExportDoc(){
  return {
    session_id: SESSION_ID,
    decided_in: 'unit review tool, possible-merges panel',
    clustering: MERGE ? MERGE.clustering : null,
    merges: mergeGroups().map(g => {
      const m = { parents: g.parents };
      if(g.label) m.quality_label = g.label;
      m.note = g.note || ('merge review; pairs ' + g.pairIds.join(', '));
      return m;
    }),
  };
}

async function exportMerges(){
  const groups = mergeGroups();
  if(!groups.length){ alert('Nothing marked "merge" yet.'); return; }
  const text = JSON.stringify(mergeExportDoc(), null, 2);
  let saved = false;
  try {
    const dl = await claude.use('downloads');
    if(dl){ await dl.save({ filename: 'merges_' + SESSION_ID + '.json', data: text }); saved = true; }
  } catch(e){ /* declined or unavailable -- fall through to the clipboard */ }
  if(!saved){
    try { await navigator.clipboard.writeText(text); } catch(e){}
  }
  alert(groups.length + ' merge(s) ' + (saved ? 'downloaded' : 'copied to the clipboard') +
        '.\\n\\nNext: run scripts/write_unit_merges.py with this file, then re-run ' +
        'sync_pipeline for the session -- nothing reaches synced/ until you do.\\n\\n' +
        groups.map(g => '  ' + g.parents.join(' + ') + (g.label ? '  [' + g.label + ']' : '')).join('\\n'));
}

/* --- the panel ------------------------------------------------------------- */
function renderMergePanel(container, uid){
  if(mergeLoadState === 'idle'){ loadMergeData(); }
  if(mergeLoadState === 'loading' || mergeLoadState === 'idle'){
    container.innerHTML = '<div class="merge-head"><div class="panel-label">Possible merges</div></div>' +
      '<div class="merge-why">loading merge candidates...</div>';
    return;
  }
  if(mergeLoadState === 'failed'){
    container.innerHTML = '<div class="merge-head"><div class="panel-label">Possible merges</div></div>' +
      '<div class="merge-why">could not load data/merge_candidates.json. Rebuild it with ' +
      'scripts/build_merge_candidates.py and republish.</div>';
    return;
  }
  const all = (mergeByUnit[uid] || []).slice().sort((x, y) => {
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
  const list = showAll ? all : strong.concat(decidedWeak);
  const nDecided = Object.keys(mergeVerdicts).filter(id => mergeVerdicts[id].verdict === 'merge').length;
  let h = '<div class="merge-head"><div class="panel-label">Possible merges &middot; ' +
    (strong.length ? strong.length + ' worth a look' : 'none convincing') +
    ' of ' + all.length + ' neighbour' + (all.length === 1 ? '' : 's') + ' for unit #' + uid + '</div>' +
    '<button class="merge-export" id="mergeExportBtn">export ' + nDecided + ' merge' +
    (nDecided === 1 ? '' : 's') + '</button></div>';

  if(!all.length){
    h += '<div class="merge-why">No other cluster has its peak channel within ' +
      MERGE.max_dist_um + ' &micro;m of this one, so there is nothing here that could be the ' +
      'same neuron. (Units classified as noise are left out of the search entirely.)</div>';
    container.innerHTML = h;
    const b = container.querySelector('#mergeExportBtn');
    if(b) b.addEventListener('click', exportMerges);
    return;
  }

  if(!list.length){
    h += '<div class="merge-why">' + all.length + ' cluster' + (all.length === 1 ? ' is' : 's are') +
      ' within ' + MERGE.max_dist_um + ' &micro;m of this one, but none of them shows the ' +
      'evidence a merge needs: no readable refractory dip, no drift-pair signature. ' +
      'That is a result, not an empty list &mdash; on a dense probe most neighbours are ' +
      'simply other neurons.</div>';
  }

  h += '<div class="merge-why">Clusters within ' + MERGE.max_dist_um + ' &micro;m of this unit ' +
    'that might be the <b>same neuron</b>, best evidence first. The decider is the ' +
    'cross-correlogram: one neuron cannot fire twice inside its refractory period, so if two ' +
    'clusters are one neuron the &plusmn;1.5 ms window is empty. <b>dip</b> is what is observed ' +
    'there over what the correlogram&rsquo;s own flanks predict, and it is only a measurement ' +
    'when enough coincident spikes exist &mdash; which is why pairs that cannot be read say so ' +
    'rather than showing a flattering 0.00. Deciding here changes nothing by itself: export, ' +
    'run write_unit_merges.py, then re-run the pipeline.</div>';

  list.forEach(p => {
    const other = partnerOf(p, uid);
    const ou = byId[other] || {};
    const [cls, say] = dipClass(p);
    const v = mergeVerdicts[p.pair_id];
    const open = mergeOpen[p.pair_id];
    const cost = p.merged_refractory.violation_pct;
    h += '<div class="mc' + (open ? ' open' : '') + '" data-pair="' + p.pair_id + '">' +
      '<div class="mc-head" data-toggle="' + p.pair_id + '">' +
      '<span class="pid">#' + other + '</span>' +
      '<span class="tag ' + cls + '">' + (p.ccg.dip_ratio === null ? 'no dip' : 'dip ' + mFmt(p.ccg.dip_ratio)) + '</span>' +
      (isDrift(p) ? '<span class="tag drift">drift pair?</span>' : '') +
      (v ? '<span class="tag decided">' + v.verdict.toUpperCase() + (v.quality_label ? ' / ' + v.quality_label : '') + '</span>' : '') +
      '<span class="spacer"></span>' +
      '<span class="nums">' +
      '<span>' + (ou.label || '?') + '</span>' +
      '<span>wf r <b>' + mFmt(p.waveform_corr) + '</b></span>' +
      '<span><b>' + mFmt(p.distance_um, 0) + '</b>&micro;m</span>' +
      '<span>merged viol <b>' + mFmt(cost, 3) + '%</b></span>' +
      '<span>' + (open ? 'hide' : 'inspect') + '</span></span></div>' +
      '<div class="mc-body">' + (open ? mergeBody(p, uid, other, say) : '') + '</div></div>';
  });

  if(weak.length){
    h += '<button class="merge-export" id="mergeShowAllBtn" style="border-color:var(--border);' +
      'background:var(--surface);color:var(--text-3);">' +
      (showAll ? 'hide the ' + weak.length + ' unconvincing neighbour' + (weak.length === 1 ? '' : 's')
               : 'show the other ' + weak.length + ' neighbour' + (weak.length === 1 ? '' : 's') +
                 ' (flat correlograms)') + '</button>';
  }
  container.innerHTML = h;
  const allBtn = container.querySelector('#mergeShowAllBtn');
  if(allBtn) allBtn.addEventListener('click', () => { mergeShowAll[uid] = !mergeShowAll[uid]; renderDetail(uid); });

  container.querySelectorAll('.mc-head').forEach(el => {
    el.addEventListener('click', () => {
      const id = el.dataset.toggle;
      mergeOpen[id] = !mergeOpen[id];
      renderDetail(uid);
    });
  });
  container.querySelectorAll('[data-mv]').forEach(b => {
    b.addEventListener('click', () => setMergeVerdict(b.dataset.pair, JSON.parse(b.dataset.mv)));
  });
  container.querySelectorAll('.mc-note').forEach(t => {
    let timer = null;
    t.addEventListener('input', () => {
      clearTimeout(timer);
      const val = t.value, id = t.dataset.pair;
      timer = setTimeout(() => {
        const prev = mergeVerdicts[id] || {};
        mergeVerdicts[id] = Object.assign({}, prev, { note: val });
        if(db && prev.verdict) setMergeVerdict(id, { note: val });
      }, 700);
    });
  });
  const btn = container.querySelector('#mergeExportBtn');
  if(btn) btn.addEventListener('click', exportMerges);
}

function mergeBody(p, uid, other, say){
  const a = MERGE.units[String(p.unit_a)], b = MERGE.units[String(p.unit_b)];
  const shared = a.waveform.channels.filter(c => b.waveform.channels.indexOf(c) >= 0);
  let ia = 0, ib = 0, chTxt = 'no shared channel';
  if(shared.length){
    let best = shared[0], bestAmp = -1;
    shared.forEach(c => {
      const amp = Math.max.apply(null, a.waveform.uv[a.waveform.channels.indexOf(c)].map(Math.abs));
      if(amp > bestAmp){ bestAmp = amp; best = c; }
    });
    ia = a.waveform.channels.indexOf(best); ib = b.waveform.channels.indexOf(best);
    chTxt = 'channel ' + best;
  }
  const v = mergeVerdicts[p.pair_id] || {};
  const pick = (field, val, cls, text) => {
    const on = (field === 'verdict' ? (v.verdict === val) : ((v.quality_label || '') === val));
    const patch = {}; patch[field] = val;
    return '<button class="mbtn' + (on ? ' ' + cls : '') + '" data-pair="' + p.pair_id +
      '" data-mv=' + JSON.stringify(JSON.stringify(patch)) + '>' + text + '</button>';
  };
  let h = '<div class="mc-charts">' +
    '<div class="mc-chart"><h5>cross-correlogram #' + p.unit_a + ' vs #' + p.unit_b + '</h5>' +
    mCorrelogram(p.ccg_hist, p.ccg_bin_ms, p.ccg_max_lag_ms, p.ccg.flank_per_bin) +
    '<div class="mc-legend">' + p.ccg.n_refractory + ' observed vs ' +
    mFmt(p.ccg.expected_refractory, 1) + ' expected &mdash; ' + say + '</div></div>' +
    '<div class="mc-chart"><h5>ACG if merged &middot; ' + mFmt(p.merged_refractory.violation_pct, 3) + '% violations</h5>' +
    mCorrelogram(p.merged_acg_hist, p.ccg_bin_ms, p.ccg_max_lag_ms, null) +
    '<div class="mc-legend">apart: ' + mFmt(p.refractory_a.violation_pct, 3) + '% / ' +
    mFmt(p.refractory_b.violation_pct, 3) + '% &mdash; if merging fills the gap in, it is not one neuron</div></div>' +
    '<div class="mc-chart"><h5>mean raw waveforms &middot; ' + chTxt + '</h5>' +
    mWaveforms(a.waveform, b.waveform, ia, ib) +
    '<div class="mc-legend">#' + p.unit_a + ' solid, #' + p.unit_b + ' dashed &middot; amplitude ratio ' +
    mFmt(p.amplitude_ratio) + ' &middot; KS template similarity ' + mFmt(p.template_similarity, 3) + '</div></div>' +
    '<div class="mc-chart mc-wide"><h5>both units across the session &mdash; the drift-pair check</h5>' +
    mTimelines(p.unit_a, p.unit_b) +
    '<div class="mc-legend">#' + p.unit_a + ' solid, #' + p.unit_b + ' dashed &middot; rate correlation ' +
    mFmt(p.rate_corr) + ' &middot; simultaneous-activity overlap ' + mFmt(p.active_overlap) +
    (isDrift(p) ? ' &mdash; <b>they trade places: a drift pair, which the correlogram cannot see</b>' : '') +
    '</div></div></div>';

  h += '<div class="mc-verdict">' +
    pick('verdict', 'merge', 'on-merge', 'MERGE') +
    pick('verdict', 'separate', 'on-sep', 'SEPARATE') +
    pick('verdict', 'unsure', 'on-unsure', 'UNSURE') +
    '<span class="mc-sep">| merged unit is</span>' +
    pick('quality_label', 'good', 'on-label', 'good') +
    pick('quality_label', 'mua', 'on-label', 'mua') +
    pick('quality_label', '', 'on-label', 'inherit') +
    '<input class="mc-note" data-pair="' + p.pair_id + '" placeholder="why (goes into the manifest)" value="' +
    String(v.note || '').replace(/"/g, '&quot;') + '">' +
    '</div>' +
    '<div class="mc-legend">Scoring the merged unit is only possible once you have seen it &mdash; ' +
    'the ACG above and the overlaid waveform ARE that unit. Left on <b>inherit</b>, the pipeline ' +
    'takes the parents&rsquo; shared label, or <b>mua</b> when they disagree, which undersells a real merge.</div>';
  return h;
}
"""


def main():
    s = io.open(SRC, encoding="utf-8").read()
    print("source %d bytes" % len(s))

    def sub(old, new, label):
        nonlocal s
        assert s.count(old) == 1, "%s: anchor found %d times" % (label, s.count(old))
        s = s.replace(old, new)
        print("  ok:", label)

    # 1. styles, at the end of the page's own stylesheet
    sub(".row .sel-dot{width:7px;height:7px;border-radius:2px;flex-shrink:0;background:var(--violet);}\n</style>",
        ".row .sel-dot{width:7px;height:7px;border-radius:2px;flex-shrink:0;background:var(--violet);}\n"
        + CSS + "</style>", "css")

    # 2. the code, immediately before renderDetail
    sub("function renderDetail(uid){", JS + "\nfunction renderDetail(uid){", "js")

    # 3. the panel itself, right after the session timeline
    sub('    <div class="timeline-panel"></div>\n',
        '    <div class="timeline-panel"></div>\n    <div class="merge-panel"></div>\n',
        "panel markup")

    # 4. render it
    sub("  renderTimeline(el.querySelector('.timeline-panel'), uid);",
        "  renderTimeline(el.querySelector('.timeline-panel'), uid);\n"
        "  renderMergePanel(el.querySelector('.merge-panel'), uid);", "panel render")

    # 5. a filter for units that have candidates at all
    sub('        <option value="no_sel">without time selection</option>',
        '        <option value="no_sel">without time selection</option>\n'
        '        <option value="has_merge">with a merge candidate worth a look</option>\n'
        '        <option value="merge_decided">with a merge decision</option>', "filter option")
    sub("  if(reviewFilter==='no_sel') list = list.filter(u => !rangesFor(u.unit_id).length);",
        "  if(reviewFilter==='no_sel') list = list.filter(u => !rangesFor(u.unit_id).length);\n"
        "  if(reviewFilter==='has_merge') list = list.filter(u => mergeStrongCount(u.unit_id) > 0);\n"
        "  if(reviewFilter==='merge_decided') list = list.filter(u => "
        "(mergeByUnit[u.unit_id]||[]).some(p => mergeVerdicts[p.pair_id]));", "filter logic")

    # 6. a marker in the list, so candidates are visible without opening every unit
    sub("""      ${nSel ? `<span class="sel-dot" title="${nSel} time range(s) selected"></span>` : ''}""",
        """      ${nSel ? `<span class="sel-dot" title="${nSel} time range(s) selected"></span>` : ''}
      ${mergeStrongCount(u.unit_id) ? `<span class="merge-dot" title="${mergeStrongCount(u.unit_id)} merge candidate(s) worth a look, of ${mergeCount(u.unit_id)} neighbour(s)">${mergeStrongCount(u.unit_id)}&#8644;</span>` : ''}""",
        "list marker")

    io.open(OUT, "w", encoding="utf-8").write(s)
    print("wrote %s (%d bytes, +%d)" % (OUT, os.path.getsize(OUT), os.path.getsize(OUT) - os.path.getsize(SRC)))


main()

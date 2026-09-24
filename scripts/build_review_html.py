import json, os, sys

# usage: python build_review_html.py <session_id>
SESSION_ID = sys.argv[1] if len(sys.argv) > 1 else "20260901_085606"
IN_PATH = (r"D:\Gil\spike_sorting_agent\outputs\pipeline_review_data.json" if SESSION_ID == "20260901_085606"
           else rf"D:\Gil\spike_sorting_agent\outputs\pipeline_review_data_{SESSION_ID}.json")
OUT_PATH = rf"D:\Gil\spike_sorting_agent\outputs\unit_review_pipeline_{SESSION_ID}.html"

with open(IN_PATH) as f:
    units = json.load(f)

html = r"""<title>Unit Review Pipeline</title>
<style>
:root{
  --bg:#f4f6f8; --surface:#ffffff; --surface-2:#eef1f4; --border:#d7dee4;
  --text:#12181f; --text-2:#445261; --text-3:#7c8b9a;
  --accent:#0f8b83; --accent-soft:#d7f0ee;
  --warn:#b5790a; --warn-soft:#fbeed2;
  --bad:#c2453f; --bad-soft:#fbe2e0;
  --good:#2f8f57; --good-soft:#ddf3e4;
  --violet:#7a54a8; --blue:#3f6fd1;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0a0e13; --surface:#12181f; --surface-2:#1a222b; --border:#232d38;
    --text:#e8edf2; --text-2:#9aacbd; --text-3:#61707e;
    --accent:#4fd1c5; --accent-soft:#173430;
    --warn:#f2b84b; --warn-soft:#332508;
    --bad:#ef6461; --bad-soft:#3a1a19;
    --good:#7dd490; --good-soft:#173021;
    --violet:#a98fd6; --blue:#7fa2f0;
  }
}
:root[data-theme="dark"]{
  --bg:#0a0e13; --surface:#12181f; --surface-2:#1a222b; --border:#232d38;
  --text:#e8edf2; --text-2:#9aacbd; --text-3:#61707e;
  --accent:#4fd1c5; --accent-soft:#173430;
  --warn:#f2b84b; --warn-soft:#332508;
  --bad:#ef6461; --bad-soft:#3a1a19;
  --good:#7dd490; --good-soft:#173021;
  --violet:#a98fd6; --blue:#7fa2f0;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'Archivo',system-ui,sans-serif;margin:0;height:100vh;overflow:hidden;}
.mono{font-family:'IBM Plex Mono',ui-monospace,monospace;font-variant-numeric:tabular-nums;}
.app{display:grid;grid-template-columns:minmax(360px,460px) 1fr;height:100vh;}
.left{border-right:1px solid var(--border);display:flex;flex-direction:column;min-height:0;}
.top{padding:14px 16px 10px;border-bottom:1px solid var(--border);}
.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin-bottom:4px;}
h1{font-size:16px;font-weight:700;margin:0 0 8px;}
.progress-bar{height:6px;background:var(--surface-2);border-radius:3px;overflow:hidden;margin-top:6px;}
.progress-fill{height:100%;background:var(--accent);transition:width .2s;}
.progress-label{font-size:11px;color:var(--text-3);margin-top:4px;}
.controls{padding:8px 16px;border-bottom:1px solid var(--border);display:flex;gap:6px;flex-wrap:wrap;align-items:center;}
.controls select, .controls input{background:var(--surface);border:1px solid var(--border);color:var(--text);
  border-radius:6px;padding:4px 7px;font-size:11.5px;font-family:'Archivo',sans-serif;}
.controls input[type=text]{width:80px;}
.chip-toggle{display:flex;gap:4px;}
.chip-toggle button{background:var(--surface);border:1px solid var(--border);color:var(--text-2);border-radius:16px;
  padding:3px 9px;font-size:10.5px;cursor:pointer;font-family:'IBM Plex Mono',monospace;}
.chip-toggle button.active[data-kind=GOOD]{background:var(--good-soft);color:var(--good);border-color:transparent;}
.chip-toggle button.active[data-kind=MUA]{background:var(--warn-soft);color:var(--warn);border-color:transparent;}
.chip-toggle button.active[data-kind=NOISE]{background:var(--bad-soft);color:var(--bad);border-color:transparent;}
.list{flex:1;overflow-y:auto;}
.row{display:flex;align-items:center;gap:8px;padding:7px 16px;border-bottom:1px solid var(--border);cursor:pointer;font-size:11.5px;}
.row:hover{background:var(--surface-2);}
.row.selected{background:var(--accent-soft);}
.row .uid{font-family:'IBM Plex Mono',monospace;font-weight:700;width:38px;flex-shrink:0;}
.row .badge{font-size:8.5px;font-family:'IBM Plex Mono',monospace;padding:1px 5px;border-radius:8px;font-weight:600;flex-shrink:0;}
.row .badge.GOOD{background:var(--good-soft);color:var(--good);}
.row .badge.MUA{background:var(--warn-soft);color:var(--warn);}
.row .badge.NOISE{background:var(--bad-soft);color:var(--bad);}
.row .meta{color:var(--text-3);font-family:'IBM Plex Mono',monospace;font-size:10px;flex:1;text-align:right;}
.row .verdict-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;background:var(--border);}
.row .verdict-dot.good{background:var(--good);}
.row .verdict-dot.mua{background:var(--warn);}
.row .verdict-dot.noise{background:var(--bad);}

.right{overflow-y:auto;padding:20px 28px;}
.detail-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:14px;}
.detail-uid{font-family:'IBM Plex Mono',monospace;font-weight:700;font-size:22px;}
.detail-badges{display:flex;gap:6px;margin-top:6px;}
.badge-lg{font-size:11px;font-family:'IBM Plex Mono',monospace;padding:2px 9px;border-radius:12px;font-weight:600;}
.badge-lg.GOOD{background:var(--good-soft);color:var(--good);}
.badge-lg.MUA{background:var(--warn-soft);color:var(--warn);}
.badge-lg.NOISE{background:var(--bad-soft);color:var(--bad);}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px;}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px;}
.panel-label{font-size:10px;color:var(--text-3);text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px;}
.panel svg{width:100%;height:220px;display:block;}
.no-data{color:var(--text-3);font-size:11px;padding:80px 0;text-align:center;}
.metrics-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px;}
.metric{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:8px 10px;}
.metric-label{font-size:9.5px;color:var(--text-3);text-transform:uppercase;letter-spacing:.03em;}
.metric-val{font-family:'IBM Plex Mono',monospace;font-size:15px;font-weight:600;margin-top:2px;}
.reasons{margin-bottom:16px;font-size:11.5px;color:var(--text-2);}
.reasons b{color:var(--text);}
.chan-info{font-size:11.5px;color:var(--text-2);margin-bottom:16px;font-family:'IBM Plex Mono',monospace;}
.chan-info a{color:var(--accent);}
.verdict-panel{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px;}
.verdict-title{font-size:11px;color:var(--text-3);text-transform:uppercase;letter-spacing:.04em;margin-bottom:10px;}
.verdict-btns{display:flex;gap:8px;}
.vbtn{flex:1;padding:10px;border-radius:8px;border:1.5px solid var(--border);background:var(--surface);
  color:var(--text-2);font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:600;cursor:pointer;}
.vbtn:hover{border-color:var(--text-3);}
.vbtn.picked-good{background:var(--good-soft);color:var(--good);border-color:var(--good);}
.vbtn.picked-mua{background:var(--warn-soft);color:var(--warn);border-color:var(--warn);}
.vbtn.picked-noise{background:var(--bad-soft);color:var(--bad);border-color:var(--bad);}
.vbtn-key{opacity:.6;font-size:10px;margin-left:4px;}
.nav-hint{font-size:10.5px;color:var(--text-3);margin-top:10px;}
.db-status{font-size:10px;color:var(--text-3);margin-top:8px;}
.count-note{font-size:11px;color:var(--text-3);margin-left:auto;}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">

<div class="app">
  <div class="left">
    <div class="top">
      <div class="eyebrow">Unit Review Pipeline &middot; 20260901_085606</div>
      <h1>401 units, your verdict per unit</h1>
      <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
      <div class="progress-label" id="progressLabel">0 / 401 reviewed</div>
    </div>
    <div class="controls">
      <div class="chip-toggle" id="labelToggle">
        <button data-kind="GOOD" class="active">GOOD</button>
        <button data-kind="MUA" class="active">MUA</button>
        <button data-kind="NOISE" class="active">NOISE</button>
      </div>
      <select id="sortBy">
        <option value="unit_id">unit ID</option>
        <option value="sep_vs_noise_asc">sep vs noise (low first)</option>
        <option value="violation_ratio_desc">violation ratio</option>
        <option value="mean_rate_hz_desc">firing rate</option>
      </select>
      <select id="reviewFilter">
        <option value="all">all units</option>
        <option value="unreviewed">unreviewed only</option>
        <option value="reviewed">reviewed only</option>
      </select>
      <select id="shapeFilter">
        <option value="all">any burst shape</option>
        <option value="decay">decay</option>
        <option value="facilitation">facilitation</option>
        <option value="dip_then_rise">dip-then-rise</option>
        <option value="rise_then_dip">rise-then-dip</option>
        <option value="flat">flat</option>
        <option value="insufficient_data">insufficient data</option>
        <option value="excluded_high_rate">excluded (rate &ge;5Hz)</option>
      </select>
      <input type="text" id="searchBox" placeholder="unit or ch">
      <div class="count-note" id="countNote"></div>
    </div>
    <div class="controls">
      <label style="font-size:10px;color:var(--text-3);text-transform:uppercase;">Add filter</label>
      <select id="filterField">
        <option value="signalToNoiseRatio">SNR</option>
        <option value="rawAmplitude">Amplitude (&micro;V)</option>
        <option value="violation_ratio">Violation ratio</option>
        <option value="footprint_concentration_ratio">Footprint conc.</option>
        <option value="footprint_flatness_ratio">Footprint flatness</option>
        <option value="peak_trough_width_ms">Width (ms)</option>
        <option value="mean_rate_hz">Firing rate (Hz)</option>
        <option value="n_spikes">N spikes</option>
        <option value="sep_vs_noise_vector">Sep. vs noise (pctl)</option>
        <option value="sep_vs_other_vector">Sep. vs other units (pctl)</option>
        <option value="isi_amp_corr_all">ISI-amp corr (all)</option>
        <option value="isi_amp_corr_burst">ISI-amp corr (burst&lt;20ms)</option>
      </select>
      <span id="filterPctlWrap" style="display:none;font-size:11px;color:var(--text-3);">p<input type="number" id="filterPctl" value="80" min="0" max="100" style="width:38px;margin-left:2px;"></span>
      <select id="filterOp">
        <option value="gt">&gt;</option>
        <option value="lt">&lt;</option>
      </select>
      <input type="number" id="filterVal" placeholder="value" step="0.1" style="width:60px;">
      <button id="filterAdd" style="font-size:10px;padding:3px 9px;border-radius:6px;border:1px solid var(--accent);background:var(--accent-soft);color:var(--accent);cursor:pointer;font-weight:600;">+ add</button>
    </div>
    <div class="controls" id="activeFilters" style="min-height:28px;"></div>
    <div class="list" id="list"></div>
  </div>
  <div class="right" id="detail"></div>
</div>

<script>
const UNITS = __UNITS_JSON__;
const CH_COLORS = ['var(--accent)','var(--violet)','var(--warn)','var(--blue)'];
const byId = Object.fromEntries(UNITS.map(u => [u.unit_id, u]));
let verdicts = {};   // unit_id -> {verdict, ts}
let selected = UNITS[0].unit_id;
let db = null;

async function initDb(){
  try{
    db = await claude.use('db');
  }catch(e){ db = null; }
  if(!db){ document.getElementById('dbStatusText') && (document.getElementById('dbStatusText').textContent='local only (db unavailable)'); return; }
  db.collection('verdicts').onSnapshot(snap => {
    snap.docs.forEach(doc => {
      const d = doc.data();
      if(d) verdicts[doc.id] = d;
    });
    updateProgress();
    renderList();
    if(selected != null) renderDetail(selected);
  }, err => { console.error('db error', err); });
}

function setVerdict(uid, verdict){
  verdicts[uid] = {verdict, ts: Date.now()};
  updateProgress(); renderList(); renderDetail(uid);
  if(db){
    db.collection('verdicts').doc(String(uid)).set({verdict, ts: Date.now()}).catch(e => console.error(e));
  }
}

function updateProgress(){
  const n = Object.keys(verdicts).length;
  document.getElementById('progressFill').style.width = (n/UNITS.length*100).toFixed(1)+'%';
  document.getElementById('progressLabel').textContent = `${n} / ${UNITS.length} reviewed`;
}

function svgEl(tag, attrs){
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for(const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}
function renderWaveform(container, templ){
  const W=300,H=220,padL=6,padR=6,padT=10,padB=10;
  const nt = templ.length, nc = templ[0].length;
  const svg = svgEl('svg',{viewBox:`0 0 ${W} ${H}`});
  let ymin=Infinity, ymax=-Infinity;
  for(let s=0;s<nt;s++) for(let c=0;c<nc;c++){ ymin=Math.min(ymin,templ[s][c]); ymax=Math.max(ymax,templ[s][c]); }
  const x = s => padL + (s/(nt-1))*(W-padL-padR);
  const y = v => padT + (1-(v-ymin)/(ymax-ymin+1e-9))*(H-padT-padB);
  svg.appendChild(svgEl('line',{x1:padL,x2:W-padR,y1:y(0),y2:y(0),stroke:'var(--border)','stroke-width':1}));
  for(let c=0;c<nc;c++){
    let line=[];
    for(let s=0;s<nt;s++) line.push(`${x(s)},${y(templ[s][c])}`);
    svg.appendChild(svgEl('polyline',{points:line.join(' '), fill:'none', stroke:CH_COLORS[c%4], 'stroke-width':1.6}));
  }
  container.innerHTML=''; container.appendChild(svg);
}
const SCATTER_FIELD_LABELS = {
  pc1:'PC1', pc2:'PC2', ptp:'Peak-to-peak', teo_max:'Teager (whole-window max)',
  psi_trough:'Teager (at trough)', teo2:'Teager (lag-2)',
};
let scatterXField = 'pc1', scatterYField = 'ptp';

function renderScatter(container, pts, xField, yField){
  if(!pts){ container.innerHTML = '<div class="no-data">cluster data not yet computed for this unit</div>'; return; }
  const W=300,H=220,padL=8,padR=8,padT=10,padB=10;
  const svg = svgEl('svg',{viewBox:`0 0 ${W} ${H}`});
  let allX=[], allY=[];
  ['noise','other','own'].forEach(cat => { (pts[cat]?.[xField]||[]).forEach(v=>allX.push(v)); (pts[cat]?.[yField]||[]).forEach(v=>allY.push(v)); });
  if(allX.length===0){ container.innerHTML = '<div class="no-data">no events detected on this channel</div>'; return; }
  const xmin=Math.min(...allX), xmax=Math.max(...allX), ymin=Math.min(...allY), ymax=Math.max(...allY);
  const x = v => padL + (v-xmin)/(xmax-xmin+1e-9)*(W-padL-padR);
  const y = v => padT + (1-(v-ymin)/(ymax-ymin+1e-9))*(H-padT-padB);
  const colors = {noise:'#9aacbd', other:'#2f8f57', own:'#c2453f'};
  const order = ['noise','other','own'];
  order.forEach(cat => {
    const p = pts[cat]; if(!p || !p[xField]) return;
    for(let i=0;i<p[xField].length;i++){
      svg.appendChild(svgEl('circle',{cx:x(p[xField][i]), cy:y(p[yField][i]), r: cat==='own'?2.2:1.6,
        fill: colors[cat], opacity: cat==='noise'?0.35:0.75}));
    }
  });
  container.innerHTML=''; container.appendChild(svg);
}

function renderAcg(container, u){
  const hist = u.acg_hist;
  if(!hist || !hist.length){ container.innerHTML = '<div class="no-data">no ACG data for this unit</div>'; return; }
  const W=300,H=220,padL=10,padR=10,padT=10,padB=18;
  const binMs = u.acg_bin_ms, maxLag = u.acg_max_lag_ms;
  const n = hist.length;
  const centers = hist.map((_,i) => -maxLag + (i+0.5)*binMs);
  const svg = svgEl('svg',{viewBox:`0 0 ${W} ${H}`});
  const ymax = Math.max(...hist, 1);
  const x = v => padL + (v+maxLag)/(2*maxLag)*(W-padL-padR);
  const y = v => padT + (1-v/ymax)*(H-padT-padB);
  const barW = (W-padL-padR)/n;
  // shade refractory zone (red-ish) and shoulder zone (blue-ish), matching violation_ratio's own definition
  const refr = u.acg_refractory_ms, shLo = u.acg_shoulder_lo_ms, shHi = u.acg_shoulder_hi_ms;
  svg.appendChild(svgEl('rect',{x:x(-refr), y:padT, width:x(refr)-x(-refr), height:H-padT-padB, fill:'var(--warn)', opacity:0.12}));
  svg.appendChild(svgEl('rect',{x:x(shLo), y:padT, width:x(shHi)-x(shLo), height:H-padT-padB, fill:'var(--accent)', opacity:0.10}));
  svg.appendChild(svgEl('rect',{x:x(-shHi), y:padT, width:x(-shLo)-x(-shHi), height:H-padT-padB, fill:'var(--accent)', opacity:0.10}));
  hist.forEach((v,i) => {
    svg.appendChild(svgEl('rect',{x:x(centers[i]-binMs/2), y:y(v), width:Math.max(barW-0.3,0.3), height:Math.max(H-padB-y(v),0), fill:'var(--text-2)'}));
  });
  svg.appendChild(svgEl('line',{x1:padL,x2:W-padR,y1:H-padB,y2:H-padB,stroke:'var(--border)','stroke-width':1}));
  container.innerHTML=''; container.appendChild(svg);
  const cap = document.createElement('div');
  cap.style.cssText='font-size:9px;color:var(--text-3);margin-top:4px;';
  cap.textContent = `x: lag (ms, ±${maxLag}) · red band = refractory zone (±${refr}ms) · blue bands = shoulder (${shLo}-${shHi}ms) used by violation_ratio`;
  container.appendChild(cap);
}

function percentile(sortedValidVals, p){
  // numpy-style linear-interpolation percentile; input must already be sorted ascending
  const n = sortedValidVals.length;
  if(n===0) return null;
  if(n===1) return sortedValidVals[0];
  const idx = (p/100)*(n-1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  if(lo===hi) return sortedValidVals[lo];
  return sortedValidVals[lo] + (sortedValidVals[hi]-sortedValidVals[lo])*(idx-lo);
}
function unitPercentile(u, field, p){
  const vec = u[field];
  if(!vec) return null;
  const valid = vec.filter(v=>v!=null).sort((a,b)=>a-b);
  if(valid.length < 3) return null;  // same minimum-data bar used server-side
  return percentile(valid, p);
}
const FIELD_LABELS = {
  signalToNoiseRatio:'SNR', rawAmplitude:'Amplitude', violation_ratio:'Violation ratio',
  footprint_concentration_ratio:'Footprint conc.', footprint_flatness_ratio:'Footprint flatness',
  peak_trough_width_ms:'Width',
  mean_rate_hz:'Rate', n_spikes:'N spikes',
  sep_vs_noise_vector:'Sep. vs noise', sep_vs_other_vector:'Sep. vs other',
  isi_amp_corr_all:'ISI-amp corr (all)', isi_amp_corr_burst:'ISI-amp corr (burst)',
};
function getFieldValue(u, field, pctl){
  if(field==='sep_vs_noise_vector' || field==='sep_vs_other_vector') return unitPercentile(u, field, pctl);
  return u[field];
}
let activeFilters = [];  // {field, pctl, op, val}
function passesFilters(u){
  return activeFilters.every(f => {
    const v = getFieldValue(u, f.field, f.pctl);
    if(v==null) return false;
    return f.op==='gt' ? v > f.val : v < f.val;
  });
}
function renderActiveFilters(){
  const el = document.getElementById('activeFilters');
  el.innerHTML = activeFilters.map((f,i) => {
    const label = FIELD_LABELS[f.field] + (f.pctl!=null ? ` (p${f.pctl})` : '');
    const opSym = f.op==='gt' ? '>' : '<';
    return `<span class="chip-toggle"><button class="active" data-idx="${i}" style="cursor:pointer;">${label} ${opSym} ${f.val} &times;</button></span>`;
  }).join(' ');
  el.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    activeFilters.splice(parseInt(b.dataset.idx),1); renderActiveFilters(); renderList();
  }));
}


const SHAPE_COLORS = {decay:'#c2453f', facilitation:'#2f8f57', dip_then_rise:'#3f7fc2', rise_then_dip:'#c2843f', flat:'#9aacbd', insufficient_data:'#9aacbd', not_applicable:'#9aacbd', excluded_high_rate:'#c9c9c9'};
const SHAPE_LABELS = {decay:'decay', facilitation:'facilitation', dip_then_rise:'dip-then-rise', rise_then_dip:'rise-then-dip', flat:'flat', insufficient_data:'insufficient data', not_applicable:'n/a (too few bursts)', excluded_high_rate:'excluded (rate ≥5Hz)'};

function renderBurstCurve(container, curve, shape){
  if(!curve || curve.length===0){
    const msg = shape==='excluded_high_rate'
      ? 'not applicable: firing rate &ge;5Hz &mdash; no reliable silence-then-burst structure at this rate (see conversation)'
      : 'not enough qualifying bursts (short ISI preceded by real silence) to compute a curve';
    container.innerHTML = `<div class="no-data">${msg}</div>`; return;
  }
  const W=300,H=220,padL=10,padR=10,padT=10,padB=22;
  const svg = svgEl('svg',{viewBox:`0 0 ${W} ${H}`});
  const xs = curve.map(c=>c.pos);
  const los = curve.map(c=>c.mean-c.sem), his = curve.map(c=>c.mean+c.sem);
  const xmin=Math.min(...xs), xmax=Math.max(...xs);
  const ymin=Math.min(...los), ymax=Math.max(...his);
  const x = v => padL + (v-xmin)/(xmax-xmin+1e-9)*(W-padL-padR);
  const y = v => padT + (1-(v-ymin)/(ymax-ymin+1e-9))*(H-padT-padB);
  const color = SHAPE_COLORS[shape] || 'var(--accent)';
  curve.forEach(c => {
    svg.appendChild(svgEl('line',{x1:x(c.pos), x2:x(c.pos), y1:y(c.mean-c.sem), y2:y(c.mean+c.sem), stroke:color, 'stroke-width':1}));
  });
  const line = curve.map(c=>`${x(c.pos)},${y(c.mean)}`).join(' ');
  svg.appendChild(svgEl('polyline',{points:line, fill:'none', stroke:color, 'stroke-width':1.6}));
  curve.forEach(c => {
    svg.appendChild(svgEl('circle',{cx:x(c.pos), cy:y(c.mean), r:2.6, fill:color}));
    const t = svgEl('text',{x:x(c.pos), y:H-6, 'font-size':7, 'text-anchor':'middle', fill:'var(--text-3)'});
    t.textContent = c.pos;
    svg.appendChild(t);
  });
  container.innerHTML=''; container.appendChild(svg);
  const cap = document.createElement('div');
  cap.style.cssText='font-size:9px;color:var(--text-3);margin-top:4px;';
  cap.textContent = `x: position in burst (<20ms ISI) · y: mean amplitude ±SEM · n at pos${curve[0].pos}=${curve[0].n}, pos${curve[curve.length-1].pos}=${curve[curve.length-1].n}`;
  container.appendChild(cap);
}

let rawSpikesCache = {};  // uid -> data or null, avoids re-fetching on repeated toggle
const RAW_TRACES_INITIAL_N = 8;  // "Show" reveals this many; "Show more" reveals the rest
const OTHER_UNIT_PALETTE = ['#2f6f9f','#3f8f5f','#b07d2f','#7f4f9f','#3f9f9f','#9f4f7f','#6f7f2f','#9f2f4f'];
function colorForOtherUnit(colorMap, unitId){
  if(!(unitId in colorMap)) colorMap[unitId] = OTHER_UNIT_PALETTE[Object.keys(colorMap).length % OTHER_UNIT_PALETTE.length];
  return colorMap[unitId];
}
function renderRawTracesLegend(container, uid, colorMap){
  container.innerHTML = '';
  container.style.cssText = 'display:flex;flex-wrap:wrap;gap:2px 14px;margin-bottom:8px;';
  const mkItem = (color, label, dashed) => {
    const item = document.createElement('span');
    item.style.cssText = 'display:inline-flex;align-items:center;gap:4px;font-size:9px;color:var(--text-3);';
    const swatch = document.createElement('span');
    swatch.style.cssText = `display:inline-block;width:14px;height:0;border-bottom:2px ${dashed?'dashed':'solid'} ${color};`;
    const txt = document.createElement('span'); txt.textContent = label;
    item.appendChild(swatch); item.appendChild(txt);
    container.appendChild(item);
  };
  mkItem('#c2453f', `unit ${uid} (this unit)`, false);
  Object.entries(colorMap).forEach(([oid,color]) => mkItem(color, `unit ${oid} (other)`, true));
}

function niceTicks(lo, hi, nTicks){
  // simple even-spaced tick generator (not necessarily "round" numbers, but
  // consistent and readable -- matches matplotlib's default tick count roughly)
  const ticks = [];
  for(let i=0;i<nTicks;i++) ticks.push(lo + (hi-lo)*i/(nTicks-1));
  return ticks;
}

function renderTracePair(container, t, padMs, tIdx, nTotal, uid, otherColorMap){
  const trace = t.trace, waveform = t.waveform;
  const W=620,H=170,padL=40,padR=14,padT=10,padB=26;
  const svgT = svgEl('svg',{viewBox:`0 0 ${W} ${H}`,style:'width:100%;height:auto;display:block;'});
  const n = trace.length;
  const ymin=Math.min(...trace), ymax=Math.max(...trace);
  const yPad = (ymax-ymin)*0.08 || 1;
  const x = i => padL + i/(n-1)*(W-padL-padR);
  const y = v => padT + (1-(v-(ymin-yPad))/((ymax+yPad)-(ymin-yPad)+1e-9))*(H-padT-padB);
  const msAt = i => -padMs + (i/(n-1))*(2*padMs);
  const xAtMs = ms => padL + (ms+padMs)/(2*padMs)*(W-padL-padR);
  const mid = Math.floor(n/2);
  // y gridlines/ticks (uV)
  niceTicks(ymin, ymax, 4).forEach(v => {
    svgT.appendChild(svgEl('line',{x1:padL,x2:W-padR,y1:y(v),y2:y(v),stroke:'var(--border)','stroke-width':0.6}));
    const lbl = svgEl('text',{x:padL-5,y:y(v)+3,'font-size':9,'text-anchor':'end',fill:'var(--text-3)'});
    lbl.textContent = v.toFixed(0); svgT.appendChild(lbl);
  });
  // x ticks (ms)
  niceTicks(-padMs, padMs, 7).forEach(ms => {
    const xi = padL + (ms+padMs)/(2*padMs)*(W-padL-padR);
    const lbl = svgEl('text',{x:xi,y:H-padB+14,'font-size':9,'text-anchor':'middle',fill:'var(--text-3)'});
    lbl.textContent = ms.toFixed(0); svgT.appendChild(lbl);
  });
  // one vertical line per spike detected in this window -- red = this unit (uid),
  // dashed colored line + "#<id>" label = a different unit's spike (see WORKFLOW.md
  // 2026-09-16 entry for what "detected in this window" means)
  const spikesHere = (t.spikes && t.spikes.length) ? t.spikes : [{unit_id: uid, offset_ms: 0}];
  spikesHere.forEach(s => {
    const xs = xAtMs(s.offset_ms);
    const isSelf = s.unit_id === uid;
    const color = isSelf ? '#c2453f' : colorForOtherUnit(otherColorMap, s.unit_id);
    svgT.appendChild(svgEl('line',{x1:xs,x2:xs,y1:padT,y2:H-padB,stroke:color,opacity:isSelf?0.65:0.7,'stroke-width':isSelf?1.8:1.4,'stroke-dasharray':isSelf?'none':'3,2'}));
    if(!isSelf){
      const lbl = svgEl('text',{x:xs,y:padT-1,'font-size':8,'text-anchor':'middle',fill:color,'font-weight':600});
      lbl.textContent = `#${s.unit_id}`; svgT.appendChild(lbl);
    }
  });
  let pts=[]; for(let i=0;i<n;i++) pts.push(`${x(i)},${y(trace[i])}`);
  svgT.appendChild(svgEl('polyline',{points:pts.join(' '),fill:'none',stroke:'var(--text)','stroke-width':1}));
  const axLbl1 = svgEl('text',{x:W/2,y:H-4,'font-size':9,'text-anchor':'middle',fill:'var(--text-3)'}); axLbl1.textContent='ms from spike time'; svgT.appendChild(axLbl1);
  const axLbl2 = svgEl('text',{x:10,y:padT+8,'font-size':9,'text-anchor':'start',fill:'var(--text-3)',transform:`rotate(-90,10,${padT+8})`}); axLbl2.textContent='µV'; svgT.appendChild(axLbl2);

  const W2=220,H2=170;
  const svgW = svgEl('svg',{viewBox:`0 0 ${W2} ${H2}`,style:'width:100%;height:auto;display:block;'});
  const nw = waveform.length;
  const ymin2=Math.min(...waveform), ymax2=Math.max(...waveform);
  const yPad2 = (ymax2-ymin2)*0.08 || 1;
  const x2 = i => padL + i/(nw-1)*(W2-padL-padR);
  const y2 = v => padT + (1-(v-(ymin2-yPad2))/((ymax2+yPad2)-(ymin2-yPad2)+1e-9))*(H2-padT-padB);
  niceTicks(ymin2, ymax2, 4).forEach(v => {
    svgW.appendChild(svgEl('line',{x1:padL,x2:W2-padR,y1:y2(v),y2:y2(v),stroke:'var(--border)','stroke-width':0.6}));
    const lbl = svgEl('text',{x:padL-5,y:y2(v)+3,'font-size':9,'text-anchor':'end',fill:'var(--text-3)'});
    lbl.textContent = v.toFixed(0); svgW.appendChild(lbl);
  });
  niceTicks(0, 2, 5).forEach(ms => {
    const xi = padL + (ms/2)*(W2-padL-padR);
    const lbl = svgEl('text',{x:xi,y:H2-padB+14,'font-size':9,'text-anchor':'middle',fill:'var(--text-3)'});
    lbl.textContent = ms.toFixed(1); svgW.appendChild(lbl);
  });
  let pts2=[]; for(let i=0;i<nw;i++) pts2.push(`${x2(i)},${y2(waveform[i])}`);
  svgW.appendChild(svgEl('polyline',{points:pts2.join(' '),fill:'none',stroke:'var(--accent)','stroke-width':1.6}));
  const axLbl3 = svgEl('text',{x:W2/2,y:H2-4,'font-size':9,'text-anchor':'middle',fill:'var(--text-3)'}); axLbl3.textContent='ms (2ms waveform)'; svgW.appendChild(axLbl3);

  const row = document.createElement('div');
  row.style.cssText='border:1px solid var(--border);border-radius:8px;padding:8px;margin-bottom:8px;';
  const caption = document.createElement('div');
  caption.style.cssText='font-size:10px;color:var(--text-3);margin-bottom:4px;font-family:IBM Plex Mono,monospace;';
  const nOther = spikesHere.filter(s => s.unit_id !== uid).length;
  caption.textContent = `spike ${tIdx+1}/${nTotal} · t=${t.t_sec.toFixed(1)}s · solid red = this unit's spike(s)${nOther?` · ${nOther} other-unit spike(s) also in this window`:''}`;
  row.appendChild(caption);
  const flexRow = document.createElement('div');
  flexRow.style.cssText='display:flex;gap:10px;align-items:flex-start;flex-wrap:wrap;';
  const tWrap = document.createElement('div'); tWrap.style.cssText='flex:2;min-width:320px;'; tWrap.appendChild(svgT);
  const wWrap = document.createElement('div'); wWrap.style.cssText='flex:1;min-width:180px;max-width:260px;'; wWrap.appendChild(svgW);
  flexRow.appendChild(tWrap); flexRow.appendChild(wWrap);
  row.appendChild(flexRow);
  container.appendChild(row);
}

async function renderRawSpikesPanel(container, uid){
  container.innerHTML = '<div class="no-data">loading raw traces&hellip;</div>';
  if(!db){ container.innerHTML = '<div class="no-data">shared database unavailable in this view (claude.use("db") did not resolve)</div>'; return; }
  let data = rawSpikesCache[uid];
  let fetchErr = null;
  if(data === undefined){
    try{
      const snap = await db.collection('raw_traces').doc(String(uid)).get();
      data = snap.exists ? snap.data() : null;
    }catch(e){ data = null; fetchErr = e; console.error('raw_traces fetch failed', e); }
    rawSpikesCache[uid] = data;
  }
  if(fetchErr){
    container.innerHTML = `<div class="no-data">error fetching raw traces: ${(fetchErr.code||'')} ${(fetchErr.message||String(fetchErr))}</div>`; return;
  }
  if(!data || !data.traces || !data.traces.length){
    container.innerHTML = `<div class="no-data">no raw trace data for unit ${uid} (doc ${data===null?'does not exist':'has no traces field'})</div>`; return;
  }
  container.innerHTML = '';
  const total = data.traces.length;
  const shownInitially = Math.min(RAW_TRACES_INITIAL_N, total);
  const cap = document.createElement('div');
  cap.style.cssText='font-size:10px;color:var(--text-3);margin-bottom:8px;';
  cap.textContent = `${total} real spikes spread across the full recording (showing ${shownInitially}) · left: ±${data.pad_ms}ms raw trace · right: exact 2ms waveform`;
  container.appendChild(cap);
  const legend = document.createElement('div');
  container.appendChild(legend);
  const otherColorMap = {};
  const list = document.createElement('div');
  let renderErrCount = 0, lastErr = null;
  const renderRange = (fromIdx, toIdx) => {
    for(let i=fromIdx; i<toIdx; i++){
      try{ renderTracePair(list, data.traces[i], data.pad_ms, i, total, uid, otherColorMap); }
      catch(e){ renderErrCount++; lastErr = e; console.error('renderTracePair failed for trace', i, e); }
    }
    renderRawTracesLegend(legend, uid, otherColorMap);
  };
  renderRange(0, shownInitially);
  container.appendChild(list);
  if(total > shownInitially){
    const moreBtn = document.createElement('button');
    moreBtn.textContent = `Show ${total - shownInitially} more traces`;
    moreBtn.style.cssText='font-size:10px;padding:4px 12px;border-radius:6px;border:1px solid var(--accent);background:var(--accent-soft);color:var(--accent);cursor:pointer;font-weight:600;margin-top:6px;';
    moreBtn.addEventListener('click', () => { renderRange(shownInitially, total); moreBtn.remove(); });
    container.appendChild(moreBtn);
  }
  if(renderErrCount>0){
    const errDiv = document.createElement('div');
    errDiv.style.cssText='font-size:10px;color:#c2453f;margin-top:6px;';
    errDiv.textContent = `${renderErrCount} trace(s) failed to render: ${lastErr && lastErr.message}`;
    container.appendChild(errDiv);
  }
}

function renderList(){
  const activeLabels = new Set([...document.querySelectorAll('#labelToggle button.active')].map(b=>b.dataset.kind));
  const q = document.getElementById('searchBox').value.trim();
  const reviewFilter = document.getElementById('reviewFilter').value;
  const shapeFilter = document.getElementById('shapeFilter').value;
  const sortKey = document.getElementById('sortBy').value;

  let list = UNITS.filter(u => activeLabels.has(u.label.replace('+NON-SOMA','')));
  if(q) list = list.filter(u => String(u.unit_id)===q || String(u.trodes_channel)===q);
  if(reviewFilter==='unreviewed') list = list.filter(u => !verdicts[u.unit_id]);
  if(reviewFilter==='reviewed') list = list.filter(u => verdicts[u.unit_id]);
  if(shapeFilter!=='all') list = list.filter(u => u.burst_shape===shapeFilter);
  list = list.filter(passesFilters);

  if(sortKey==='sep_vs_noise_asc') list = [...list].sort((a,b)=>(a.sep_vs_noise??999)-(b.sep_vs_noise??999));
  else if(sortKey==='violation_ratio_desc') list = [...list].sort((a,b)=>(b.violation_ratio??-1)-(a.violation_ratio??-1));
  else if(sortKey==='mean_rate_hz_desc') list = [...list].sort((a,b)=>b.mean_rate_hz-a.mean_rate_hz);
  else list = [...list].sort((a,b)=>a.unit_id-b.unit_id);

  const container = document.getElementById('list');
  container.innerHTML = '';
  list.forEach(u => {
    const core = u.label.replace('+NON-SOMA','');
    const row = document.createElement('div');
    row.className = 'row' + (u.unit_id===selected ? ' selected':'');
    const v = verdicts[u.unit_id];
    row.innerHTML = `
      <span class="verdict-dot ${v?v.verdict:''}"></span>
      <span class="uid">#${u.unit_id}</span>
      <span class="badge ${core}">${core}</span>
      <span class="meta">ch${u.trodes_channel ?? '—'} &middot; ${u.mean_rate_hz.toFixed(1)}Hz</span>
    `;
    row.addEventListener('click', () => { selected = u.unit_id; renderDetail(u.unit_id); renderList(); });
    container.appendChild(row);
  });
  document.getElementById('countNote').textContent = `${list.length} shown`;
}

function reasonText(u){
  const parts = [];
  if(u.noise_reasons) parts.push(`<b>noise:</b> ${u.noise_reasons}`);
  if(u.mua_reasons) parts.push(`<b>mua:</b> ${u.mua_reasons}`);
  if(u.overridden_shape_reasons) parts.push(`<b>shape flags (display-only):</b> ${u.overridden_shape_reasons}`);
  return parts.length ? parts.join('<br>') : '<span style="color:var(--text-3)">no flags</span>';
}

function renderDetail(uid){
  const u = byId[uid];
  const core = u.label.replace('+NON-SOMA','');
  const v = verdicts[uid];
  const el = document.getElementById('detail');
  el.innerHTML = `
    <div class="detail-head">
      <div>
        <div class="detail-uid">Unit #${u.unit_id}</div>
        <div class="detail-badges">
          <span class="badge-lg ${core}">${u.label}</span>
        </div>
      </div>
    </div>
    <div class="chan-info">
      Trodes channel <b>${u.trodes_channel ?? '—'}</b> &middot; KS channel ${u.peak_channel} &middot;
      co-located units: ${(u.co_located_units||[]).length ? u.co_located_units.map(o=>'#'+o).join(', ') : 'none'}
    </div>
    <div class="panels">
      <div class="panel"><div class="panel-label">KS waveform (top 4 channels)</div><div class="wave"></div></div>
      <div class="panel">
        <div class="panel-label" style="display:flex;align-items:center;justify-content:space-between;gap:6px;">
          <span>Raw-data scatter: own (red) / co-located (green) / noise (gray)</span>
          <span style="display:flex;gap:4px;text-transform:none;">
            <select id="scatterXSel" style="font-size:9px;"></select>
            <span style="color:var(--text-3);">vs</span>
            <select id="scatterYSel" style="font-size:9px;"></select>
          </span>
        </div>
        <div class="clus"></div>
      </div>
      <div class="panel"><div class="panel-label">Autocorrelogram</div><div class="acg"></div></div>
      <div class="panel"><div class="panel-label">Amplitude vs. position in burst (shape: <span style="color:${SHAPE_COLORS[u.burst_shape]||'inherit'}">${SHAPE_LABELS[u.burst_shape]||u.burst_shape}</span>)</div><div class="burstcurve"></div></div>
    </div>
    <div class="panel" style="margin-bottom:16px;">
      <div class="panel-label" style="display:flex;align-items:center;justify-content:space-between;">
        <span>Raw spike traces (verify shape in raw data)</span>
        <button id="rawSpikesToggle" style="font-size:10px;padding:3px 10px;border-radius:6px;border:1px solid var(--accent);background:var(--accent-soft);color:var(--accent);cursor:pointer;font-weight:600;">Show</button>
      </div>
      <div class="rawspikes" style="display:none;margin-top:10px;"></div>
    </div>
    <div class="metrics-grid">
      <div class="metric"><div class="metric-label">SNR</div><div class="metric-val">${u.signalToNoiseRatio}</div></div>
      <div class="metric"><div class="metric-label">Amplitude</div><div class="metric-val">${u.rawAmplitude}&micro;V</div></div>
      <div class="metric"><div class="metric-label">Violation ratio</div><div class="metric-val">${u.violation_ratio ?? '—'}</div></div>
      <div class="metric"><div class="metric-label">Footprint conc.</div><div class="metric-val">${u.footprint_concentration_ratio}</div></div>
      <div class="metric"><div class="metric-label">Footprint flatness</div><div class="metric-val">${u.footprint_flatness_ratio ?? '—'}</div></div>
      <div class="metric"><div class="metric-label">Width</div><div class="metric-val">${u.peak_trough_width_ms}ms</div></div>
      <div class="metric"><div class="metric-label">Rate</div><div class="metric-val">${u.mean_rate_hz}Hz</div></div>
      <div class="metric"><div class="metric-label">n spikes</div><div class="metric-val">${u.n_spikes.toLocaleString()}</div></div>
      <div class="metric"><div class="metric-label">Sep. vs noise (p80)</div><div class="metric-val">${u.sep_vs_noise!=null?u.sep_vs_noise.toFixed(2):'—'}</div></div>
      <div class="metric"><div class="metric-label">Sep. vs other (p80)</div><div class="metric-val">${u.sep_vs_other!=null?u.sep_vs_other.toFixed(2):'—'}</div></div>
      <div class="metric"><div class="metric-label">ISI-amp corr (all)</div><div class="metric-val">${u.isi_amp_corr_all!=null?u.isi_amp_corr_all:'—'}</div></div>
      <div class="metric"><div class="metric-label">ISI-amp corr (burst&lt;20ms)</div><div class="metric-val">${u.isi_amp_corr_burst!=null?u.isi_amp_corr_burst:'—'}</div></div>
      <div class="metric"><div class="metric-label">Burst shape</div><div class="metric-val" style="color:${SHAPE_COLORS[u.burst_shape]||'inherit'}">${SHAPE_LABELS[u.burst_shape]||u.burst_shape}</div></div>
    </div>
    <div class="reasons">
      ${u.sep_vs_noise_vector ? `vs noise, all 10 subsessions (${u.n_valid_noise}/10 valid): <span class="mono">[${u.sep_vs_noise_vector.map(v=>v==null?'—':v).join(', ')}]</span><br>` : ''}
      ${u.sep_vs_other_vector ? `vs other units, all 10 subsessions (${u.n_valid_other}/10 valid): <span class="mono">[${u.sep_vs_other_vector.map(v=>v==null?'—':v).join(', ')}]</span>` : ''}
    </div>
    <div class="reasons">${reasonText(u)}</div>
    <div class="verdict-panel">
      <div class="verdict-title">Your verdict</div>
      <div class="verdict-btns">
        <button class="vbtn ${v?.verdict==='good'?'picked-good':''}" data-v="good">GOOD <span class="vbtn-key">G</span></button>
        <button class="vbtn ${v?.verdict==='mua'?'picked-mua':''}" data-v="mua">MUA <span class="vbtn-key">M</span></button>
        <button class="vbtn ${v?.verdict==='noise'?'picked-noise':''}" data-v="noise">NOISE <span class="vbtn-key">N</span></button>
      </div>
      <div class="nav-hint">Shortcuts: G/M/N to set verdict &middot; J/K or &darr;/&uarr; to move to next/prev unit</div>
      <div class="db-status" id="dbStatusText">syncing...</div>
    </div>
  `;
  renderWaveform(el.querySelector('.wave'), u.waveform_top4);
  const scatterXSel = el.querySelector('#scatterXSel'), scatterYSel = el.querySelector('#scatterYSel');
  const fieldOptions = Object.keys(SCATTER_FIELD_LABELS).map(f => `<option value="${f}">${SCATTER_FIELD_LABELS[f]}</option>`).join('');
  scatterXSel.innerHTML = fieldOptions; scatterYSel.innerHTML = fieldOptions;
  scatterXSel.value = scatterXField; scatterYSel.value = scatterYField;
  renderScatter(el.querySelector('.clus'), u.cluster_points, scatterXField, scatterYField);
  scatterXSel.addEventListener('change', () => { scatterXField = scatterXSel.value; renderScatter(el.querySelector('.clus'), u.cluster_points, scatterXField, scatterYField); });
  scatterYSel.addEventListener('change', () => { scatterYField = scatterYSel.value; renderScatter(el.querySelector('.clus'), u.cluster_points, scatterXField, scatterYField); });
  renderAcg(el.querySelector('.acg'), u);
  renderBurstCurve(el.querySelector('.burstcurve'), u.burst_curve, u.burst_shape);
  el.querySelectorAll('.vbtn').forEach(b => b.addEventListener('click', () => setVerdict(uid, b.dataset.v)));
  if(db) document.getElementById('dbStatusText').textContent = 'saved to shared review database';

  const rawToggleBtn = el.querySelector('#rawSpikesToggle');
  const rawContainer = el.querySelector('.rawspikes');
  rawToggleBtn.addEventListener('click', () => {
    const visible = rawContainer.style.display !== 'none';
    if(visible){
      rawContainer.style.display = 'none';
      rawContainer.innerHTML = '';
      rawToggleBtn.textContent = 'Show';
    } else {
      rawContainer.style.display = 'block';
      rawToggleBtn.textContent = 'Hide';
      renderRawSpikesPanel(rawContainer, uid);
    }
  });
}

document.querySelectorAll('#labelToggle button').forEach(btn => {
  btn.addEventListener('click', () => { btn.classList.toggle('active'); renderList(); });
});
document.getElementById('sortBy').addEventListener('change', renderList);
document.getElementById('reviewFilter').addEventListener('change', renderList);
document.getElementById('shapeFilter').addEventListener('change', renderList);
document.getElementById('searchBox').addEventListener('input', renderList);
function updatePctlVisibility(){
  const f = document.getElementById('filterField').value;
  document.getElementById('filterPctlWrap').style.display =
    (f==='sep_vs_noise_vector' || f==='sep_vs_other_vector') ? 'inline' : 'none';
}
document.getElementById('filterField').addEventListener('change', updatePctlVisibility);
updatePctlVisibility();
document.getElementById('filterAdd').addEventListener('click', () => {
  const field = document.getElementById('filterField').value;
  const op = document.getElementById('filterOp').value;
  const valRaw = document.getElementById('filterVal').value;
  if(valRaw === '') return;
  const isPctlField = field==='sep_vs_noise_vector' || field==='sep_vs_other_vector';
  activeFilters.push({
    field, op, val: parseFloat(valRaw),
    pctl: isPctlField ? parseFloat(document.getElementById('filterPctl').value) : null,
  });
  document.getElementById('filterVal').value = '';
  renderActiveFilters(); renderList();
});

document.addEventListener('keydown', (e) => {
  if(document.activeElement.tagName==='INPUT') return;
  const k = e.key.toLowerCase();
  if(k==='g') setVerdict(selected,'good');
  else if(k==='m') setVerdict(selected,'mua');
  else if(k==='n') setVerdict(selected,'noise');
  else if(k==='j' || e.key==='ArrowDown'){
    const rows=[...document.querySelectorAll('.row')];
    const idx=rows.findIndex(r=>r.classList.contains('selected'));
    if(idx>=0 && idx<rows.length-1) rows[idx+1].click();
  } else if(k==='k' || e.key==='ArrowUp'){
    const rows=[...document.querySelectorAll('.row')];
    const idx=rows.findIndex(r=>r.classList.contains('selected'));
    if(idx>0) rows[idx-1].click();
  }
});

renderList();
renderDetail(selected);
initDb();
</script>
"""

html = html.replace("__UNITS_JSON__", json.dumps(units))
html = html.replace("Unit Review Pipeline &middot; 20260901_085606", f"Unit Review Pipeline &middot; {SESSION_ID}")
html = html.replace("<title>Unit Review Pipeline</title>", f"<title>Unit Review {SESSION_ID}</title>")

out_path = OUT_PATH
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print("wrote", out_path, len(html), "bytes")


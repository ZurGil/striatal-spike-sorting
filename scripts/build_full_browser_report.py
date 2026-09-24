import json

with open(r"D:\Gil\spike_sorting_agent\outputs\full_unit_browser_data.json") as f:
    units = json.load(f)

REASON_LABELS = {
    "footprint_cliff": "footprint cliff",
    "width_outlier": "width outlier",
    "violation_ratio": "elevated ACG",
    "spikesMissing": "spikes missing",
    "nSpikes": "too few spikes",
    "presenceRatio": "low presence",
    "amplitude": "low amplitude",
    "SNR": "low SNR",
    "nPeaks_nan": "no peaks",
    "too_many_peaks": "too many peaks",
    "too_many_troughs": "too many troughs",
    "duration_too_short": "duration too short",
    "duration_too_long": "duration too long",
    "baseline_not_flat": "baseline not flat",
    "2nd_peak_too_big": "2nd peak too big",
    "spatial_decay_abnormal": "spatial decay abnormal",
}

html = r"""<title>Full Unit Browser</title>
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
body{background:var(--bg);color:var(--text);font-family:'Archivo',system-ui,sans-serif;margin:0;padding:0 20px 64px;}
.mono{font-family:'IBM Plex Mono',ui-monospace,monospace;font-variant-numeric:tabular-nums;}
h1{font-size:26px;font-weight:700;letter-spacing:-.01em;text-wrap:balance;margin:0;}
.wrap{max-width:1600px;margin:0 auto;}
.top{padding:28px 0 14px;border-bottom:1px solid var(--border);}
.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin-bottom:8px;}
.sub{color:var(--text-2);font-size:13.5px;max-width:82ch;margin-top:8px;line-height:1.55;}

.stats{display:flex;gap:14px;margin:16px 0;flex-wrap:wrap;}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:8px 14px;font-size:12px;}
.stat b{font-family:'IBM Plex Mono',monospace;font-size:15px;display:block;color:var(--text);}

.controls{position:sticky;top:0;z-index:10;background:var(--bg);padding:12px 0;border-bottom:1px solid var(--border);
  display:flex;gap:10px;flex-wrap:wrap;align-items:center;}
.controls label{font-size:11px;color:var(--text-3);text-transform:uppercase;letter-spacing:.04em;margin-right:5px;}
.controls select, .controls input{background:var(--surface);border:1px solid var(--border);color:var(--text);
  border-radius:6px;padding:5px 9px;font-size:12.5px;font-family:'Archivo',sans-serif;}
.controls input[type=text]{width:110px;}
.chip-toggle{display:flex;gap:5px;}
.chip-toggle button{background:var(--surface);border:1px solid var(--border);color:var(--text-2);border-radius:20px;
  padding:4px 12px;font-size:11.5px;cursor:pointer;font-family:'IBM Plex Mono',monospace;}
.chip-toggle button.active[data-kind=GOOD]{background:var(--good-soft);color:var(--good);border-color:transparent;}
.chip-toggle button.active[data-kind=MUA]{background:var(--warn-soft);color:var(--warn);border-color:transparent;}
.chip-toggle button.active[data-kind=NOISE]{background:var(--bad-soft);color:var(--bad);border-color:transparent;}
.count-note{font-size:12px;color:var(--text-3);margin-left:auto;}

.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(258px,1fr));gap:12px;margin-top:16px;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:9px;padding:11px 12px;}
.card-head{display:flex;justify-content:space-between;align-items:flex-start;gap:6px;}
.uid{font-family:'IBM Plex Mono',monospace;font-weight:700;font-size:14px;}
.badges{display:flex;gap:4px;flex-wrap:wrap;margin-top:4px;}
.badge{font-size:9.5px;font-family:'IBM Plex Mono',monospace;padding:1.5px 6px;border-radius:10px;font-weight:600;}
.badge.GOOD{background:var(--good-soft);color:var(--good);}
.badge.MUA{background:var(--warn-soft);color:var(--warn);}
.badge.NOISE{background:var(--bad-soft);color:var(--bad);}
.badge.soma{background:var(--surface-2);color:var(--text-2);border:1px solid var(--border);}
.badge.ct{background:var(--surface-2);color:var(--violet);border:1px solid var(--border);}
.card svg{width:100%;height:auto;display:block;margin-top:6px;}
.reasons{margin-top:6px;display:flex;gap:3px;flex-wrap:wrap;}
.reason{font-size:9px;background:var(--bad-soft);color:var(--bad);padding:1px 5px;border-radius:4px;}
.reason.mua{background:var(--warn-soft);color:var(--warn);}
.metrics{margin-top:6px;font-size:9.5px;color:var(--text-2);font-family:'IBM Plex Mono',monospace;line-height:1.5;}
.metrics b{color:var(--text);}
.hidden{display:none!important;}
.footer{margin-top:36px;padding-top:16px;border-top:1px solid var(--border);color:var(--text-3);font-size:11.5px;line-height:1.7;}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">

<div class="wrap">
  <div class="top">
    <div class="eyebrow">Full Unit Browser &middot; 401 units &middot; 20260901_085606</div>
    <h1>Every unit: waveform, category, and why</h1>
    <div class="sub">Category = bombcell's shape-based noise/non-soma criteria + our own <code>violation_ratio</code>
    (replacing bombcell's broken RPV check) + footprint-concentration/width outlier checks. Non-soma is shown as a
    tag, not a rejection -- it's driven purely by peak-first waveform polarity and includes real units (our clean
    reference units 285/295 and the likely-TAN unit 30 are both tagged non-soma). <b>The peak/trough-COUNT noise
    rule (nPeaks/nTroughs) only counts if corroborated by an actual quality problem</b> (poor SNR/amplitude,
    elevated contamination, or a footprint cliff) -- 32 units were rejected by that rule alone despite strong SNR,
    amplitude, ACG, and footprint (unit 59: SNR=26, amplitude=106&micro;V, rejected only for having 2 troughs
    instead of &le;1) and are now promoted, marked "shape-count overridden." Cell type is bombcell's rule-based
    MSN/FSI/TAN/UIN classifier (waveform duration + post-spike suppression + proportion of long ISIs) -- unreliable
    so far (only 2 TANs found session-wide, missing our best candidate), included for reference only.</div>
    <div class="stats" id="stats"></div>
  </div>

  <div class="controls">
    <div class="chip-toggle" id="labelToggle">
      <label style="align-self:center;">Label</label>
      <button data-kind="GOOD" class="active">GOOD</button>
      <button data-kind="MUA" class="active">MUA</button>
      <button data-kind="NOISE" class="active">NOISE</button>
    </div>
    <div>
      <label>Non-soma</label>
      <select id="nonSomaFilter">
        <option value="all">show all</option>
        <option value="only">only non-soma</option>
        <option value="hide">hide non-soma</option>
      </select>
    </div>
    <div>
      <label>Cell type</label>
      <select id="cellTypeFilter">
        <option value="all">all</option>
        <option value="MSN">MSN</option>
        <option value="FSI">FSI</option>
        <option value="TAN">TAN</option>
        <option value="UIN">UIN</option>
        <option value="Unknown">Unknown</option>
      </select>
    </div>
    <div>
      <label>Sort</label>
      <select id="sortBy">
        <option value="unit_id">unit ID</option>
        <option value="violation_ratio_desc">ACG violation (high&rarr;low)</option>
        <option value="peak_trough_width_ms_desc">width (wide&rarr;narrow)</option>
        <option value="footprint_concentration_ratio_asc">footprint concentration (cliff first)</option>
        <option value="n_spikes_desc">spike count (high&rarr;low)</option>
        <option value="mean_rate_hz_desc">firing rate (high&rarr;low)</option>
      </select>
    </div>
    <div>
      <label>Unit ID</label>
      <input type="text" id="searchBox" placeholder="e.g. 187">
    </div>
    <div>
      <label>Channel</label>
      <input type="text" id="channelBox" placeholder="trodes id, e.g. 1105">
    </div>
    <div class="count-note" id="countNote"></div>
  </div>

  <div class="grid" id="grid"></div>

  <div class="footer">
    Waveform: KS template on the top 4 footprint channels by peak-to-peak amplitude (not raw-averaged -- more
    reliable, see conversation). violation_ratio: ACG-shoulder-normalized refractory metric, our replacement for
    bombcell's RPV (0.06-0.22 = clean reference range, 0.59-0.94 = known-contaminated range, calibrated this
    session). footprint_conc.: 2nd-strongest footprint channel's amplitude &divide; peak channel's (near 1 = smooth
    real spread, near 0 = single-channel-isolated). trodes ch: the Trodes NTrode id for this unit's peak channel --
    open the raw file in Trodes and look up this channel to inspect the actual trace yourself.
  </div>
</div>

<script>
const UNITS = __UNITS_JSON__;
const REASON_LABELS = __REASON_LABELS_JSON__;
const CH_COLORS = ['var(--accent)','var(--violet)','var(--warn)','var(--blue)'];

function svgEl(tag, attrs){
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for(const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}
function renderWaveform(container, templ){
  const W=230,H=90,padL=4,padR=4,padT=6,padB=6;
  const nt = templ.length;
  const svg = svgEl('svg',{viewBox:`0 0 ${W} ${H}`});
  let ymin=Infinity, ymax=-Infinity;
  for(let s=0;s<nt;s++) for(let c=0;c<4;c++){ ymin=Math.min(ymin,templ[s][c]); ymax=Math.max(ymax,templ[s][c]); }
  const x = s => padL + (s/(nt-1))*(W-padL-padR);
  const y = v => padT + (1-(v-ymin)/(ymax-ymin+1e-9))*(H-padT-padB);
  svg.appendChild(svgEl('line',{x1:padL,x2:W-padR,y1:y(0),y2:y(0),stroke:'var(--border)','stroke-width':1}));
  for(let c=0;c<4;c++){
    let line=[];
    for(let s=0;s<nt;s++) line.push(`${x(s)},${y(templ[s][c])}`);
    svg.appendChild(svgEl('polyline',{points:line.join(' '), fill:'none', stroke:CH_COLORS[c], 'stroke-width':1.4}));
  }
  container.appendChild(svg);
}

function reasonChips(mua_reasons, noise_reasons, isNoise){
  const all = [];
  (noise_reasons||'').split(',').filter(Boolean).forEach(r => all.push({r, cls:''}));
  if(!isNoise) (mua_reasons||'').split(',').filter(Boolean).forEach(r => all.push({r, cls:'mua'}));
  return all.map(({r,cls}) => `<span class="reason ${cls}">${REASON_LABELS[r]||r}</span>`).join('');
}

function renderCard(u){
  const card = document.createElement('div'); card.className='card';
  card.dataset.label = u.is_noise ? 'NOISE' : (u.is_mua ? 'MUA' : 'GOOD');
  card.dataset.nonsoma = u.is_non_soma ? '1' : '0';
  card.dataset.celltype = u.cell_type;
  card.dataset.unitId = u.unit_id;
  card.dataset.trodesId = u.trodes_ntrode_id ?? '';
  card.dataset.peakChannel = u.peak_channel;

  const coreLabel = u.is_noise ? 'NOISE' : (u.is_mua ? 'MUA' : 'GOOD');
  card.innerHTML = `
    <div class="card-head">
      <div>
        <div class="uid">#${u.unit_id}</div>
        <div class="badges">
          <span class="badge ${coreLabel}">${coreLabel}</span>
          ${u.is_non_soma ? '<span class="badge soma">non-soma</span>' : ''}
          <span class="badge ct">${u.cell_type}</span>
        </div>
      </div>
    </div>
    <div class="wave"></div>
    <div class="reasons">${reasonChips(u.mua_reasons, u.noise_reasons, u.is_noise)}
      ${u.overridden_count_reasons ? `<span class="reason" style="background:var(--accent-soft);color:var(--accent);" title="Rejected by bombcell's peak/trough-count rule alone, but SNR/amplitude/ACG/footprint all check out -- see conversation on unit 59">shape-count overridden (strong signal)</span>` : ''}
    </div>
    <div class="metrics">
      viol.ratio <b>${u.violation_ratio!=null?u.violation_ratio.toFixed(3):'—'}</b> ·
      conc. <b>${u.footprint_concentration_ratio!=null?u.footprint_concentration_ratio.toFixed(2):'—'}</b> ·
      width <b>${u.peak_trough_width_ms!=null?u.peak_trough_width_ms.toFixed(2)+'ms':'—'}</b><br>
      n=<b>${u.n_spikes.toLocaleString()}</b> · rate <b>${u.mean_rate_hz.toFixed(2)}Hz</b> ·
      trodes ch <b>${u.trodes_ntrode_id ?? '—'}</b>
    </div>
  `;
  renderWaveform(card.querySelector('.wave'), u.ks_template);
  return card;
}

const grid = document.getElementById('grid');
const cardEls = [];
UNITS.forEach(u => {
  const el = renderCard(u);
  grid.appendChild(el);
  cardEls.push(el);
});

// stats
const statsEl = document.getElementById('stats');
const counts = {GOOD:0, MUA:0, NOISE:0};
UNITS.forEach(u => counts[u.is_noise?'NOISE':(u.is_mua?'MUA':'GOOD')]++);
statsEl.innerHTML = `
  <div class="stat"><b>${UNITS.length}</b>total units</div>
  <div class="stat" style="color:var(--good)"><b>${counts.GOOD}</b>good</div>
  <div class="stat" style="color:var(--warn)"><b>${counts.MUA}</b>mua</div>
  <div class="stat" style="color:var(--bad)"><b>${counts.NOISE}</b>noise</div>
`;

// filters
const activeLabels = new Set(['GOOD','MUA','NOISE']);
document.querySelectorAll('#labelToggle button').forEach(btn => {
  btn.addEventListener('click', () => {
    const k = btn.dataset.kind;
    if(activeLabels.has(k)){ activeLabels.delete(k); btn.classList.remove('active'); }
    else { activeLabels.add(k); btn.classList.add('active'); }
    applyFilters();
  });
});
const nonSomaFilter = document.getElementById('nonSomaFilter');
const cellTypeFilter = document.getElementById('cellTypeFilter');
const sortBy = document.getElementById('sortBy');
const searchBox = document.getElementById('searchBox');
const channelBox = document.getElementById('channelBox');
const countNote = document.getElementById('countNote');
[nonSomaFilter, cellTypeFilter, sortBy].forEach(el => el.addEventListener('change', applyFilters));
searchBox.addEventListener('input', applyFilters);
channelBox.addEventListener('input', applyFilters);

function applyFilters(){
  const ns = nonSomaFilter.value;
  const ct = cellTypeFilter.value;
  const q = searchBox.value.trim();
  const chq = channelBox.value.trim();
  let visible = 0;
  cardEls.forEach(el => {
    let show = activeLabels.has(el.dataset.label);
    if(show && ns==='only') show = el.dataset.nonsoma==='1';
    if(show && ns==='hide') show = el.dataset.nonsoma==='0';
    if(show && ct!=='all') show = el.dataset.celltype===ct;
    if(show && q) show = el.dataset.unitId===q;
    if(show && chq) show = el.dataset.trodesId===chq || el.dataset.peakChannel===chq;
    el.classList.toggle('hidden', !show);
    if(show) visible++;
  });
  countNote.textContent = `${visible} / ${UNITS.length} shown`;

  const sortKey = sortBy.value;
  if(sortKey !== 'unit_id'){
    const [field, dir] = sortKey.includes('_desc') ? [sortKey.replace('_desc',''), -1] : [sortKey.replace('_asc',''), 1];
    const byId = Object.fromEntries(UNITS.map(u=>[u.unit_id, u]));
    const sorted = [...cardEls].sort((a,b) => {
      const ua = byId[a.dataset.unitId][field], ub = byId[b.dataset.unitId][field];
      const va = (ua==null) ? -Infinity : ua, vb = (ub==null) ? -Infinity : ub;
      return dir * (va - vb);
    });
    sorted.forEach(el => grid.appendChild(el));
  } else {
    [...cardEls].sort((a,b)=>+a.dataset.unitId-+b.dataset.unitId).forEach(el => grid.appendChild(el));
  }
}
applyFilters();
</script>
"""

html = html.replace("__UNITS_JSON__", json.dumps(units))
html = html.replace("__REASON_LABELS_JSON__", json.dumps(REASON_LABELS))

out_path = r"D:\Gil\spike_sorting_agent\outputs\full_unit_browser.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print("wrote", out_path, len(html), "bytes")

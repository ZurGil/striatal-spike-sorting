import json

with open(r"D:\Gil\spike_sorting_agent\outputs\random_pile_set.json") as f:
    units = json.load(f)

html = r"""<title>Random Artifact Pile</title>
<style>
:root{
  --bg:#f4f6f8; --surface:#ffffff; --surface-2:#eef1f4; --border:#d7dee4;
  --text:#12181f; --text-2:#445261; --text-3:#7c8b9a;
  --accent:#0f8b83; --accent-soft:#d7f0ee;
  --warn:#b5790a; --warn-soft:#fbeed2;
  --bad:#c2453f; --bad-soft:#fbe2e0;
  --good:#2f8f57; --good-soft:#ddf3e4;
  --violet:#7a54a8; --blue:#3f6fd1;
  --shoulder:#e4ebf0;
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
    --shoulder:#1c2530;
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
  --shoulder:#1c2530;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'Archivo',system-ui,sans-serif;margin:0;padding:0 24px 64px;}
.mono{font-family:'IBM Plex Mono',ui-monospace,monospace;font-variant-numeric:tabular-nums;}
h1,h3{text-wrap:balance;font-weight:700;margin:0;}
.wrap{max-width:1240px;margin:0 auto;}
.top{padding:40px 0 20px;border-bottom:1px solid var(--border);}
.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin-bottom:10px;}
h1{font-size:32px;letter-spacing:-.01em;}
.sub{color:var(--text-2);font-size:15px;max-width:70ch;margin-top:10px;line-height:1.55;}
.banner{margin:24px 0;padding:20px 22px;border-radius:10px;background:var(--warn-soft);border:1px solid color-mix(in srgb, var(--warn) 40%, transparent);}
.banner h3{font-size:14px;color:var(--warn);text-transform:uppercase;letter-spacing:.05em;font-family:'IBM Plex Mono',monospace;margin-bottom:10px;}
.banner p{margin:0 0 8px;font-size:14px;line-height:1.6;color:var(--text);}
.banner p:last-child{margin-bottom:0}
.banner b{color:var(--text)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:20px;margin-top:28px;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px 22px 22px;}
.card-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:4px;}
.uid{font-size:20px;font-weight:700;font-family:'IBM Plex Mono',monospace;}
.chips{display:flex;gap:6px;margin-top:6px;flex-wrap:wrap;}
.chip{font-size:11px;font-family:'IBM Plex Mono',monospace;padding:2px 8px;border-radius:20px;border:1px solid var(--border);color:var(--text-2);}
.chip.bad{background:var(--bad-soft);color:var(--bad);border:none;font-weight:600;}
.score-big{font-family:'IBM Plex Mono',monospace;font-size:28px;font-weight:700;text-align:right;line-height:1;color:var(--bad);}
.score-label{font-size:10px;color:var(--text-3);text-transform:uppercase;letter-spacing:.05em;text-align:right;margin-top:4px;}
.comp-bars{margin-top:14px;}
.comp-row{display:flex;align-items:center;gap:10px;margin-bottom:5px;font-size:11px;}
.comp-row .lbl{width:88px;color:var(--text-3);font-family:'IBM Plex Mono',monospace;flex-shrink:0;}
.comp-row .track{flex:1;height:6px;background:var(--surface-2);border-radius:3px;overflow:hidden;}
.comp-row .fill{height:100%;background:var(--accent);border-radius:3px;}
.comp-row .val{width:34px;text-align:right;font-family:'IBM Plex Mono',monospace;color:var(--text-2);flex-shrink:0;}
.chartrow{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:16px;}
.chartbox{background:var(--surface-2);border-radius:8px;padding:10px 12px 8px;}
.chartbox h4{font-size:11px;color:var(--text-3);text-transform:uppercase;letter-spacing:.05em;font-weight:600;margin-bottom:4px;font-family:'IBM Plex Mono',monospace;}
.chartbox svg{width:100%;height:auto;display:block;}
.metrics{margin-top:16px;display:grid;grid-template-columns:repeat(4,1fr);gap:10px 16px;font-size:12px;}
.metric .k{color:var(--text-3);font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;}
.metric .v{font-family:'IBM Plex Mono',monospace;font-size:14px;margin-top:2px;color:var(--text);}
.metric .v.bad{color:var(--bad)}
.footer{margin-top:48px;padding-top:20px;border-top:1px solid var(--border);color:var(--text-3);font-size:12px;line-height:1.7;}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">

<div class="wrap">
  <div class="top">
    <div class="eyebrow">Random Sample &middot; Flagged Pile &middot; 20260901_085606</div>
    <h1>8 random units from the artifact pile</h1>
    <div class="sub">Drawn at random (seed=7) from 70 units flagged <code>likely_shared_artifact</code> out of a
    120-unit random sample across the full 401-unit session (58.3% flagged rate, matching the earlier biased
    estimate of 60.7% -- this is a real session-wide rate, not an ID-ordering artifact).</div>
  </div>

  <div class="banner">
    <h3>&#9888; A different pattern than 187/156/151</h3>
    <p>These 8 are flagged purely on <b>cross-unit ACG correlation</b> (r=0.90&ndash;0.98) &mdash; but unlike
    187/156/151, most do <b>not</b> show the extreme footprint cliff (concentration ratios here are 0.41&ndash;0.95,
    mostly normal-looking) or the obviously broadened, non-biological waveform width (0.33&ndash;0.47ms, close to
    the clean-reference range of 0.37&ndash;0.43ms). The distances to their best-match partner are large &mdash;
    441&ndash;3620&micro;m, in one case nearly the full probe length &mdash; which is hard to explain as a real
    neuron's own bursting synchrony.</p>
    <p>This is worth your own judgment call: does a high cross-unit ACG match on its own (without the other two
    corroborating signs) mean the same thing here as it did for 187/156/151, or could some of these be real,
    separate neurons that happen to share timing structure with something else (e.g. a shared behavioral/movement
    trigger, or genuine network synchrony)? The wide range of distances (up to 3.6mm) argues against ordinary local
    network synchrony and toward something recorded broadly across the whole probe.</p>
  </div>

  <div class="grid" id="grid"></div>

  <div class="footer">
    Waveform panel shows KS's own template (top 4 footprint channels by peak-to-peak amplitude).
    Structural score capped at 0.15 for all 8 per the cross-unit-ACG hard-flag rule in structural_score.py.
    Generated by scripts/random_pile_enrich.py + scripts/build_random_pile_report.py.
  </div>
</div>

<script>
const UNITS = __UNITS_JSON__;
const CH_COLORS = ['var(--accent)','var(--violet)','var(--warn)','var(--blue)'];
function svgEl(tag, attrs){
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for(const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}
function renderWaveform(container, unit){
  const W=260,H=150,padL=34,padR=8,padT=8,padB=28;
  const templ = unit.ks_template;
  const nt = templ.length;
  const svg = svgEl('svg',{viewBox:`0 0 ${W} ${H}`});
  let ymin=Infinity, ymax=-Infinity;
  for(let s=0;s<nt;s++) for(let c=0;c<4;c++){ ymin=Math.min(ymin,templ[s][c]); ymax=Math.max(ymax,templ[s][c]); }
  const x = s => padL + (s/(nt-1))*(W-padL-padR);
  const y = v => padT + (1-(v-ymin)/(ymax-ymin))*(H-padT-padB);
  svg.appendChild(svgEl('line',{x1:padL,x2:W-padR,y1:y(0),y2:y(0),stroke:'var(--border)','stroke-width':1}));
  for(let c=0;c<4;c++){
    let line=[];
    for(let s=0;s<nt;s++) line.push(`${x(s)},${y(templ[s][c])}`);
    svg.appendChild(svgEl('polyline',{points:line.join(' '), fill:'none', stroke:CH_COLORS[c], 'stroke-width':1.7}));
  }
  const meta = svgEl('text',{x:padL,y:H-4,'font-size':8.5,fill:'var(--text-2)','font-family':'IBM Plex Mono, monospace','font-weight':600});
  meta.textContent = `peak↔trough ${unit.ks_peak_trough_width_ms.toFixed(2)}ms · amp ${unit.ks_template_amplitude.toFixed(1)}` + (unit.ks_trough_before_peak ? ' · trough-first' : '');
  svg.appendChild(meta);
  container.appendChild(svg);
}
function renderACG(container, unit){
  const W=260,H=150,padL=8,padR=8,padT=8,padB=20;
  const centers = unit.acg_bin_centers_ms, counts = unit.acg_counts;
  const svg = svgEl('svg',{viewBox:`0 0 ${W} ${H}`});
  const maxC = Math.max(...counts, 1);
  const x = ms => padL + ((ms+30)/60)*(W-padL-padR);
  const y = c => padT + (1-c/maxC)*(H-padT-padB);
  const barW = (W-padL-padR)/centers.length;
  const refr = svgEl('rect',{x:x(-1.5),y:padT,width:x(1.5)-x(-1.5),height:H-padT-padB,fill:'var(--bad)',opacity:0.14});
  const sh1 = svgEl('rect',{x:x(-25),y:padT,width:x(-5)-x(-25),height:H-padT-padB,fill:'var(--shoulder)'});
  const sh2 = svgEl('rect',{x:x(5),y:padT,width:x(25)-x(5),height:H-padT-padB,fill:'var(--shoulder)'});
  svg.appendChild(sh1); svg.appendChild(sh2); svg.appendChild(refr);
  centers.forEach((c,i)=>{
    const h = (H-padT-padB) - (y(counts[i])-padT);
    svg.appendChild(svgEl('rect',{x:x(c)-barW/2, y:y(counts[i]), width:Math.max(barW-0.4,0.6), height:h, fill:'var(--text-2)', opacity:0.75}));
  });
  svg.appendChild(svgEl('line',{x1:padL,x2:W-padR,y1:H-padB,y2:H-padB,stroke:'var(--border)','stroke-width':1}));
  const lbl = svgEl('text',{x:W-padR,y:H-6,'text-anchor':'end','font-size':8,fill:'var(--bad)','font-family':'IBM Plex Mono, monospace'});
  lbl.textContent = 'ccg score '+unit.acg_violation_ratio.toFixed(3);
  svg.appendChild(lbl);
  container.appendChild(svg);
}
function renderCard(unit){
  const card = document.createElement('div'); card.className='card';
  card.innerHTML = `
    <div class="card-head">
      <div>
        <div class="uid"><span style="color:var(--text-3);font-weight:400;">unit</span> ${unit.unit_id}</div>
        <div class="chips">
          <span class="chip bad">⚠ likely shared artifact</span>
          <span class="chip">KS: ${unit.ks_label} · ContamPct ${unit.ks_contam_pct?.toFixed(1) ?? '—'}%</span>
          <span class="chip">${unit.n_spikes.toLocaleString()} spikes · ${unit.mean_rate_hz.toFixed(1)}Hz</span>
        </div>
      </div>
      <div>
        <div class="score-big">${unit.structural_score.toFixed(2)}</div>
        <div class="score-label">structural score</div>
      </div>
    </div>
    <div class="comp-bars">
      ${Object.entries(unit.components).map(([k,v])=>`
        <div class="comp-row">
          <span class="lbl">${k}</span>
          <span class="track"><span class="fill" style="width:${Math.max(0,Math.min(1,v))*100}%"></span></span>
          <span class="val">${v.toFixed(2)}</span>
        </div>`).join('')}
    </div>
    <div class="chartrow">
      <div class="chartbox"><h4>KS template shape (top 4 chans)</h4><div class="wave"></div></div>
      <div class="chartbox"><h4>Autocorrelogram (±30ms)</h4><div class="acg"></div></div>
    </div>
    <div class="metrics">
      <div class="metric"><div class="k">Best ACG match</div><div class="v bad">unit ${unit.cross_unit_acg_info.best_match_unit} · r=${unit.cross_unit_acg_info.correlation.toFixed(2)}</div></div>
      <div class="metric"><div class="k">Distance to match</div><div class="v bad">${unit.cross_unit_acg_info.distance_um.toFixed(0)}µm</div></div>
      <div class="metric"><div class="k">Footprint concentration</div><div class="v">${unit.footprint_concentration_ratio.toFixed(2)}</div></div>
      <div class="metric"><div class="k">Peak↔trough width</div><div class="v">${unit.ks_peak_trough_width_ms.toFixed(2)}ms</div></div>
      <div class="metric"><div class="k">Peak channel</div><div class="v">${unit.peak_channel}</div></div>
      <div class="metric"><div class="k">Trough-first?</div><div class="v">${unit.ks_trough_before_peak?'yes':'no'}</div></div>
    </div>
  `;
  renderWaveform(card.querySelector('.wave'), unit);
  renderACG(card.querySelector('.acg'), unit);
  return card;
}
const grid = document.getElementById('grid');
UNITS.forEach(u => grid.appendChild(renderCard(u)));
</script>
"""

html = html.replace("__UNITS_JSON__", json.dumps(units))
out_path = r"D:\Gil\spike_sorting_agent\outputs\random_pile_report.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print("wrote", out_path, len(html), "bytes")

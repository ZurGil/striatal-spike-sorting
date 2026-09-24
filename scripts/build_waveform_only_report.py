import json

with open(r"D:\Gil\spike_sorting_agent\outputs\waveform_only_set.json") as f:
    units = json.load(f)

units = sorted(units, key=lambda u: u["ks_peak_trough_width_ms"])

html = r"""<title>Waveform-Only Check</title>
<style>
:root{
  --bg:#f4f6f8; --surface:#ffffff; --surface-2:#eef1f4; --border:#d7dee4;
  --text:#12181f; --text-2:#445261; --text-3:#7c8b9a;
  --accent:#0f8b83; --violet:#7a54a8; --warn:#b5790a; --blue:#3f6fd1;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0a0e13; --surface:#12181f; --surface-2:#1a222b; --border:#232d38;
    --text:#e8edf2; --text-2:#9aacbd; --text-3:#61707e;
    --accent:#4fd1c5; --violet:#a98fd6; --warn:#f2b84b; --blue:#7fa2f0;
  }
}
:root[data-theme="dark"]{
  --bg:#0a0e13; --surface:#12181f; --surface-2:#1a222b; --border:#232d38;
  --text:#e8edf2; --text-2:#9aacbd; --text-3:#61707e;
  --accent:#4fd1c5; --violet:#a98fd6; --warn:#f2b84b; --blue:#7fa2f0;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'Archivo',system-ui,sans-serif;margin:0;padding:0 24px 64px;}
h1{font-size:28px;font-weight:700;letter-spacing:-.01em;text-wrap:balance;}
.wrap{max-width:1300px;margin:0 auto;}
.top{padding:36px 0 16px;border-bottom:1px solid var(--border);}
.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin-bottom:10px;}
.sub{color:var(--text-2);font-size:14px;max-width:72ch;margin-top:8px;line-height:1.55;}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px;margin-top:24px;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px;}
.card .uid{font-family:'IBM Plex Mono',monospace;font-weight:700;font-size:15px;}
.card svg{width:100%;height:auto;display:block;margin-top:8px;}
.card .meta{font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:var(--text-2);margin-top:6px;line-height:1.6;}
.card .meta b{color:var(--text)}
.footer{margin-top:40px;padding-top:16px;border-top:1px solid var(--border);color:var(--text-3);font-size:12px;line-height:1.7;}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">

<div class="wrap">
  <div class="top">
    <div class="eyebrow">Waveform Only &middot; 19 Flagged Units &middot; 20260901_085606</div>
    <h1>Just the shape -- no ACG, no scores</h1>
    <div class="sub">All 19 units flagged by the corrected pipeline (footprint-cliff or width-outlier),
    sorted by peak-trough width, narrowest first. KS's own template on the peak channel, plotted alone so
    it's not primed by anything else on the page.</div>
  </div>
  <div class="grid" id="grid"></div>
  <div class="footer">
    x-axis: time (peak channel, 61-sample window). y-axis: template amplitude (raw units, uncalibrated to µV).
    width = peak&harr;trough duration. amp = peak-to-peak template amplitude. conc. = footprint concentration
    ratio (2nd-strongest channel / peak channel; low = single-channel-isolated). asym. = return-phase length
    &divide; fast-phase length around the dominant deflection (>1 means a slower return than rise, the
    biological pattern bombcell-style checks look for; ~1 or wildly off means symmetric/anomalous).
  </div>
</div>

<script>
const UNITS = __UNITS_JSON__;
function svgEl(tag, attrs){
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for(const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}
function renderWaveform(container, unit){
  const W=250,H=130,padL=6,padR=6,padT=8,padB=8;
  const templ = unit.ks_template;
  const nt = templ.length;
  const svg = svgEl('svg',{viewBox:`0 0 ${W} ${H}`});
  let ymin=Infinity, ymax=-Infinity;
  for(let s=0;s<nt;s++) for(let c=0;c<4;c++){ ymin=Math.min(ymin,templ[s][c]); ymax=Math.max(ymax,templ[s][c]); }
  const x = s => padL + (s/(nt-1))*(W-padL-padR);
  const y = v => padT + (1-(v-ymin)/(ymax-ymin+1e-9))*(H-padT-padB);
  svg.appendChild(svgEl('line',{x1:padL,x2:W-padR,y1:y(0),y2:y(0),stroke:'var(--border)','stroke-width':1}));
  const colors = ['var(--accent)','var(--violet)','var(--warn)','var(--blue)'];
  for(let c=0;c<4;c++){
    let line=[];
    for(let s=0;s<nt;s++) line.push(`${x(s)},${y(templ[s][c])}`);
    svg.appendChild(svgEl('polyline',{points:line.join(' '), fill:'none', stroke:colors[c], 'stroke-width':1.6}));
  }
  container.appendChild(svg);
}
function renderCard(unit){
  const card = document.createElement('div'); card.className='card';
  card.innerHTML = `
    <div class="uid">unit ${unit.unit_id}</div>
    <div class="wave"></div>
    <div class="meta">
      width <b>${unit.ks_peak_trough_width_ms.toFixed(2)}ms</b> &middot;
      amp <b>${unit.ks_template_amplitude.toFixed(1)}</b> &middot;
      conc. <b>${unit.footprint_concentration_ratio.toFixed(2)}</b><br>
      asym. <b>${unit.asymmetry_ratio.toFixed(2)}</b> &middot;
      ${unit.ks_trough_before_peak ? 'trough-first' : 'peak-first'}
    </div>
  `;
  renderWaveform(card.querySelector('.wave'), unit);
  return card;
}
const grid = document.getElementById('grid');
UNITS.forEach(u => grid.appendChild(renderCard(u)));
</script>
"""

html = html.replace("__UNITS_JSON__", json.dumps(units))
out_path = r"D:\Gil\spike_sorting_agent\outputs\waveform_only_report.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print("wrote", out_path, len(html), "bytes")

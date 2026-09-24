import json

with open(r"D:\Gil\spike_sorting_agent\outputs\calibration_set.json") as f:
    units = json.load(f)

BEFORE_FIX = {  # this session's earlier buggy run, for the transparency panel
    187: 21.52, 156: 36.98, 139: 55.09, 151: 30.50, 69: 19.89, 229: 91.65,
}

html = r"""<title>Recovery Calibration Set</title>
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
h1,h2,h3{text-wrap:balance;font-weight:700;margin:0;}
a{color:var(--accent)}

.wrap{max-width:1240px;margin:0 auto;}
.top{padding:40px 0 20px;border-bottom:1px solid var(--border);}
.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin-bottom:10px;}
h1{font-size:32px;letter-spacing:-.01em;}
.sub{color:var(--text-2);font-size:15px;max-width:66ch;margin-top:10px;line-height:1.55;}

.banner{margin:24px 0;padding:20px 22px;border-radius:10px;background:var(--warn-soft);border:1px solid color-mix(in srgb, var(--warn) 40%, transparent);}
.banner h3{font-size:14px;color:var(--warn);text-transform:uppercase;letter-spacing:.05em;font-family:'IBM Plex Mono',monospace;margin-bottom:10px;}
.banner p{margin:0 0 8px;font-size:14px;line-height:1.6;color:var(--text);}
.banner p:last-child{margin-bottom:0}
.banner b{color:var(--text)}

.fixtable{width:100%;border-collapse:collapse;margin-top:14px;font-size:13px;}
.fixtable th{text-align:left;color:var(--text-3);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.04em;padding:6px 10px;border-bottom:1px solid var(--border);}
.fixtable td{padding:6px 10px;border-bottom:1px solid var(--border);}
.fixtable .arrow{color:var(--text-3);padding:0 6px;}
.fixtable .before{color:var(--bad);}
.fixtable .after{color:var(--good);font-weight:600;}

.legend-row{display:flex;gap:18px;flex-wrap:wrap;margin:18px 0 8px;font-size:12px;color:var(--text-2);align-items:center;}
.legend-row .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px;}

.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:20px;margin-top:28px;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px 22px 22px;}
.card-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:4px;}
.uid{font-size:20px;font-weight:700;font-family:'IBM Plex Mono',monospace;}
.uid .hash{color:var(--text-3);font-weight:400;}
.chips{display:flex;gap:6px;margin-top:6px;flex-wrap:wrap;}
.chip{font-size:11px;font-family:'IBM Plex Mono',monospace;padding:2px 8px;border-radius:20px;border:1px solid var(--border);color:var(--text-2);}
.chip.tier{border:none;font-weight:600;}
.chip.tier2{background:var(--warn-soft);color:var(--warn);}
.chip.tier1{background:var(--good-soft);color:var(--good);}
.chip.tier3{background:var(--bad-soft);color:var(--bad);}

.score-big{font-family:'IBM Plex Mono',monospace;font-size:28px;font-weight:700;text-align:right;line-height:1;}
.score-label{font-size:10px;color:var(--text-3);text-transform:uppercase;letter-spacing:.05em;text-align:right;margin-top:4px;}

.chartrow{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:16px;}
.chartbox{background:var(--surface-2);border-radius:8px;padding:10px 12px 8px;}
.chartbox h4{font-size:11px;color:var(--text-3);text-transform:uppercase;letter-spacing:.05em;font-weight:600;margin-bottom:4px;font-family:'IBM Plex Mono',monospace;}
.chartbox svg{width:100%;height:auto;display:block;}
.chart-note{font-size:11px;color:var(--text-3);margin-top:4px;}
.chart-note b{color:var(--text-2)}

.metrics{margin-top:16px;display:grid;grid-template-columns:repeat(4,1fr);gap:10px 16px;font-size:12px;}
.metric .k{color:var(--text-3);font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;}
.metric .v{font-family:'IBM Plex Mono',monospace;font-size:14px;margin-top:2px;color:var(--text);}
.metric .v.warn{color:var(--warn)}
.metric .v.bad{color:var(--bad)}
.metric .v.good{color:var(--good)}

.comp-bars{margin-top:14px;}
.comp-row{display:flex;align-items:center;gap:10px;margin-bottom:5px;font-size:11px;}
.comp-row .lbl{width:88px;color:var(--text-3);font-family:'IBM Plex Mono',monospace;flex-shrink:0;}
.comp-row .track{flex:1;height:6px;background:var(--surface-2);border-radius:3px;overflow:hidden;}
.comp-row .fill{height:100%;background:var(--accent);border-radius:3px;}
.comp-row .val{width:34px;text-align:right;font-family:'IBM Plex Mono',monospace;color:var(--text-2);flex-shrink:0;}

.footer{margin-top:48px;padding-top:20px;border-top:1px solid var(--border);color:var(--text-3);font-size:12px;line-height:1.7;}
</style>

<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">

<div class="wrap">
  <div class="top">
    <div class="eyebrow">Section 7 &middot; Calibration Set &middot; 20260901_085606</div>
    <h1>Ground-truth-by-eye reference set</h1>
    <div class="sub">8 units spanning the full ContamPct/amplitude range this session has evidence about &mdash;
    from clean (0% ContamPct) through the mua/bursty band (27&ndash;30%) and the 30&ndash;45% ACG-filtered band,
    plus one deliberate outlier and one known-severely-contaminated unit as negative controls. Per spec Section 7:
    tune Tier 1 thresholds until the automated calls below match your visual read, before trusting auto-apply at scale.</div>
  </div>

  <div class="banner">
    <h3>&#9888; Bug found and fixed during this calibration run</h3>
    <p>The first calibration pass showed recovery percentages 5&ndash;10&times; higher than the validated Section 1
    gate results for these same units. Root cause: the ACG-inflection stopping rule's fallback defaulted to the
    <b>most permissive</b> tested cutoff when no clear inflection was detected &mdash; and with no floor on candidate
    quality, tens of thousands of near-noise local maxima per unit diluted the violation-ratio metric until it
    stopped climbing at all, so the fallback triggered almost everywhere. Fixed two ways: (1) candidates below 15%
    of the mean accepted-spike score are now dropped before the sweep, (2) the fallback now defaults to the
    <b>strictest</b> tested cutoff, and routes to Tier 2 rather than Tier 1 when no inflection is found.</p>
    <table class="fixtable">
      <tr><th>Unit</th><th>Before fix</th><th></th><th>After fix</th><th>Inflection found?</th></tr>
      __FIX_ROWS__
    </table>
    <p style="margin-top:14px;"><b>Still open:</b> for 6 of these 8 units, no genuine inflection is ever detected
    across the tested percentile range &mdash; the violation ratio just decreases monotonically as more candidates
    are admitted. The conservative fallback keeps results sane, but the "required control" from spec Section 2.7
    isn't actually triggering as designed for anything but the two cleanest units. Visible below as the
    <span class="chip" style="background:var(--warn-soft);color:var(--warn);border:none;">no inflection</span> marker
    on the sweep-curve chart.</p>
  </div>

  <div class="banner" style="background:var(--accent-soft);border-color:color-mix(in srgb, var(--accent) 40%, transparent);">
    <h3 style="color:var(--accent);">&#128269; Waveform shape: a real finding, not a display bug</h3>
    <p>A visual read of unit 187's waveform first looked wrong &mdash; broad and sinusoidal rather than a sharp
    spike. The first hypothesis (raw snippets weren't highpass-filtered before averaging) turned out to be
    <b>incorrect</b>: filtering with far more padding than the original bug (3000 samples vs. 150) barely changed
    the shape. Checked instead against KS's own template for each unit (averaged over far more spikes, spatially
    whitened &mdash; the most reliable shape estimate available) and found a real, consistent pattern:</p>
    <table class="fixtable">
      <tr><th>Unit</th><th>Category</th><th>Peak&ndash;trough width</th><th>Template amplitude</th><th>Trough before peak?</th></tr>
      __WIDTH_ROWS__
    </table>
    <p style="margin-top:14px;">The units the ACG check called "probably clean" (187, 156, 139, 151) all have
    meaningfully broader, ~5&ndash;10&times; weaker waveforms than the two unambiguous clean units &mdash; consistent
    with them being real but sitting right at the detection floor, which also explains their high recovery yields
    (weak signals generate more near-threshold candidate matches). <b>Unit 69 is the interesting reversal:</b>
    despite its elevated ACG ratio, it has the sharpest, most textbook action-potential shape of the whole set, and
    is the only unit where the trough arrives before the peak (the classic fast-sodium/slower-potassium signature).
    That combination looks more like a real, well-defined neuron picking up interference from something nearby
    (a genuine collision) than a poorly-resolved unit. <b>Net effect: the ACG-only read of "these mua units are
    probably clean" needs tempering</b> &mdash; waveform shape is catching something ACG cleanliness alone misses,
    which is exactly why the spec calls for multiple independent structural signals rather than one filter.</p>
  </div>

  <div class="banner" style="background:var(--bad-soft);border-color:color-mix(in srgb, var(--bad) 40%, transparent);">
    <h3 style="color:var(--bad);">&#10060; 187 / 156 / 151 confirmed as likely artifact, not real neurons</h3>
    <p>Two new checks added to the structural score, both quantitatively confirming the visual read:</p>
    <p><b>Cross-unit ACG similarity</b> &mdash; a real neuron's autocorrelogram reflects its own refractory/bursting
    dynamics; two independent real neurons essentially never produce near-identical ACG shapes. Units 187, 156 and
    151 pairwise-correlate at <b>r&nbsp;=&nbsp;0.97&ndash;0.99</b> despite sitting 440&ndash;480&micro;m apart on the
    probe &mdash; every other pair in this batch is r&nbsp;=&nbsp;0.06&ndash;0.74. That combination (near-identical
    timing statistics, physically separate locations) reads as a shared external artifact each channel is picking
    up independently, not three real cells that happen to fire alike.</p>
    <p><b>Footprint concentration</b> &mdash; ratio of the 2nd-strongest channel's amplitude to the peak channel's.
    187 and 156 score 0.07&ndash;0.15 (essentially invisible one channel over) vs. 0.70&ndash;1.00 for every
    genuine-looking unit in the batch, including 151 (1.00) &mdash; a real dipole source doesn't vanish between
    adjacent contacts. 151's footprint looks smooth/real on this axis alone, which is why it wasn't obvious from
    amplitude spread &mdash; but combined with the identical ACG and non-biological waveform shape, it's flagged
    as the same artifact family.</p>
    <p>Both checks are now wired into <code>structural_score.py</code>: a cross-unit ACG correlation &ge; 0.9 to a
    unit farther than the footprint radius is a hard ceiling on the structural score (0.15 max), not just a
    weighted-average penalty &mdash; this pattern means "not a real neuron," not "somewhat lower quality."
    Result: 187, 156 and 151 all now score <b>0.15</b>, sharply separated from every other unit in this set
    (0.56&ndash;0.76).</p>
  </div>

  <div class="legend-row">
    <span><span class="sw" style="background:var(--bad)"></span>refractory zone (&plusmn;1.5ms)</span>
    <span><span class="sw" style="background:var(--shoulder);border:1px solid var(--border)"></span>shoulder zone (5&ndash;25ms, the "expected if flat" reference)</span>
    <span><span class="sw" style="background:var(--good)"></span>inflection found</span>
    <span><span class="sw" style="background:var(--warn)"></span>no inflection &mdash; conservative fallback used</span>
  </div>

  <div class="grid" id="grid"></div>

  <div class="footer">
    Structural score = weighted(footprint 30%, waveform shape 25%, CCG/ACG cleanliness 25%, refractory 20%).
    KS ContamPct shown for reference only per this session's finding that it over-flags clean bursty units by
    assuming a homogeneous spike rate &mdash; never used as a filter here. Waveform panel shows KS's own template
    (not a raw-snippet average, which was tried and found too low-SNR to trust on its own) across the top 4
    footprint channels by peak-to-peak amplitude.
    Generated by scripts/calibration_set.py + scripts/build_calibration_report.py.
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
  // Plots KS's own template (averaged over far more spikes than a quick raw-snippet
  // sample, spatially whitened) -- the most reliable shape estimate available. A raw
  // 200-spike single-channel average was tried first and found too low-SNR to trust on
  // its own (std trace nearly as large as the mean) -- see conversation.
  const W=260,H=150,padL=34,padR=8,padT=8,padB=28;
  const templ = unit.ks_template; // (nt, 4)
  const nt = templ.length, nt0min=20;
  const svg = svgEl('svg',{viewBox:`0 0 ${W} ${H}`});
  let ymin=Infinity, ymax=-Infinity;
  for(let s=0;s<nt;s++) for(let c=0;c<4;c++){
    ymin=Math.min(ymin,templ[s][c]); ymax=Math.max(ymax,templ[s][c]);
  }
  const x = s => padL + (s/(nt-1))*(W-padL-padR);
  const y = v => padT + (1-(v-ymin)/(ymax-ymin))*(H-padT-padB);

  svg.appendChild(svgEl('line',{x1:padL,x2:W-padR,y1:y(0),y2:y(0),stroke:'var(--border)','stroke-width':1}));
  for(let c=0;c<4;c++){
    let line=[];
    for(let s=0;s<nt;s++) line.push(`${x(s)},${y(templ[s][c])}`);
    svg.appendChild(svgEl('polyline',{points:line.join(' '), fill:'none', stroke:CH_COLORS[c], 'stroke-width':1.7}));
  }
  const t0 = svgEl('text',{x:padL,y:H-16,'font-size':8,fill:'var(--text-3)','font-family':'IBM Plex Mono, monospace'}); t0.textContent='-0.67ms';
  const t2 = svgEl('text',{x:W-padR,y:H-16,'text-anchor':'end','font-size':8,fill:'var(--text-3)','font-family':'IBM Plex Mono, monospace'}); t2.textContent='+1.37ms';
  svg.appendChild(t0); svg.appendChild(t2);
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
  container.appendChild(document.createElement('div'));
  svg.appendChild(refr);
  const sh1 = svgEl('rect',{x:x(-25),y:padT,width:x(-5)-x(-25),height:H-padT-padB,fill:'var(--shoulder)'});
  const sh2 = svgEl('rect',{x:x(5),y:padT,width:x(25)-x(5),height:H-padT-padB,fill:'var(--shoulder)'});
  svg.insertBefore(sh2, refr); svg.insertBefore(sh1, refr);

  centers.forEach((c,i)=>{
    const h = (H-padT-padB) - (y(counts[i])-padT);
    svg.appendChild(svgEl('rect',{x:x(c)-barW/2, y:y(counts[i]), width:Math.max(barW-0.4,0.6), height:h, fill:'var(--text-2)', opacity:0.75}));
  });
  svg.appendChild(svgEl('line',{x1:padL,x2:W-padR,y1:H-padB,y2:H-padB,stroke:'var(--border)','stroke-width':1}));
  const lbl = svgEl('text',{x:W-padR,y:H-6,'text-anchor':'end','font-size':8,fill:'var(--bad)','font-family':'IBM Plex Mono, monospace'});
  lbl.textContent = 'ratio '+unit.acg_violation_ratio.toFixed(3);
  svg.appendChild(lbl);
  container.appendChild(svg);
}

function renderSweep(container, unit){
  const W=260,H=150,padL=30,padR=10,padT=10,padB=20;
  const sweep = unit.sweep_curve;
  const svg = svgEl('svg',{viewBox:`0 0 ${W} ${H}`});
  if(!sweep.length){ container.appendChild(svg); return; }
  const ratios = sweep.map(s=>s.acg_violation_ratio);
  const ymin = Math.min(...ratios,0), ymax = Math.max(...ratios)*1.1 || 1;
  const x = i => padL + (i/(sweep.length-1||1))*(W-padL-padR);
  const y = v => padT + (1-(v-ymin)/(ymax-ymin||1))*(H-padT-padB);

  svg.appendChild(svgEl('line',{x1:padL,x2:padL,y1:padT,y2:H-padB,stroke:'var(--border)','stroke-width':1}));
  svg.appendChild(svgEl('line',{x1:padL,x2:W-padR,y1:H-padB,y2:H-padB,stroke:'var(--border)','stroke-width':1}));

  const pts = sweep.map((s,i)=>`${x(i)},${y(s.acg_violation_ratio)}`).join(' ');
  svg.appendChild(svgEl('polyline',{points:pts, fill:'none', stroke:'var(--text-2)', 'stroke-width':1.6}));
  sweep.forEach((s,i)=>{
    svg.appendChild(svgEl('circle',{cx:x(i), cy:y(s.acg_violation_ratio), r:2.2, fill:'var(--text-2)'}));
  });
  // chosen cutoff marker = first point (strictest tested, since sweep is ordered strict->permissive
  // and both real inflections and the fallback land on an entry already in the array)
  let chosenIdx = sweep.findIndex(s=>Math.abs(s.cutoff-unit.chosen_cutoff)<1e-3);
  if(chosenIdx<0) chosenIdx=0;
  const col = unit.inflection_found ? 'var(--good)' : 'var(--warn)';
  svg.appendChild(svgEl('circle',{cx:x(chosenIdx), cy:y(sweep[chosenIdx].acg_violation_ratio), r:4.5, fill:col, stroke:'var(--surface)','stroke-width':1.5}));

  const ylab = svgEl('text',{x:2,y:padT+4,'font-size':7.5,fill:'var(--text-3)','font-family':'IBM Plex Mono, monospace'}); ylab.textContent=ymax.toFixed(2);
  const xlab0 = svgEl('text',{x:padL,y:H-6,'font-size':7.5,fill:'var(--text-3)','font-family':'IBM Plex Mono, monospace'}); xlab0.textContent='p50';
  const xlab1 = svgEl('text',{x:W-padR,y:H-6,'text-anchor':'end','font-size':7.5,fill:'var(--text-3)','font-family':'IBM Plex Mono, monospace'}); xlab1.textContent='p1';
  svg.appendChild(ylab); svg.appendChild(xlab0); svg.appendChild(xlab1);
  container.appendChild(svg);
}

function fmtPct(v){ return v.toFixed(2)+'%'; }

function renderCard(unit){
  const card = document.createElement('div'); card.className='card';
  const tierClass = unit.tier.includes('tier1') ? 'tier1' : unit.tier.includes('tier3') ? 'tier3' : 'tier2';
  const tierLabel = unit.tier.replace('tier1_auto_applied','Tier 1 · auto-applied')
                              .replace('tier2_scored_logged','Tier 2 · scored & logged')
                              .replace('tier3_insufficient_evidence','Tier 3 · insufficient evidence');

  card.innerHTML = `
    <div class="card-head">
      <div>
        <div class="uid"><span class="hash">unit</span> ${unit.unit_id}</div>
        <div class="chips">
          <span class="chip tier ${tierClass}">${tierLabel}</span>
          ${unit.likely_shared_artifact ? '<span class="chip tier tier3">⚠ likely shared artifact</span>' : ''}
          <span class="chip">KS: ${unit.ks_label} · ContamPct ${unit.ks_contam_pct?.toFixed(1) ?? '—'}%</span>
          <span class="chip">${unit.n_spikes.toLocaleString()} spikes · ${unit.mean_rate_hz.toFixed(1)}Hz</span>
        </div>
      </div>
      <div>
        <div class="score-big" style="${unit.likely_shared_artifact ? 'color:var(--bad)' : ''}">${unit.structural_score.toFixed(2)}</div>
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
    <div class="chartrow" style="grid-template-columns:1fr;">
      <div class="chartbox"><h4>Recovery sweep: cutoff percentile vs. ACG violation ratio</h4><div class="sweep"></div>
        <div class="chart-note">chosen cutoff: <b>p${['50','40','30','20','10','5','2','1'][unit.sweep_curve.findIndex(s=>Math.abs(s.cutoff-unit.chosen_cutoff)<1e-3)] ?? '50'}</b>
        &nbsp;&middot;&nbsp; ${unit.inflection_found ? '<b style="color:var(--good)">inflection found</b>' : '<b style="color:var(--warn)">no inflection — conservative fallback</b>'}</div>
      </div>
    </div>

    <div class="metrics">
      <div class="metric"><div class="k">Spikes added</div><div class="v ${unit.pct_spikes_added>10?'bad':unit.pct_spikes_added>3?'warn':''}">${fmtPct(unit.pct_spikes_added)} (${unit.n_recovered.toLocaleString()})</div></div>
      <div class="metric"><div class="k">ACG viol. before → after</div><div class="v">${unit.acg_violation_before.toFixed(3)} → ${unit.acg_violation_after.toFixed(3)}</div></div>
      <div class="metric"><div class="k">CV / LV</div><div class="v">${unit.cv.toFixed(2)} / ${unit.lv.toFixed(2)}</div></div>
      <div class="metric"><div class="k">Burst / pause frac.</div><div class="v">${(unit.burst_fraction*100).toFixed(0)}% / ${(unit.pause_time_fraction*100).toFixed(0)}%</div></div>
      <div class="metric"><div class="k">Peak channel</div><div class="v">${unit.peak_channel}</div></div>
      <div class="metric"><div class="k">Peak↔trough width</div><div class="v">${unit.ks_peak_trough_width_ms.toFixed(2)}ms</div></div>
      <div class="metric"><div class="k">Trough-first (canonical)?</div><div class="v ${unit.ks_trough_before_peak?'good':''}">${unit.ks_trough_before_peak?'yes':'no'}</div></div>
      <div class="metric"><div class="k">Footprint concentration</div><div class="v ${unit.footprint_concentration_ratio<0.3?'bad':''}">${unit.footprint_concentration_ratio.toFixed(2)}</div></div>
      <div class="metric"><div class="k">Best ACG match (other unit)</div><div class="v ${unit.cross_unit_acg_info.correlation>=0.9?'bad':''}">unit ${unit.cross_unit_acg_info.best_match_unit} · r=${unit.cross_unit_acg_info.correlation.toFixed(2)} · ${unit.cross_unit_acg_info.distance_um.toFixed(0)}µm</div></div>
      <div class="metric"><div class="k">n bursts</div><div class="v">${unit.n_bursts.toLocaleString()}</div></div>
      <div class="metric"><div class="k">Candidates (raw → after floor)</div><div class="v">${unit.n_candidates_before_floor.toLocaleString()} → ${unit.n_candidates_after_floor.toLocaleString()}</div></div>
      <div class="metric"><div class="k">Collision tags (n=${unit.collision_n_sampled})</div><div class="v" style="font-size:11px;">${Object.entries(unit.collision_counts).map(([k,v])=>`${k}:${v}`).join(', ')}</div></div>
    </div>
  `;
  renderWaveform(card.querySelector('.wave'), unit);
  renderACG(card.querySelector('.acg'), unit);
  renderSweep(card.querySelector('.sweep'), unit);
  return card;
}

const grid = document.getElementById('grid');
UNITS.forEach(u => grid.appendChild(renderCard(u)));
</script>
"""

fix_rows = ""
for u in units:
    uid = u["unit_id"]
    after = u["pct_spikes_added"]
    if uid in BEFORE_FIX:
        before = BEFORE_FIX[uid]
        found = "yes" if u["inflection_found"] else "no (fallback)"
        fix_rows += (f'<tr><td class="mono">{uid}</td><td class="mono before">{before:.2f}%</td>'
                     f'<td class="arrow">&rarr;</td><td class="mono after">{after:.2f}%</td>'
                     f'<td class="mono">{found}</td></tr>\n')
    else:
        found = "yes" if u["inflection_found"] else "no (fallback)"
        fix_rows += (f'<tr><td class="mono">{uid}</td><td class="mono" style="color:var(--text-3)">'
                     f'unaffected (few candidates)</td><td class="arrow"></td>'
                     f'<td class="mono after">{after:.2f}%</td><td class="mono">{found}</td></tr>\n')

CATEGORY = {285:"clean reference",295:"clean reference",187:"ACG-clean mua",156:"ACG-clean mua",
            139:"ACG-clean 30-45%",151:"ACG-clean 30-45%",69:"ACG-flagged outlier",229:"known-bad"}
width_rows = ""
for u in sorted(units, key=lambda u: u["ks_peak_trough_width_ms"]):
    uid = u["unit_id"]
    tf = "yes" if u["ks_trough_before_peak"] else "no"
    tf_style = ' style="color:var(--accent);font-weight:600;"' if u["ks_trough_before_peak"] else ""
    width_rows += (f'<tr><td class="mono">{uid}</td><td>{CATEGORY.get(uid,"")}</td>'
                    f'<td class="mono">{u["ks_peak_trough_width_ms"]:.2f}ms</td>'
                    f'<td class="mono">{u["ks_template_amplitude"]:.1f}</td>'
                    f'<td class="mono"{tf_style}>{tf}</td></tr>\n')

html = html.replace("__FIX_ROWS__", fix_rows)
html = html.replace("__WIDTH_ROWS__", width_rows)
html = html.replace("__UNITS_JSON__", json.dumps(units))

out_path = r"D:\Gil\spike_sorting_agent\outputs\calibration_report.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print("wrote", out_path, len(html), "bytes")

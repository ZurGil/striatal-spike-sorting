# -*- coding: utf-8 -*-
"""Third pass over the merge panel. Applied after v1 (the panel) and v2 (the metric).

1. THE VERDICT BUTTONS NEVER WORKED. `data-mv=' + JSON.stringify(JSON.stringify(patch))`
   writes backslash-escaped quotes into an HTML attribute, which HTML does not
   understand: the attribute terminates at the first inner quote, `dataset.mv` comes
   back as the two characters `{\`, and JSON.parse throws on every click. Silent --
   the button highlights nothing and no row is written. Present since the panel was
   first spliced in, which is why `merge_verdicts` is empty. Replaced with two plain
   attributes that need no quoting at all.

2. The panel moves to the BOTTOM of the unit view: examine the unit, then decide
   whether any of it should be merged.

3. Jump to the other unit and back. Each candidate row carries the partner's id as a
   button; following it pushes the unit you came from onto a stack, and a back chip
   at the top of the detail walks it. Going 42 -> 50 -> 61 and back three times
   returns the way you came.

4. KILOSORT SIMILARITY IS PROMOTED, because measuring it showed Gil's recollection
   was right and my blanket dismissal was too strong. Of 2554 pairs with a readable
   CCG, 0.51% show a one-neuron dip. By similarity band:

       0.00-0.50   2051 pairs   0.2%    0.5x chance
       0.50-0.80    418 pairs   1.4%    2.8x
       0.80-0.90     57 pairs   1.8%    3.4x
       0.90-0.95     15 pairs   6.7%   13.1x
       >=0.95        17 pairs   0.0%     --

   So it carries real information and is not noise. What the top band shows is that
   similarity alone cannot close a case: every pair at >=0.95 has waveform r between
   0.88 and 0.97 and a partial dip (0.25-0.80), and not one of them has enough
   coincident spikes to prove anything -- exactly where the footprint and the
   timelines have to decide instead. Those pairs were being hidden behind "show the
   other N neighbours", which is wrong. They now surface as `twin templates`.
"""
import io
import os
import sys

SRC = (sys.argv[1] if len(sys.argv) > 1 else
       r"C:\Users\Adam\AppData\Local\Temp\claude\D--Gil-spike-sorting-agent"
       r"\8186c809-76fa-41ff-bbca-59d44125e5fc\scratchpad\review_tool_v2.html")
DST = (sys.argv[2] if len(sys.argv) > 2 else
       r"C:\Users\Adam\AppData\Local\Temp\claude\D--Gil-spike-sorting-agent"
       r"\8186c809-76fa-41ff-bbca-59d44125e5fc\scratchpad\review_tool_v3.html")

# ---------------------------------------------------------------- 1. the verdict bug
OLD_PICK = u"""  const pick = (field, val, cls, text) => {
    const on = (field === 'verdict' ? (v.verdict === val) : ((v.quality_label || '') === val));
    const patch = {}; patch[field] = val;
    return '<button class="mbtn' + (on ? ' ' + cls : '') + '" data-pair="' + p.pair_id +
      '" data-mv=' + JSON.stringify(JSON.stringify(patch)) + '>' + text + '</button>';
  };"""

NEW_PICK = u"""  /* Two plain attributes, not JSON. JSON.stringify escapes quotes with backslashes,
     which HTML attributes do not understand -- the value terminated at the first
     inner quote and every click threw SyntaxError before reaching the database.
     field is 'verdict' or 'quality_label' and val is a bare word, so neither needs
     quoting. */
  const pick = (field, val, cls, text) => {
    const on = (field === 'verdict' ? (v.verdict === val) : ((v.quality_label || '') === val));
    return '<button class="mbtn' + (on ? ' ' + cls : '') + '" data-pair="' + p.pair_id +
      '" data-mvf="' + field + '" data-mvv="' + val + '">' + text + '</button>';
  };"""

OLD_MV_HANDLER = u"""  container.querySelectorAll('[data-mv]').forEach(b => {
    b.addEventListener('click', () => setMergeVerdict(b.dataset.pair, JSON.parse(b.dataset.mv)));
  });"""

NEW_MV_HANDLER = u"""  container.querySelectorAll('[data-mvf]').forEach(b => {
    b.addEventListener('click', () => {
      const patch = {};
      patch[b.dataset.mvf] = b.dataset.mvv;
      setMergeVerdict(b.dataset.pair, patch);
    });
  });
  container.querySelectorAll('[data-goto]').forEach(b => {
    b.addEventListener('click', e => { e.stopPropagation(); mergeGoto(Number(b.dataset.goto)); });
  });"""

# ------------------------------------------------- 2. the panel moves to the bottom
OLD_SLOT = u"""    <div class="timeline-panel"></div>
    <div class="merge-panel"></div>"""
NEW_SLOT = u"""    <div class="timeline-panel"></div>"""

OLD_TAIL = u"""      <div class="db-status" id="dbStatusText">syncing...</div>
    </div>
  `;"""
NEW_TAIL = u"""      <div class="db-status" id="dbStatusText">syncing...</div>
    </div>
    <div class="merge-panel" style="margin-top:16px;"></div>
  `;"""

OLD_PANELS = u"""    <div class="panels">
      <div class="panel"><div class="panel-label">KS waveform (top 4 channels)</div><div class="wave"></div></div>"""
NEW_PANELS = u"""    <div class="merge-nav"></div>
    <div class="panels">
      <div class="panel"><div class="panel-label">KS waveform (top 4 channels)</div><div class="wave"></div></div>"""

OLD_RENDER = u"""  renderMergePanel(el.querySelector('.merge-panel'), uid);"""
NEW_RENDER = u"""  renderMergePanel(el.querySelector('.merge-panel'), uid);
  renderMergeNav(el.querySelector('.merge-nav'));"""

# ------------------------------------------------------------- 3 + 4. nav and twins
OLD_DRIFT = u"""function isDrift(p){
  return p.rate_corr !== null && p.rate_corr <= -0.1 &&
         p.active_overlap !== null && p.active_overlap <= 0.35;
}"""

NEW_DRIFT = u"""function isDrift(p){
  return p.rate_corr !== null && p.rate_corr <= -0.1 &&
         p.active_overlap !== null && p.active_overlap <= 0.35;
}

/* Kilosort's template similarity cannot DECIDE a merge -- it compares templates, so
   two different neurons with similar shapes score high. But measured on this session
   it is far from uninformative: of the 2554 pairs with a readable correlogram 0.51%
   show a one-neuron dip, and that rises to 1.4% at similarity 0.5-0.8 (2.8x chance),
   1.8% at 0.8-0.9 (3.4x) and 6.7% at 0.90-0.95 (13x). Above 0.95 something different
   happens: all 17 such pairs have waveform r of 0.88-0.97 and a partial dip, and not
   one has enough coincident spikes to settle it. Those are precisely the pairs where
   the footprint and the session timelines have to decide, so they belong in front of
   you rather than behind "show the other neighbours". */
const SIM_TWIN = 0.95, WFR_TWIN = 0.85;
function isTwin(p){
  const c = dipClass(p)[0];
  return (p.template_similarity || 0) >= SIM_TWIN &&
         (p.waveform_corr === null ? false : p.waveform_corr >= WFR_TWIN) &&
         (c === 'thin' || c === 'partial');
}

/* --- jumping to the other unit, and back ----------------------------------- */
let mergeNav = [];                     /* units you came from, most recent last */
function mergeGoto(to){
  if(selected != null && selected !== to) mergeNav.push(selected);
  selected = to;
  renderList();
  renderDetail(to);
  const r = document.querySelector('.right');
  if(r) r.scrollTop = 0;
}
function mergeBack(){
  if(!mergeNav.length) return;
  selected = mergeNav.pop();
  renderList();
  renderDetail(selected);
  const r = document.querySelector('.right');
  if(r) r.scrollTop = 0;
}
function renderMergeNav(container){
  if(!container) return;
  if(!mergeNav.length){ container.innerHTML = ''; return; }
  const trail = mergeNav.slice(-5).map(u => '#' + u).join(' &rarr; ');
  container.innerHTML =
    '<button class="merge-back-btn" id="mergeBackBtn">&larr; back to #' +
    mergeNav[mergeNav.length - 1] + '</button>' +
    '<span class="merge-trail">came from ' + trail + '</span>';
  const b = container.querySelector('#mergeBackBtn');
  if(b) b.addEventListener('click', mergeBack);
}"""

OLD_WORTH = u"function isWorthLook(p){ return dipClass(p)[0] === 'one' || isDrift(p); }"
NEW_WORTH = u"function isWorthLook(p){ return dipClass(p)[0] === 'one' || isDrift(p) || isTwin(p); }"

OLD_PID = u"""      '<span class="pid">#' + other + '</span>' +"""
NEW_PID = u"""      '<span class="pid">#' + other + '</span>' +
      '<button class="mc-goto" data-goto="' + other + '" title="open unit #' + other +
      ' (a back button appears at the top)">open &#8599;</button>' +"""

OLD_DRIFT_TAG = u"""      (isDrift(p) ? '<span class="tag drift">drift pair?</span>' : '') +"""
NEW_DRIFT_TAG = u"""      (isDrift(p) ? '<span class="tag drift">drift pair?</span>' : '') +
      (isTwin(p) ? '<span class="tag twin">twin templates</span>' : '') +"""

OLD_NUMS = u"""      '<span>wf r <b>' + mFmt(p.waveform_corr) + '</b></span>' +"""
NEW_NUMS = u"""      '<span>wf r <b>' + mFmt(p.waveform_corr) + '</b></span>' +
      '<span>KS sim <b>' + mFmt(p.template_similarity, 3) + '</b></span>' +"""

OLD_CSS = u".mc-a{stroke:var(--accent);fill:none;stroke-width:2;}"
NEW_CSS = u""".mc-a{stroke:var(--accent);fill:none;stroke-width:2;}
.tag.twin{background:var(--warn-soft);color:var(--warn);}
.mc-goto{font-size:9px;font-family:'IBM Plex Mono',monospace;color:var(--accent);
  background:transparent;border:1px solid var(--accent);border-radius:8px;
  padding:0 6px;cursor:pointer;flex-shrink:0;line-height:1.5;}
.mc-goto:hover{background:var(--accent-soft);}
.merge-nav{display:flex;align-items:center;gap:10px;margin-bottom:12px;}
.merge-nav:empty{display:none;}
.merge-back-btn{font-size:11px;font-family:'IBM Plex Mono',monospace;color:var(--accent);
  background:var(--accent-soft);border:1px solid var(--accent);border-radius:14px;
  padding:3px 12px;cursor:pointer;font-weight:600;}
.merge-trail{font-size:10px;color:var(--text-3);font-family:'IBM Plex Mono',monospace;}"""

OLD_BLURB_TAIL = u"""'&le;&minus;0.1 with simultaneous-activity overlap &le;0.35), which is a drift pair the ' +
    'correlogram cannot see. Everything else stays one click away rather than crowding those out. ' +"""

NEW_BLURB_TAIL = u"""'&le;&minus;0.1 with simultaneous-activity overlap &le;0.35), which is a drift pair the ' +
    'correlogram cannot see, or the pair is a <b>twin</b>: KS template similarity &ge;' + SIM_TWIN +
    ' with waveform r &ge;' + WFR_TWIN + ' but a correlogram too thin to settle it. Similarity never ' +
    'decides &mdash; it compares templates, so two neurons of similar shape score high &mdash; but on ' +
    'this session it is informative: against a measured base rate of <b>0.51%</b> of readable pairs ' +
    'showing a one-neuron dip, the 0.90&ndash;0.95 band runs at 6.7%, which is <b>13&times; chance</b>. ' +
    'Everything else stays one click away rather than crowding those out. ' +"""

SUBS = [
    (OLD_PICK, NEW_PICK, "verdict buttons: plain attributes instead of embedded JSON"),
    (OLD_MV_HANDLER, NEW_MV_HANDLER, "verdict + goto click handlers"),
    (OLD_SLOT, NEW_SLOT, "remove the merge panel from under the timeline"),
    (OLD_TAIL, NEW_TAIL, "merge panel moved to the bottom of the unit view"),
    (OLD_PANELS, NEW_PANELS, "back-navigation slot at the top of the detail"),
    (OLD_RENDER, NEW_RENDER, "render the back chip"),
    (OLD_DRIFT, NEW_DRIFT, "isTwin + unit navigation"),
    (OLD_WORTH, NEW_WORTH, "twins count as worth a look"),
    (OLD_PID, NEW_PID, "open-the-other-unit button on each row"),
    (OLD_DRIFT_TAG, NEW_DRIFT_TAG, "twin tag"),
    (OLD_NUMS, NEW_NUMS, "KS similarity shown on the collapsed row"),
    (OLD_CSS, NEW_CSS, "styles for twin tag, goto button, back chip"),
    (OLD_BLURB_TAIL, NEW_BLURB_TAIL, "blurb explains twins with the measured chance rate"),
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
    if "data-mv=" in s:
        raise SystemExit("the broken data-mv attribute is still present")
    d = os.path.dirname(DST)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    io.open(DST, "w", encoding="utf-8").write(s)
    print("wrote %s (%d bytes)" % (DST, os.path.getsize(DST)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""Add the "export all for sync" button to the review tool.

    python scripts/patch_review_tool_export_all.py <src.html> <dst.html>

WHY THIS BUTTON EXISTS
----------------------
Gil's decisions live in the artifact's shared database on claude.ai. The pipeline reads
three plain files in the Kilosort folder. Something has to carry the first to the
second, and a web page cannot: the browser sandbox has no filesystem and cannot start a
process, so a button can never run `sync_session.py` itself. That is a hard limit of
where the page runs, not a missing feature.

What a page CAN do is hand the viewer files, through the `downloads` capability the
tool already uses for its merge export. So this button writes all three collections to
disk in one click, named exactly as `collect_verdict_dumps.py` expects:

    verdicts_<session>.json          every good / mua / noise verdict
    trial_selections_<session>.json  every stability window, full documents
    merge_verdicts_<session>.json    every merge / separate decision

After which Gil runs one command and never needs Claude in the loop:

    python scripts/sync_session.py <sid> --rec <rec> --rat_root <root> --collect_from_downloads

The merge panel's own "export N merges" button is left alone: it writes the
`write_unit_merges.py` *input* shape (already grouped), which is a different thing from
a raw dump of the collection, and both are useful.

ONE DETAIL THAT MATTERS
-----------------------
The page's `timeSel` cache keeps only each unit's `ranges`, dropping `ephys_t0_s` /
`ephys_t1_s` / `n_markers_inside`. The session ephys window is NOT optional downstream:
`write_unit_merges.py` needs it to express "this merged unit keeps every trial" when a
merge mixes a restricted parent with an unrestricted one, and refuses rather than
silently narrowing if it is missing. So this patch adds a full-document cache and
exports that.
"""
import io
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else None
DST = sys.argv[2] if len(sys.argv) > 2 else None
if not SRC or not DST:
    raise SystemExit(__doc__)

# --- 1. keep the FULL trial_selections documents, not just their ranges -------------
A1_OLD = u"""let timeSel = {};    // unit_id -> [{t0,t1}, ...]  -- empty/absent means KEEP EVERY TRIAL"""
A1_NEW = u"""let timeSel = {};    // unit_id -> [{t0,t1}, ...]  -- empty/absent means KEEP EVERY TRIAL
/* The whole trial_selections document per unit. timeSel keeps only `ranges`, but the
   export has to carry ephys_t0_s/ephys_t1_s too: write_unit_merges.py needs the session
   window to say "keeps every trial" for a merge of a restricted and an unrestricted
   parent, and refuses rather than guess if it is absent. */
let timeSelDocs = {};"""

A2_OLD = u"""      if(d && Array.isArray(d.ranges)) timeSel[doc.id] = d.ranges;"""
A2_NEW = u"""      if(d && Array.isArray(d.ranges)) { timeSel[doc.id] = d.ranges; timeSelDocs[doc.id] = d; }"""

# --- 2. the exporter ----------------------------------------------------------------
A3_OLD = u"""async function exportMerges(){"""
A3_NEW = u"""/* Write all three decision collections to disk, named as collect_verdict_dumps.py
   expects. This is the browser half of the hop to the dataset; the other half is one
   local command. A button cannot run that command -- no filesystem, no processes in a
   page -- so this hands over the files instead. */
async function exportAllForSync(){
  const files = [
    ['verdicts_' + SESSION_ID + '.json', verdicts],
    ['trial_selections_' + SESSION_ID + '.json', timeSelDocs],
    ['merge_verdicts_' + SESSION_ID + '.json', mergeVerdicts],
  ];
  const counts = files.map(f => Object.keys(f[1] || {}).length);
  if(counts.every(n => n === 0)){
    alert('No decisions recorded yet -- nothing to export.');
    return;
  }
  let dl = null;
  try { dl = await claude.use('downloads'); } catch(e){ dl = null; }
  if(!dl){
    alert('Downloads are not available in this view, so the files cannot be saved.\\n\\n' +
          'Open the artifact on claude.ai (not in a preview) and try again, or ask ' +
          'Claude to dump the verdicts instead.');
    return;
  }
  const done = [], failed = [];
  for(let i = 0; i < files.length; i++){
    const name = files[i][0], obj = files[i][1] || {};
    try {
      await dl.save({ filename: name, data: JSON.stringify(obj, null, 2) });
      done.push(name + '  (' + counts[i] + ')');
    } catch(e){
      /* the viewer can decline each save individually */
      failed.push(name);
    }
  }
  let msg = '';
  if(done.length) msg += 'Saved:\\n  ' + done.join('\\n  ') + '\\n\\n';
  if(failed.length) msg += 'Not saved (declined or failed):\\n  ' + failed.join('\\n  ') +
                           '\\n\\nAll three are needed -- the sync refuses to run ' +
                           'without them.\\n\\n';
  if(!failed.length){
    msg += 'Now run, on the recording PC:\\n\\n' +
           '  python scripts/sync_session.py ' + SESSION_ID + ' ^\\n' +
           '      --rec F:\\\\Gil\\\\Shamir\\\\' + SESSION_ID + '.rec ^\\n' +
           '      --rat_root Z:\\\\Gil\\\\Shamir_1 ^\\n' +
           '      --collect_from_downloads\\n\\n' +
           'That applies every decision, re-runs the pipeline, and verifies that each ' +
           'one reached synced/.';
  }
  alert(msg);
}

async function exportMerges(){"""

# --- 3. the button ------------------------------------------------------------------
A4_OLD = u"""    <div class="controls" id="activeFilters" style="min-height:28px;"></div>
    <div class="list" id="list"></div>"""
A4_NEW = u"""    <div class="controls" id="activeFilters" style="min-height:28px;"></div>
    <div class="controls" style="border-top:1px solid var(--border);padding-top:7px;">
      <button id="exportAllBtn" title="Save all three decision files, then run sync_session.py on the recording PC"
              style="font-size:10px;padding:4px 10px;border-radius:6px;border:1px solid var(--accent);
                     background:var(--accent-soft);color:var(--accent);cursor:pointer;font-weight:600;">
        \u2193 export all for sync</button>
      <span id="exportAllCount" style="font-size:10px;color:var(--text-3);"></span>
    </div>
    <div class="list" id="list"></div>"""

# --- 4. wire it up, and keep the count honest --------------------------------------
A5_OLD = u"""function renderList(){"""
A5_NEW = u"""/* The button's subtitle: what WOULD be exported right now. Reading it off the same
   caches the export uses means it cannot claim decisions that would not be written.

   The try/catch is load-bearing, not decoration. This runs from the top of
   renderList(), and `mergeVerdicts` is a `let` declared further down the file: on any
   call before that line is reached it sits in its temporal dead zone, where even
   `typeof` throws ReferenceError (typeof is only safe for genuinely undeclared names).
   Unguarded, that would kill renderList() and blank the unit list -- the exact failure
   mode an undeclared constant in the merge panel caused here once already. So the
   catch is what makes this safe; the typeof checks only cover the simpler case of the
   cache not existing at all. */
function refreshExportAllCount(){
  try {
    const el = document.getElementById('exportAllCount');
    if(!el) return;
    const size = o => (o && typeof o === 'object') ? Object.keys(o).length : 0;
    const nv = (typeof verdicts     !== 'undefined') ? size(verdicts)     : 0;
    const nt = (typeof timeSelDocs  !== 'undefined') ? size(timeSelDocs)  : 0;
    const nm = (typeof mergeVerdicts !== 'undefined') ? size(mergeVerdicts) : 0;
    el.textContent = nv + ' verdict' + (nv === 1 ? '' : 's') + ' \\u00b7 ' +
                     nt + ' window' + (nt === 1 ? '' : 's') + ' \\u00b7 ' +
                     nm + ' merge call' + (nm === 1 ? '' : 's');
  } catch(e){ /* never let the counter break the list */ }
}

function renderList(){
  refreshExportAllCount();"""

# The page has no DOMContentLoaded hook -- its script sits at the end of the body and
# bootstraps inline, so this attaches on the same tail.
A6_OLD = u"""updateProgress();
renderList();
renderDetail(selected);
initDb();"""
A6_NEW = u"""updateProgress();
renderList();
renderDetail(selected);
(function(){
  const eab = document.getElementById('exportAllBtn');
  if(eab) eab.addEventListener('click', exportAllForSync);
  refreshExportAllCount();
})();
initDb();"""

PATCHES = [
    (A1_OLD, A1_NEW, "full trial_selections cache (declaration)"),
    (A2_OLD, A2_NEW, "full trial_selections cache (snapshot)"),
    (A3_OLD, A3_NEW, "exportAllForSync()"),
    (A4_OLD, A4_NEW, "export-all button"),
    (A5_OLD, A5_NEW, "refreshExportAllCount()"),
    (A6_OLD, A6_NEW, "button wiring"),
]


def main():
    s = io.open(SRC, encoding="utf-8").read()
    for old, new, label in PATCHES:
        n = s.count(old)
        if n != 1:
            raise SystemExit("anchor %r matched %d times (expected 1) in %s -- the "
                             "generator has drifted; fix the anchor rather than "
                             "publishing a half-patched page." % (label, n, SRC))
        s = s.replace(old, new)
        print("  ok: %s" % label)
    io.open(DST, "w", encoding="utf-8").write(s)
    print("wrote %s" % DST)


if __name__ == "__main__":
    main()

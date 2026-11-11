# -*- coding: utf-8 -*-
"""Make an archived review tool show its verdicts without the online database.

    python scripts/patch_review_tool_offline_verdicts.py <bundle_dir>

WHY
---
The review tool keeps verdicts in the published page's own database on claude.ai.
A copy of the page on the archive has no database, so `claude.use('db')` returns
null, `initDb` returns early, and every unit renders as un-reviewed. The decisions
are not lost -- they are exported beside the page under `verdicts/` -- but a reader
opening the archived page sees a blank slate and reasonably concludes the curation
was never saved. An archive that cannot display what it archives is not much of an
archive.

So: when the database is unavailable, load the same three collections from the
sibling JSON files and populate exactly the globals the snapshot handlers populate.
Nothing else about the page changes, and the online page is untouched -- this patch
is applied only to the copy in the bundle.

The page is left READ-ONLY on purpose. `setVerdict` and `setMergeVerdict` already
guard their writes with `if(db)`, so clicking a verdict here changes the screen and
nothing else. That silent no-op is its own trap, so the banner says plainly that
this is the archived copy and that changes will not be kept.
"""
import io
import json
import os
import sys

ANCHOR = (
    "  if(!db){ document.getElementById('dbStatusText') && "
    "(document.getElementById('dbStatusText').textContent='local only "
    "(db unavailable)'); return; }"
)

REPLACEMENT = """  if(!db){ await loadArchivedVerdicts(); return; }"""

LOADER = """
/* ---- archived copy: verdicts come from files, not from the database ----
   Added by scripts/patch_review_tool_offline_verdicts.py. The published page never
   takes this path, because there `claude.use('db')` succeeds. */
async function loadArchivedVerdicts(){
  const el = document.getElementById('dbStatusText');
  let loaded = {verdicts: 0, trial_selections: 0, merge_verdicts: 0};
  async function grab(name){
    const r = await fetch('verdicts/' + name + '.json', {cache: 'no-store'});
    if(!r.ok) throw new Error(name + ': HTTP ' + r.status);
    return await r.json();
  }
  try{
    const v = await grab('verdicts');
    for(const uid in v){
      const d = v[uid];
      verdicts[uid] = (d && typeof d === 'object') ? d : {verdict: d};
      loaded.verdicts++;
    }
  }catch(e){ console.error('archived verdicts', e); }
  try{
    const t = await grab('trial_selections');
    for(const uid in t){
      const d = t[uid];
      if(d && Array.isArray(d.ranges)){ timeSel[uid] = d.ranges; timeSelDocs[uid] = d; loaded.trial_selections++; }
    }
  }catch(e){ console.error('archived trial_selections', e); }
  try{
    const m = await grab('merge_verdicts');
    for(const pid in m){ mergeVerdicts[pid] = m[pid]; loaded.merge_verdicts++; }
  }catch(e){ console.error('archived merge_verdicts', e); }

  if(el){
    el.textContent = 'ARCHIVED COPY \\u2014 ' + loaded.verdicts + ' verdicts, '
      + loaded.merge_verdicts + ' merge decisions and ' + loaded.trial_selections
      + ' stability selections loaded from files. Read-only: changes are NOT saved.';
    el.style.color = '#b45309';
    el.style.fontWeight = '600';
  }
  updateProgress();
  renderList();
  if(selected != null) renderDetail(selected);
}
"""


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: %s <bundle_dir>" % os.path.basename(sys.argv[0]))
    bundle = sys.argv[1]

    pages = [f for f in os.listdir(bundle)
             if f.startswith("unit_review_tool_") and f.endswith(".html")]
    if len(pages) != 1:
        raise SystemExit("expected exactly one unit_review_tool_*.html in %s, found %d"
                         % (bundle, len(pages)))
    page = os.path.join(bundle, pages[0])

    # The three exports must be present and readable before the page is told to
    # depend on them -- a page that promises verdicts and then 404s is worse than
    # one that admits it has none.
    counts = {}
    for name in ("verdicts", "trial_selections", "merge_verdicts"):
        p = os.path.join(bundle, "verdicts", name + ".json")
        if not os.path.isfile(p):
            raise SystemExit("missing export: %s" % p)
        with io.open(p, encoding="utf-8") as f:
            counts[name] = len(json.load(f))

    with io.open(page, encoding="utf-8") as f:
        html = f.read()

    if "loadArchivedVerdicts" in html:
        raise SystemExit("already patched: %s" % page)

    n = html.count(ANCHOR)
    if n != 1:
        raise SystemExit("anchor found %d times, expected exactly 1 -- the page has "
                         "changed and this patch must be re-checked before use" % n)

    html = html.replace(ANCHOR, REPLACEMENT)

    # Put the loader immediately before initDb so it is defined in the same scope.
    marker = "async function initDb(){"
    if html.count(marker) != 1:
        raise SystemExit("initDb marker not unique")
    html = html.replace(marker, LOADER.strip() + "\n\n" + marker)

    with io.open(page, "w", encoding="utf-8") as f:
        f.write(html)

    print("patched %s" % page)
    for k, v in sorted(counts.items()):
        print("   will load %-18s %d documents" % (k, v))


if __name__ == "__main__":
    main()

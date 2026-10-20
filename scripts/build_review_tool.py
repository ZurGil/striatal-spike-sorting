# -*- coding: utf-8 -*-
"""Build the COMPLETE review tool for a session, merge panel and all. One command.

    python scripts/build_review_tool.py <session_id>

Why this exists: `build_review_html.py` produces the unit review page (waveforms,
scatter, ACG, burst curve, session timeline, verdicts) but knows nothing about
merges. The merge panel was developed as three successive patches spliced onto the
page, which is reproducible only if someone remembers the order. This runs the whole
chain, in order, failing loudly if any anchor has drifted:

    build_review_html.py            the base page, from pipeline_review_data_<sid>.json
      -> patch_review_tool_merge_panel.py   the possible-merges panel (v1)
      -> patch_review_tool_merge_v2.py      dip on shoulders + Poisson; footprint;
                                            firing rates above the merged ACG
      -> patch_review_tool_export_all.py    one button that saves all three decision
                                            files for scripts/sync_session.py
      -> patch_review_tool_merge_v3.py      working verdict buttons; panel at the
                                            bottom; jump to the other unit; twins

Each patch asserts its anchors match exactly once, so if `build_review_html.py` is
edited in a way that moves one, this stops rather than publishing a half-patched page.

INPUTS (build these first, in this order):
    scripts/build_pipeline_review_data.py <sid>    -> outputs/pipeline_review_data_<sid>.json
    scripts/build_session_timeline_data.py <sid>   -> outputs/session_timeline_<sid>.json
    scripts/build_merge_candidates.py <sid>        -> outputs/merge_candidates_<sid>.json

OUTPUT:
    outputs/unit_review_tool_<sid>.html

PUBLISHING: the page fetches its merge data lazily from the relative path
`data/merge_candidates.json`, so it must be published WITH that file:

    Artifact publish  file_path = outputs/unit_review_tool_<sid>.html
                      files     = {"data/merge_candidates.json":
                                   "outputs/merge_candidates_<sid>.json"}
                      capabilities = {"db": {}, "downloads": true}

The page needs `db` for verdicts (collections `trial_selections` and
`merge_verdicts`) and `downloads` for the export button. Publishing to an EXISTING
artifact keeps its URL and therefore its verdicts; publishing without a url makes a
new artifact with an empty database.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
OUT = os.path.join(ROOT, "outputs")

# (script, what it adds) applied in this order; each takes <src> <dst>
PATCHES = [
    ("patch_review_tool_merge_panel.py", "possible-merges panel"),
    ("patch_review_tool_merge_v2.py", "shoulder dip, Poisson test, footprint, rates"),
    ("patch_review_tool_merge_v3.py", "verdict buttons, panel at bottom, unit jump, twins"),
    ("patch_review_tool_export_all.py", "export-all-for-sync button"),
    ("patch_review_tool_sync_command.py", "the sync command on the page, with a copy button"),
]


def run(args, label):
    print("\n=== %s" % label)
    print("    " + " ".join(args))
    r = subprocess.run(args, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit("FAILED: %s (exit %d)" % (label, r.returncode))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    sid = sys.argv[1]
    py = sys.executable

    review = os.path.join(OUT, "pipeline_review_data_%s.json" % sid)
    if not os.path.exists(review):
        raise SystemExit("missing %s -- run scripts/build_pipeline_review_data.py %s first"
                         % (review, sid))
    cands = os.path.join(OUT, "merge_candidates_%s.json" % sid)
    if not os.path.exists(cands):
        print("NOTE: %s does not exist yet. The page will build, but the merge panel\n"
              "      will say its data could not be loaded until you run\n"
              "      scripts/build_merge_candidates.py %s and publish it as\n"
              "      data/merge_candidates.json alongside the page." % (cands, sid))

    base = os.path.join(OUT, "unit_review_pipeline_%s.html" % sid)
    run([py, os.path.join(SCRIPTS, "build_review_html.py"), sid], "base page")
    if not os.path.exists(base):
        raise SystemExit("build_review_html.py did not produce %s" % base)

    src = base
    tmp = []
    for i, (script, what) in enumerate(PATCHES):
        dst = os.path.join(OUT, "_review_tool_stage%d_%s.html" % (i + 1, sid))
        run([py, os.path.join(SCRIPTS, script), src, dst], "patch %d/%d -- %s"
            % (i + 1, len(PATCHES), what))
        if not os.path.exists(dst):
            raise SystemExit("%s did not write %s" % (script, dst))
        src = dst
        tmp.append(dst)

    final = os.path.join(OUT, "unit_review_tool_%s.html" % sid)
    if os.path.exists(final):
        os.remove(final)
    os.rename(src, final)
    tmp = tmp[:-1]
    for t in tmp:
        try:
            os.remove(t)
        except OSError:
            pass

    size = os.path.getsize(final)
    print("\n" + "=" * 70)
    print("wrote %s" % final)
    print("      %d bytes (%.2f MiB of the 16 MiB artifact limit)" % (size, size / 1024.0 / 1024))
    for must in ("renderMergePanel", "mFootprint", "dipStats", "data-mvf", "mergeGoto",
                 "exportAllForSync", "exportAllBtn", "syncCommand", "copyCmdBtn"):
        n = open(final, encoding="utf-8").read().count(must)
        print("      %-18s %d" % (must, n))
        if not n:
            raise SystemExit("patched page is missing %s" % must)
    print("\npublish it WITH its data file:")
    print('  file_path = outputs/unit_review_tool_%s.html' % sid)
    print('  files     = {"data/merge_candidates.json": "outputs/merge_candidates_%s.json"}' % sid)
    print('  capabilities = {"db": {}, "downloads": true}')
    print("  pass the EXISTING artifact url to keep its verdicts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

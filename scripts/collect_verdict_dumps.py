# -*- coding: utf-8 -*-
"""Pick up the review tool's downloaded decision files and put them where sync reads.

    python scripts/collect_verdict_dumps.py <session_id> [--from <dir>]

The review tool's "export all for sync" button saves three files to the browser's
download folder:

    verdicts_<session>.json
    trial_selections_<session>.json
    merge_verdicts_<session>.json

This copies them to `outputs/review_verdicts/<session>/` under the plain names
`sync_session.py` expects, and prints what it found. `sync_session.py
--collect_from_downloads` runs it for you, so the normal path is one command.

WHY IT IS NOT JUST A COPY
-------------------------
Browsers do not overwrite. Export twice and the second file lands as
`verdicts_<session> (1).json`, so a plain copy would silently use the FIRST, older
export -- which is exactly the class of bug this whole chain exists to prevent. So:

- every spelling a browser produces is considered (` (1)`, `(1)`, `-1`, `_1`)
- the NEWEST by modification time wins
- the chosen file's age is printed, and anything older than --max_age_hours is refused
  rather than used, because a stale dump is worse than a missing one
- the contents are validated as the right collection before being written anywhere

It refuses on anything surprising rather than guessing. Nothing is copied unless all
three files pass.
"""
import argparse
import datetime
import glob
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# collection -> (downloaded stem, destination name, how to recognise a document)
WANTED = {
    "verdicts": ("verdicts", "verdicts.json", "verdict"),
    "trial_selections": ("trial_selections", "trial_selections.json", "ranges"),
    "merge_verdicts": ("merge_verdicts", "merge_verdicts.json", "unit_a"),
}
# What a browser appends when a name is already taken. Matched against the FULL
# expected filename, never used to guess where the name ends: the session id contains
# an underscore followed by digits (20260916_110311), so a pattern that hunts for a
# trailing "_<digits>" eats the session's time and silently finds nothing. That is a
# bug this script had, and the test for duplicate handling is what exposed it.
DUP_SUFFIX = r"(?: ?\(\d+\)|[-_]\d+)"


def default_download_dirs():
    home = os.path.expanduser("~")
    return [d for d in (os.path.join(home, "Downloads"),
                        os.path.join(home, "Download"),
                        os.environ.get("CLAUDE_DOWNLOADS", "")) if d and os.path.isdir(d)]


def candidates(folder, stem, sid):
    """Every file in `folder` that is a download of `<stem>_<sid>.json`.

    Exact name, or exact name plus one browser duplicate suffix. Anchored both ends so
    `verdicts_<sid>` cannot also claim `merge_verdicts_<sid>` (the stems overlap).
    """
    expected = "%s_%s" % (stem, sid)
    pat = re.compile(r"^" + re.escape(expected) + DUP_SUFFIX + r"$")
    out = []
    for path in glob.glob(os.path.join(folder, "*.json")):
        base = os.path.basename(path)[:-len(".json")]
        if base == expected or pat.match(base):
            out.append(path)
    return sorted(out, key=os.path.getmtime, reverse=True)


def documents(doc):
    """Normalise to a list of documents, whatever shape the dump arrived in."""
    if isinstance(doc, dict):
        inner = doc.get("selections", doc.get("merges", doc))
        if isinstance(inner, dict):
            return [v.get("data", v) if isinstance(v, dict) else v
                    for v in inner.values()]
        doc = inner
    return [r.get("data", r) if isinstance(r, dict) else r for r in (doc or [])]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id")
    ap.add_argument("--from", dest="src", default=None,
                    help="where the browser saved them (default: your Downloads folder)")
    ap.add_argument("--dest", default=None,
                    help="default outputs/review_verdicts/<session_id>")
    ap.add_argument("--max_age_hours", type=float, default=24.0,
                    help="refuse a dump older than this. A stale export silently syncs "
                         "yesterday's decisions, so it is refused rather than used. "
                         "0 disables the check.")
    args = ap.parse_args()

    sid = args.session_id
    folders = [args.src] if args.src else default_download_dirs()
    folders = [f for f in folders if f and os.path.isdir(f)]
    if not folders:
        raise SystemExit("no download folder found. Pass --from <dir>.")
    dest = args.dest or os.path.join(ROOT, "outputs", "review_verdicts", sid)

    print("looking for the review tool's export of %s" % sid)
    for f in folders:
        print("  in %s" % f)

    now = datetime.datetime.now()
    chosen, problems = {}, []
    for coll, (stem, out_name, marker) in sorted(WANTED.items()):
        found = []
        for f in folders:
            found.extend(candidates(f, stem, sid))
        found.sort(key=os.path.getmtime, reverse=True)
        if not found:
            problems.append("%s: no %s_%s.json in %s"
                            % (coll, stem, sid, " or ".join(folders)))
            continue
        path = found[0]
        age_h = (now - datetime.datetime.fromtimestamp(os.path.getmtime(path))).total_seconds() / 3600.0

        try:
            docs = documents(json.load(open(path)))
        except ValueError as exc:
            problems.append("%s: %s is not valid JSON (%s)" % (coll, path, exc))
            continue
        if docs and not any(isinstance(d, dict) and marker in d for d in docs):
            problems.append("%s: %s has %d document(s) but none carries %r -- is this "
                            "the right collection?" % (coll, path, len(docs), marker))
            continue

        note = ""
        if len(found) > 1:
            note = "  (newest of %d; ignoring %s)" % (
                len(found), ", ".join(os.path.basename(p) for p in found[1:4]))
        print("  %-17s %-46s %4d doc(s)  %5.1f h old%s"
              % (coll, os.path.basename(path), len(docs), age_h, note))

        if args.max_age_hours and age_h > args.max_age_hours:
            problems.append("%s: %s is %.1f h old (limit %.1f). Re-export from the "
                            "review tool, or pass --max_age_hours 0 if you really mean "
                            "to use it." % (coll, os.path.basename(path), age_h,
                                            args.max_age_hours))
            continue
        chosen[coll] = (path, out_name, len(docs))

    if problems:
        print("\nREFUSING TO COPY -- nothing was written:\n")
        for p in problems:
            print("  %s" % p)
        print("\nIn the review tool, press \"export all for sync\" and accept all three "
              "download prompts.")
        return 2

    if not os.path.isdir(dest):
        os.makedirs(dest)
    print("")
    for coll, (path, out_name, n) in sorted(chosen.items()):
        target = os.path.join(dest, out_name)
        shutil.copy2(path, target)
        print("  %-17s -> %s  (%d documents)" % (coll, target, n))
    print("\nReady. Now:\n  python scripts/sync_session.py %s --rec <rec_folder> "
          "--rat_root <data_root>" % sid)
    return 0


if __name__ == "__main__":
    sys.exit(main())

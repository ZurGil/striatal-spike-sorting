"""Stop a transient document-store error from reading as "this unit has no raw data".

The bug, seen on unit 51 of 20260915_084812_103710_concat
---------------------------------------------------------
The raw-spikes panel cached the RESULT of its fetch, including the failures:

    try{ ... }catch(e){ data = null; fetchErr = e; }
    rawSpikesCache[uid] = data;        // null, whether absent or unreachable

`fetchErr` is a local, so it is gone by the next render. From then on `data` is
null rather than undefined, the panel never retries, and it falls through to the
branch that interprets null as "doc does not exist". One bad moment therefore
became a permanent and factually wrong claim about the data -- the document was
on the server the whole time, 16 snippets of it, readable at viewer access level.
What the store had actually answered was:

    unavailable: The document store is temporarily unavailable; retry later.

which is the one thing the panel then refused to tell anyone.

The fix has three parts, in order of importance:
  1. never cache a failed fetch -- only a real answer, present or absent, is
     remembered. An error is not evidence about whether a document exists.
  2. retry with backoff, since "temporarily unavailable" usually is.
  3. say which of the two it was, and offer a button, so the reader is not left
     reloading the page to find out.

Usage:
    python scripts/patch_review_tool_raw_trace_retry.py \
        scripts/build_review_html.py \
        outputs/unit_review_tool_20260915_084812_103710_concat.html

Each file is patched in place after a check that the old block is present
exactly once -- a built tool is 15 MB of one-shot HTML and a partial match
would be far worse than no match.
"""
from __future__ import print_function

import io
import os
import sys

OLD = """async function renderRawSpikesPanel(container, uid){
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
"""

NEW = """async function fetchRawTraces(uid, attempts){
  /* The document store answers `unavailable` from time to time. That is a
     passing server state, not a statement about this unit, so try again before
     giving up -- and hand the caller enough to tell a failure apart from a
     document that genuinely is not there. */
  let lastErr = null;
  for(let i = 0; i < attempts; i++){
    try{
      const snap = await db.collection('raw_traces').doc(String(uid)).get();
      return { ok: true, data: snap.exists ? snap.data() : null };
    }catch(e){
      lastErr = e;
      console.error('raw_traces fetch failed for unit ' + uid +
                    ' (attempt ' + (i+1) + ' of ' + attempts + ')', e);
      if(i < attempts - 1) await new Promise(r => setTimeout(r, 400 * Math.pow(2, i)));
    }
  }
  return { ok: false, err: lastErr };
}

async function renderRawSpikesPanel(container, uid){
  container.innerHTML = '<div class="no-data">loading raw traces&hellip;</div>';
  if(!db){ container.innerHTML = '<div class="no-data">shared database unavailable in this view (claude.use("db") did not resolve)</div>'; return; }
  let data = rawSpikesCache[uid];
  if(data === undefined){
    const res = await fetchRawTraces(uid, 4);
    if(!res.ok){
      /* Deliberately NOT cached. A fetch that failed says nothing about whether
         the document exists, and remembering it would turn one bad moment into a
         permanent "no data" for this unit. */
      const e = res.err || {};
      const detail = ((e.code || '') + ' ' + (e.message || String(res.err))).trim();
      container.innerHTML = `<div class="no-data">could not reach the document store for unit ${uid} &mdash; ${detail}<br>` +
        `<span style="opacity:.75">The traces are stored; this is the store being unreachable, not a unit without data.</span></div>`;
      const again = document.createElement('button');
      again.textContent = 'Try again';
      again.style.cssText='font-size:10px;padding:4px 12px;border-radius:6px;border:1px solid var(--accent);background:var(--accent-soft);color:var(--accent);cursor:pointer;font-weight:600;margin-top:6px;';
      again.addEventListener('click', () => renderRawSpikesPanel(container, uid));
      container.appendChild(again);
      return;
    }
    data = res.data;          /* only a real answer, present or absent, is remembered */
    rawSpikesCache[uid] = data;
  }
"""


def patch(path):
    with io.open(path, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()

    crlf = "\r\n" in text
    old = OLD.replace("\n", "\r\n") if crlf else OLD
    new = NEW.replace("\n", "\r\n") if crlf else NEW

    n = text.count(old)
    if n != 1:
        if text.count(new.split("\n")[0].rstrip("\r")) > 0:
            print("%s: already patched, leaving it alone" % path)
            return 0
        raise SystemExit("%s: expected the old block exactly once, found %d. "
                         "Refusing to touch a 15 MB one-shot file on a partial "
                         "match." % (path, n))

    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text.replace(old, new))
    print("%s: patched (%s line endings, %d bytes)"
          % (path, "CRLF" if crlf else "LF", os.path.getsize(path)))
    return 1


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for path in sys.argv[1:]:
        patch(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

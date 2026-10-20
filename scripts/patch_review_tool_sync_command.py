# -*- coding: utf-8 -*-
"""Show the sync command on the page, next to the export button, ready to copy.

    python scripts/patch_review_tool_sync_command.py <src.html> <dst.html>

The export button hands over three files; a command then applies them. Retyping that
command from a chat message is where it goes wrong, so the page prints it with the
session id already filled in and a copy button next to it.

The command is built to actually work when pasted into cmd.exe, which means three
things a hand-written version tends to miss:

  * `cd /d <repo>` first, because the script is referenced by a relative path
  * the FULL path to the right interpreter. `python` on PATH here is the base conda
    install, which has no pyarrow, so the pipeline would die writing its first
    parquet. The pipeline runs under the phy2_ky environment.
  * `--rat_root`, because the Bpod behaviour lives with the archive on Z: while the
    ephys works from F:, and without it the session cannot be auto-matched.

Those four paths are machine-specific, so they are constants right here. If the repo
moves, the environment is rebuilt, or another rat is recorded, change them in this one
place and rebuild the tool.
"""
import io
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else None
DST = sys.argv[2] if len(sys.argv) > 2 else None
if not SRC or not DST:
    raise SystemExit(__doc__)

REPO_DIR = u"D:\\\\Gil\\\\spike_sorting_agent"
PYTHON = u"C:\\\\Users\\\\Adam\\\\anaconda3\\\\envs\\\\phy2_ky\\\\python.exe"
REC_ROOT = u"F:\\\\Gil\\\\Shamir"
RAT_ROOT = u"Z:\\\\Gil\\\\Shamir_1"

# --- 1. one definition of the command, used by the panel and the alert -------------
A1_OLD = u"""/* Write all three decision collections to disk, named as collect_verdict_dumps.py"""
A1_NEW = u"""/* The command that finishes the job, with this session's id already in it.
   Written to be pasted into cmd.exe as-is: `cd /d` because the script path is
   relative, and the FULL interpreter path because `python` on PATH is the base conda
   install with no pyarrow -- the pipeline would run and then die writing its first
   parquet file. --rat_root is required because the Bpod behaviour sits with the
   archive on Z: while the ephys works from F:. */
function syncCommand(){
  return 'cd /d %(repo)s && "%(python)s" scripts\\\\sync_session.py ' +
         SESSION_ID +
         ' --rec "%(rec)s\\\\' + SESSION_ID + '.rec"' +
         ' --rat_root "%(rat)s"' +
         ' --collect_from_downloads';
}

async function copySyncCommand(){
  const cmd = syncCommand();
  const btn = document.getElementById('copyCmdBtn');
  let ok = false;
  try { await navigator.clipboard.writeText(cmd); ok = true; }
  catch(e){
    /* clipboard refused (not focused, or no permission): select it so Ctrl+C works */
    try {
      const el = document.getElementById('syncCmd');
      const r = document.createRange(); r.selectNodeContents(el);
      const s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
    } catch(e2){}
  }
  if(btn){
    btn.textContent = ok ? 'copied' : 'selected \\u2014 press Ctrl+C';
    setTimeout(() => { btn.textContent = 'copy'; }, 2200);
  }
}

/* Write all three decision collections to disk, named as collect_verdict_dumps.py""" % {
    "repo": REPO_DIR, "python": PYTHON, "rec": REC_ROOT, "rat": RAT_ROOT}

# --- 2. the panel ------------------------------------------------------------------
A2_OLD = u"""      <span id="exportAllCount" style="font-size:10px;color:var(--text-3);"></span>
    </div>
    <div class="list" id="list"></div>"""
A2_NEW = u"""      <span id="exportAllCount" style="font-size:10px;color:var(--text-3);"></span>
    </div>
    <div style="padding:0 16px 9px;border-bottom:1px solid var(--border);">
      <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:3px;">
        <span style="font-size:9px;color:var(--text-3);text-transform:uppercase;letter-spacing:.05em;">
          2 \u00b7 then paste this into cmd</span>
        <button id="copyCmdBtn" style="font-size:9px;padding:1px 8px;border-radius:5px;
                border:1px solid var(--border);background:var(--surface);color:var(--text-2);
                cursor:pointer;font-family:'IBM Plex Mono',monospace;">copy</button>
      </div>
      <code id="syncCmd" class="mono" style="display:block;font-size:9.5px;line-height:1.5;
            color:var(--text-2);background:var(--surface-2);border:1px solid var(--border);
            border-radius:6px;padding:6px 8px;white-space:pre-wrap;word-break:break-all;
            user-select:all;"></code>
    </div>
    <div class="list" id="list"></div>"""

# --- 3. fill it in, and label the button as step 1 ---------------------------------
A3_OLD = u"""        \u2193 export all for sync</button>"""
A3_NEW = u"""        \u2193 1 \u00b7 export all for sync</button>"""

A4_OLD = u"""function refreshExportAllCount(){
  try {
    const el = document.getElementById('exportAllCount');
    if(!el) return;"""
A4_NEW = u"""function refreshExportAllCount(){
  try {
    const cmdEl = document.getElementById('syncCmd');
    if(cmdEl && !cmdEl.textContent) cmdEl.textContent = syncCommand();
    const el = document.getElementById('exportAllCount');
    if(!el) return;"""

A5_OLD = u"""  const eab = document.getElementById('exportAllBtn');
  if(eab) eab.addEventListener('click', exportAllForSync);"""
A5_NEW = u"""  const eab = document.getElementById('exportAllBtn');
  if(eab) eab.addEventListener('click', exportAllForSync);
  const ccb = document.getElementById('copyCmdBtn');
  if(ccb) ccb.addEventListener('click', copySyncCommand);
  const sc = document.getElementById('syncCmd');
  if(sc) sc.textContent = syncCommand();"""

PATCHES = [
    (A1_OLD, A1_NEW, "syncCommand() + copySyncCommand()"),
    (A2_OLD, A2_NEW, "command panel"),
    (A3_OLD, A3_NEW, "button labelled step 1"),
    (A4_OLD, A4_NEW, "fill the command on render"),
    (A5_OLD, A5_NEW, "copy button wiring"),
]

# The success alert printed its own, slightly different command (no interpreter path).
# Two spellings of the same thing is how one of them goes stale, so the alert is
# rewritten to use syncCommand(). Located by its first and last line rather than by
# quoting the whole block, whose nested escapes are a liability in an anchor.
ALERT_FROM = u"msg += 'Now run, on the recording PC:"
ALERT_TO = u"'one reached synced/.';"
ALERT_NEW = u"""msg += 'Now run this on the recording PC -- it is on the page under the ' +
           'export button, with a copy button:\\n\\n' + syncCommand() + '\\n\\n' +
           'That applies every decision, re-runs the pipeline, and verifies that each ' +
           'one reached synced/.';"""


def main():
    s = io.open(SRC, encoding="utf-8").read()
    for old, new, label in PATCHES:
        n = s.count(old)
        if n != 1:
            raise SystemExit("anchor %r matched %d times (expected 1) in %s -- fix the "
                             "anchor rather than publishing a half-patched page."
                             % (label, n, SRC))
        s = s.replace(old, new)
        print("  ok: %s" % label)

    if s.count(ALERT_FROM) != 1 or s.count(ALERT_TO) != 1:
        raise SystemExit("alert block: %d start and %d end marker(s), expected 1 each"
                         % (s.count(ALERT_FROM), s.count(ALERT_TO)))
    i = s.index(ALERT_FROM)
    j = s.index(ALERT_TO, i) + len(ALERT_TO)
    s = s[:i] + ALERT_NEW + s[j:]
    print("  ok: alert uses syncCommand()")

    io.open(DST, "w", encoding="utf-8").write(s)
    print("wrote %s" % DST)


if __name__ == "__main__":
    main()

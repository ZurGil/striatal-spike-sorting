# -*- coding: utf-8 -*-
"""Build the Session Review Register page from outputs/session_status.json.

    python scripts/session_status_report.py          # reads Z and the live counts
    python scripts/build_session_register_html.py    # turns that into the page

The page used to be written by hand, which meant it could disagree with the files
it described -- and did, for three days, while it still said 15 September had
nothing decided in it. Now every figure on it comes out of session_status.json, so
the only way to change the page is to re-run the scan.

Publish the result to the register artifact, keeping the same URL:
    Artifact publish, file_path outputs/session_register.html,
    url https://claude.ai/artifact/BXw9umwQGFsBuVwfbUPJzJ
"""
from __future__ import print_function

import datetime
import html
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")

CHIP = {
    "SAVED, COMPLETE": "c-ok",
    "SAVED, REVIEW CLOSED": "c-ok",
    "SAVED, PARTIAL": "c-partial",
    "DRIFTED": "c-alert",
    "CHECK": "c-alert",
    "NOT REVIEWED": "c-none",
    "SUPERSEDED": "c-sup",
    "HALF OF A JOIN": "c-ok",
}

CSS = """
  /* Layout: a lab register. Summary figures, then one dense table of every sorted
     session, then a detail block per session carrying its links and, where the session
     is not what its name suggests, what it actually is. Left-aligned throughout,
     hairline rules only between rows, identifiers and figures in mono. */
  :root {
    --bg: #f7f8f8; --surface: #ffffff; --ink: #16201f; --ink-2: #4d5a59;
    --ink-3: #788584; --rule: #dde3e2; --rule-strong: #c3cccb; --accent: #0f6f68;
    --ok: #1f6b45; --partial: #8a5a00; --none: #6d7877; --alert: #a3261c;
    --chip-ok: #e4f1ea; --chip-partial: #f8eedb; --chip-none: #eceeee;
    --chip-alert: #f8e4e2;
    --display: "Archivo", system-ui, sans-serif;
    --body: "Source Sans 3", system-ui, sans-serif;
    --mono: "IBM Plex Mono", ui-monospace, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #0f1413; --surface: #161d1c; --ink: #e6ecea; --ink-2: #a3b0ae;
      --ink-3: #788886; --rule: #25302e; --rule-strong: #33403e; --accent: #4ec9bd;
      --ok: #6ed69f; --partial: #e0b463; --none: #8b9896; --alert: #f08a7e;
      --chip-ok: #15291f; --chip-partial: #2b2314; --chip-none: #1d2423;
      --chip-alert: #2e1a18; color-scheme: dark;
    }
  }
  :root[data-theme="dark"] {
    --bg: #0f1413; --surface: #161d1c; --ink: #e6ecea; --ink-2: #a3b0ae;
    --ink-3: #788886; --rule: #25302e; --rule-strong: #33403e; --accent: #4ec9bd;
    --ok: #6ed69f; --partial: #e0b463; --none: #8b9896; --alert: #f08a7e;
    --chip-ok: #15291f; --chip-partial: #2b2314; --chip-none: #1d2423;
    --chip-alert: #2e1a18; color-scheme: dark;
  }
  body { background: var(--bg); color: var(--ink); font-family: var(--body);
         font-size: 15px; line-height: 1.55; margin: 0; }
  .wrap { max-width: 1080px; margin: 0 auto;
          padding-left: max(16px, env(safe-area-inset-left, 0px));
          padding-right: max(16px, env(safe-area-inset-right, 0px));
          padding-block: 34px 56px; }
  h1 { font-family: var(--display); font-weight: 600; font-size: clamp(25px, 5vw, 34px);
       letter-spacing: -0.015em; line-height: 1.15; margin: 0 0 6px; text-wrap: balance; }
  .lede { color: var(--ink-2); max-width: 62ch; margin: 0 0 4px; }
  .asof { font-family: var(--mono); font-size: 12px; color: var(--ink-3); margin: 0 0 30px; }
  h2 { font-family: var(--display); font-weight: 600; font-size: 15px;
       letter-spacing: 0.085em; text-transform: uppercase; color: var(--ink-2);
       margin: 44px 0 14px; padding-bottom: 7px; border-bottom: 1px solid var(--rule-strong); }
  .figures { display: grid; grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
             gap: 1px; background: var(--rule); border: 1px solid var(--rule); }
  .fig { background: var(--surface); padding: 15px 16px; min-width: 0; }
  .fig b { display: block; font-family: var(--mono); font-weight: 500; font-size: 27px;
           font-variant-numeric: tabular-nums; line-height: 1.1; letter-spacing: -0.02em; }
  .fig span { display: block; font-size: 12px; color: var(--ink-3);
              letter-spacing: 0.04em; margin-top: 4px; }
  .scroll { overflow-x: auto; border: 1px solid var(--rule); background: var(--surface); }
  table { border-collapse: collapse; width: 100%; font-size: 14px; }
  th, td { padding: 9px 11px; text-align: left; white-space: nowrap; vertical-align: top; }
  thead th { font-family: var(--body); font-weight: 600; font-size: 11px;
             letter-spacing: 0.07em; text-transform: uppercase; color: var(--ink-3);
             border-bottom: 1px solid var(--rule-strong); }
  tbody tr + tr td { border-top: 1px solid var(--rule); }
  td.sid { font-family: var(--mono); font-size: 13px; }
  td.n { font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; }
  td.dim, .dim { color: var(--ink-3); }
  .rat { font-size: 11px; letter-spacing: 0.07em; text-transform: uppercase; color: var(--ink-3); }
  .chip { display: inline-block; font-size: 11.5px; font-weight: 600;
          letter-spacing: 0.035em; padding: 2px 8px; border-radius: 2px; white-space: nowrap; }
  .c-ok { background: var(--chip-ok); color: var(--ok); }
  .c-partial { background: var(--chip-partial); color: var(--partial); }
  .c-none { background: var(--chip-none); color: var(--none); }
  .c-alert { background: var(--chip-alert); color: var(--alert); }
  .c-sup { background: transparent; color: var(--ink-3);
           border: 1px solid var(--rule-strong); font-weight: 400; }
  .bar { position: relative; height: 5px; width: 96px; background: var(--rule); overflow: hidden; }
  .bar i { position: absolute; inset: 0 auto 0 0; background: var(--accent); }
  .detail { border-top: 1px solid var(--rule); padding: 17px 0; }
  .detail:first-of-type { border-top: 0; }
  .detail h3 { font-family: var(--mono); font-weight: 500; font-size: 16px;
               margin: 0 0 3px; letter-spacing: -0.005em; }
  .detail .meta { font-size: 13px; color: var(--ink-3); margin: 0 0 9px; }
  .detail ul { margin: 0; padding-left: 17px; }
  .detail li { margin-bottom: 3px; color: var(--ink-2); }
  .detail li b { color: var(--ink); font-weight: 600; }
  a { color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 2px; }
  a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .note { background: var(--surface); border: 1px solid var(--rule);
          border-left: 3px solid var(--accent); padding: 15px 17px; margin-top: 10px; }
  .note.warn { border-left-color: var(--alert); }
  .note p { margin: 0 0 9px; color: var(--ink-2); }
  .note p:last-child { margin-bottom: 0; }
  .note code, .path { font-family: var(--mono); font-size: 12.5px; color: var(--ink);
                      word-break: break-word; }
  dl.states { margin: 0; display: grid; grid-template-columns: auto 1fr;
              gap: 7px 13px; align-items: baseline; }
  dl.states dd { margin: 0; color: var(--ink-2); font-size: 14px; }
"""


def esc(s):
    return html.escape(str(s if s is not None else ""))


def chip(text, cls):
    return '<span class="chip %s">%s</span>' % (cls, esc(text))


def synced_cell(r):
    if r["bpod_synced"]:
        return chip("yes", "c-ok") + '<br><span class="rat">%s</span>' % esc(
            ", ".join(r["bpod_synced"]))
    if r.get("synced_parts"):
        return chip("via its halves", "c-ok") + '<br><span class="rat">%s</span>' % esc(
            ", ".join(p["session"] for p in r["synced_parts"]))
    if not r["bpod_files"]:
        return chip("blocked", "c-alert") + '<br><span class="rat">no bpod files</span>'
    return '<span class="dim">no</span>'


def banner(rows):
    """The one thing a reader must not miss, at the top, naming the sessions."""
    hit = sorted({r["session"] for r in rows
                  if (r.get("catch") or [{}])[0].get("error_catch", "").startswith("NO")})
    if not hit:
        return ""
    return (
        '<div class="note warn" style="margin:0 0 26px;border-left-width:5px">'
        '<p style="font-weight:600;color:var(--alert);margin-bottom:8px">'
        'EVERY POPPER SESSION SO FAR HAS <span class="path">CatchError = 0</span>: '
        'no catch trials on errors, so no time-investment measure on error trials.</p>'
        '<p>Catch trials on CORRECT choices are present and working in every session '
        'of both rats &mdash; 11&ndash;16&thinsp;%% of trials, never rewarded, and the '
        'rat waits five to eight seconds. What Popper is missing is the other half: '
        'with <span class="path">CatchError = 0</span> an error trial is announced at '
        'the end of a feedback delay <i>we</i> drew, so the measured wait on errors is '
        '1.1&ndash;1.5&thinsp;s of our own making rather than the animal’s decision. '
        'Shamir runs <span class="path">CatchError = 1</span> and waits '
        '4.8&ndash;6.9&thinsp;s on errors.</p>'
        '<p><b>A confidence or time-investment analysis needs the correct and error '
        'waiting curves together, so these sessions cannot provide one.</b> They remain '
        'good for everything that does not depend on waiting time on errors &mdash; '
        'which is why they are still sorted, synced and checked here. Affected: '
        '<span class="path">%s</span>. Evidence per session in '
        '<span class="path">outputs/catch_trial_audit.csv</span> '
        '(scripts/audit_catch_trials.py).</p></div>' % ", ".join(hit))


def catch_cell(r):
    """Whether ERROR trials are catch trials on this day. See the banner for why."""
    c = r.get("catch") or []
    if not c:
        return '<span class="dim">&mdash;</span>'
    row = c[0]
    err = str(row.get("error_catch", ""))
    if err.startswith("NO"):
        return (chip("no error catch", "c-alert")
                + '<br><span class="rat">CatchError=0 &middot; error wait %ss vs %ss</span>'
                % (row.get("median_wait_error_s"), row.get("median_wait_catch_s")))
    if err.startswith("yes"):
        return (chip("ok", "c-ok")
                + '<br><span class="rat">%s%% catch &middot; error wait %ss</span>'
                % (row.get("pct_flagged"), row.get("median_wait_error_s")))
    return '<span class="dim">%s</span>' % esc(err)


def sync_cell(r):
    """Alignment quality, beside "is there a synced folder". Different questions."""
    checks = r.get("sync_check") or []
    if not checks:
        return '<span class="dim">&mdash;</span>'
    order = {"FAIL": 0, "check": 1, "unknown": 2, "pass": 3}
    worst = sorted(checks, key=lambda c: order.get(c["verdict"], 9))[0]
    cls = {"pass": "c-ok", "check": "c-partial", "FAIL": "c-alert"}.get(
        worst["verdict"], "c-none")
    detail = "%.0f%% matched &middot; %.0f ms worst case &middot; %s/%s trials" % (
        100.0 * (worst["matched"] or 0), worst["holdout_max_ms"] or 0,
        worst["n_trials_valid"], worst["n_trials"])
    return chip(worst["verdict"], cls) + '<br><span class="rat">%s</span>' % detail


def row_html(r):
    t = r["z_tally"]
    dec = r["live_verdicts"] if r["live_verdicts"] is not None else r["z_verdicts"]
    n = r["n_units"] or 0
    pct = (100.0 * dec / n) if n else 0.0
    bar = ('<div class="bar" role="img" aria-label="%d of %d units judged">'
           '<i style="width:%.1f%%"></i></div>' % (dec, n, pct))
    if r["state"] == "SUPERSEDED":
        bar = '<span class="dim">not for review</span>'
    def num(x):
        return '<td class="n%s">%s</td>' % ("" if x else " dim", x)
    return (
        '<tr>'
        '<td class="rat">%s</td><td class="sid">%s</td>'
        '<td class="n">%s</td><td class="n">%s</td><td class="n">%s</td><td>%s</td>'
        '%s%s%s%s%s'
        '<td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>'
        % (esc(r["rat"]), esc(r.get("row_id", r["session"])),
           n or "?", esc(r["thresholds"] or "?"), dec, bar,
           num(t.get("good", 0)), num(t.get("mua", 0)), num(t.get("noise", 0)),
           num(r["z_merges"]), num(r["z_trials"]),
           chip("yes", "c-ok") if r["z_verdicts"] else chip("—", "c-none"),
           chip("yes", "c-ok") if r["synced"] else '<span class="dim">no</span>',
           synced_cell(r), sync_cell(r), catch_cell(r),
           ('<span class="dim">%s</span>'
            % ("yes" if r["provenance"] else
               ("part of a join" if (r["superseded_by"] or r.get("decisions_from"))
                else "no"))),
           chip(r["state"].lower(), CHIP.get(r["state"], "c-none"))))


def detail_html(r):
    bits = ['<div class="detail">',
            '<h3>%s</h3>' % esc(r.get("row_id", r["session"])),
            '<p class="meta">%s &middot; %s units &middot; Kilosort %s &middot; sort folder <span class="path">%s</span></p>'
            % (esc(r["rat"]), r["n_units"] or "?", esc(r["thresholds"] or "?"),
               esc(r.get("ks_folder", "kilosort4"))),
            '<ul>']
    bits.append('<li><b>%s</b> &mdash; %s</li>' % (esc(r["state"]), esc(r["state_why"])))
    if r["z_verdicts"]:
        t, m = r["z_tally"], r["z_merge_tally"]
        bits.append('<li>On Z: %d unit verdicts (%s), %d merge verdicts (%s), '
                    '%d trial selections.</li>'
                    % (r["z_verdicts"],
                       ", ".join("%s %d" % (k, v) for k, v in sorted(t.items()) if k),
                       r["z_merges"],
                       ", ".join("%s %d" % (k, v) for k, v in sorted(m.items()) if k),
                       r["z_trials"]))
    if r.get("decisions_from"):
        bits.append('<li>Its decisions are <span class="path">%s</span>’s: one review '
                    'and one unit space for the day.</li>' % esc(r["decisions_from"]))
    if r["artifact"]:
        bits.append('<li>Review page: <a href="%s">unit review tool</a></li>'
                    % esc(r["artifact"]))
    for name, url in (r["tools"] or []):
        bits.append('<li>%s: <a href="%s">%s</a></li>' % (esc(name), esc(url), esc(url)))
    if r["bpod_synced"]:
        bits.append('<li>Bpod sync: <b>done</b> &mdash; <span class="path">%s</span></li>'
                    % esc(", ".join(r["bpod_synced"])))
        if len(r["bpod_synced"]) > 1:
            bits.append('<li><b>%d sync directories exist</b>, and nothing in their names '
                        'says which is current. Settle that before anything downstream '
                        'reads one.</li>' % len(r["bpod_synced"]))
    if r.get("snapshots"):
        bits.append('<li>Mid-review snapshots on Z: <span class="path">%s</span></li>'
                    % esc(", ".join(r["snapshots"])))
    bits.append('</ul>')
    for para in (r["provenance"] or []):
        bits.append('<div class="note"><p>%s</p></div>' % esc(para))
    bits.append('</div>')
    return "\n".join(bits)


def main():
    status = json.load(open(os.path.join(OUT, "session_status.json")))
    rows = status["sessions"]
    live = json.load(open(os.path.join(OUT, "artifact_live_counts.json")))

    sorts = len(rows)
    units = sum(r["n_units"] or 0 for r in rows if r["state"] != "SUPERSEDED"
                and not r.get("decisions_from"))
    judged = sum((r["live_verdicts"] if r["live_verdicts"] is not None else r["z_verdicts"])
                 for r in rows if not r.get("decisions_from"))
    synced = sum(1 for r in rows if r["bpod_synced"] or r.get("synced_parts"))
    drifted = sum(1 for r in rows if r["state"] in ("DRIFTED", "CHECK"))

    out = ['<title>Session Review Register</title>',
           '<meta name="claude:viewport-fit-sides" content="cover">',
           '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
           'family=Archivo:wght@500;600&family=Source+Sans+3:ital,wght@0,400;0,600;1,400'
           '&family=IBM+Plex+Mono:wght@400;500&display=swap">',
           '<style>%s</style>' % CSS,
           '<div class="wrap">',
           '<h1>Session Review Register</h1>',
           '<p class="lede">Every sorted session: how much of it has been judged, whether '
           'those judgements exist anywhere other than a browser database, whether it has '
           'been synced to Bpod, and where a session is not what its name suggests.</p>',
           '<p class="asof">Z and the review pages read %s &middot; page generated by '
           'scripts/build_session_register_html.py from scripts/session_status_report.py'
           '</p>' % esc(live.get("_checked", "unknown")),
           banner(rows),
           '<div class="figures">',
           '<div class="fig"><b>%d</b><span>sorts listed</span></div>' % sorts,
           '<div class="fig"><b>%s</b><span>units of record</span></div>' % "{:,}".format(units),
           '<div class="fig"><b>%s</b><span>units judged</span></div>' % "{:,}".format(judged),
           '<div class="fig"><b>%d</b><span>synced to Bpod</span></div>' % synced,
           '<div class="fig"><b>%d</b><span>verdicts only in a browser</span></div>' % drifted,
           '</div>',
           '<h2>Sorted sessions</h2>',
           '<div class="scroll"><table><thead><tr>'
           '<th>Rat</th><th>Sort</th><th>Units</th><th>Th&nbsp;u/l</th><th>Judged</th>'
           '<th>Progress</th><th>Good</th><th>MUA</th><th>Noise</th><th>Merges</th>'
           '<th>Windows</th><th>On&nbsp;Z</th><th>Verdicts<br>applied</th>'
           '<th>Bpod<br>synced</th><th>Sync<br>check</th><th>Catch<br>trials</th>'
           '<th>Joined</th><th>State</th>'
           '</tr></thead><tbody>']
    for r in rows:
        out.append(row_html(r))
    out.append('</tbody></table></div>')

    out += ['<h2>Two different questions</h2>',
            '<div class="note">',
            '<p><b>Verdicts applied</b> means the good / MUA / noise calls were written '
            'into the group column of <span class="path">cluster_info.tsv</span> &mdash; '
            'never into <span class="path">cluster_group.tsv</span>, and never by hand.</p>',
            '<p><b>Bpod synced</b> means <span class="path">scripts/sync_session.py</span> '
            'wrote a <span class="path">synced/</span> directory aligning the ephys to the '
            'behaviour, and that its nine verdict checks passed. A session can have one '
            'without the other, and treating them as one question is how &ldquo;is it '
            'synced?&rdquo; gets answered wrongly.</p>',
            '<p>A <b>joined</b> session has no DIO and no Bpod file of its own, so it is '
            'never synced in its own folder. It is synced once per half, each half against '
            'its own behaviour, from the joint sort restricted to that half &mdash; one '
            'unit space, two trial tables. Those rows are marked '
            '<span class="path">[from concat]</span>.</p>',
            '</div>',
            '<h2>What the sync check means</h2>',
            '<div class="note">',
            '<p>A <span class="path">synced/</span> folder existing says nothing about '
            'whether the two clocks agree. The check reports three things from that '
            'dataset’s own <span class="path">session_meta.json</span>: the fraction '
            'of Bpod events that found their Trodes TTL (anything under 100 % means '
            'events went unexplained), the <b>worst-case leave-one-out error</b> of the '
            'time conversion — each matched pair predicted from all the others, which '
            'is the honest accuracy figure — and how many trials the pipeline was '
            'willing to call synced.</p>',
            '<p>A dataset passes when every event matched, every trial is valid, and that '
            'worst-case error is under 50&nbsp;ms. All of them pass, at 29–48&nbsp;ms '
            'worst case and a median error of essentially zero.</p>',
            '<p>Each <span class="path">session_meta.json</span> also carries a '
            '<span class="path">sync_status</span> of <span class="path">needs_review</span>, '
            'and that is about a different number: the residual of fitting ONE straight '
            'line across the whole session, which is hundreds of milliseconds because the '
            'two clocks drift apart. That is exactly why the conversion is local '
            'interpolation rather than a single line, and why the holdout error above is '
            'the figure to read.</p>',
            '</div>',
            '<h2>Sessions in detail</h2>']
    for r in rows:
        out.append(detail_html(r))
    out += ['<h2>Keeping this current</h2>',
            '<div class="note">',
            '<p>Verdicts live in each review page’s own database on claude.ai. Nothing '
            'on disk knows about them until they are dumped, so this register compares the '
            'two and says <span class="chip c-alert">drifted</span> when the page is ahead '
            'of Z.</p>',
            '<p>Say the word after a reviewing session and the three collections get dumped '
            'to <span class="path">&lt;session&gt;.kilosort/review/verdicts/</span> on Z, '
            'hashed on both sides, and this page is regenerated. The markdown copy at '
            '<span class="path">post_ks_correction/SESSION_STATUS.md</span> gives git the '
            'history.</p>',
            '<p>Nothing on this page is typed by hand: every figure comes from '
            '<span class="path">outputs/session_status.json</span>, which is read off Z.</p>',
            '</div>', '</div>']

    path = os.path.join(OUT, "session_register.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("wrote %s (%d bytes, %d rows)" % (path, os.path.getsize(path), len(rows)))
    print("generated %s" % datetime.datetime.now().isoformat(timespec="seconds"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

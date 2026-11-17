# -*- coding: utf-8 -*-
r"""Build the review tool for one session. One command, no arguments to remember.

    python scripts/build_review_for_session.py <session_id>

WHY THIS EXISTS
---------------
Every session's review tool was being built by hand, and every build needed a
correction afterwards that the previous build had already taught us:

  * `--max_pairs` defaulted to 150, so 94 % of the co-located pairs were never
    compared and a unit missing from the merge panel read as "checked, nothing
    found" (RESEARCH_LOG 5bd.49, fixed again in 5bd.62);
  * `--uv_per_bit` defaulted to a SpikeGLX constant 10.6x too large for our Trodes
    exports, so the uV axis disagreed with the rest of the pipeline;
  * the sync command printed on the page hardcoded `F:\Gil\Shamir`, which does not
    exist for a session sorted on RIS and archived straight to Z:;
  * the four expensive inputs the cluster had already produced sat on Z and had to be
    copied in by hand, under four different names;
  * the finished page lived only in `outputs/`, so "where is the review tool for
    this session" had no answer that did not involve asking.

Each of those was fixed in the script that caused it, in its default, so this is not
a wrapper that patches over them. What this adds is the one thing a wrapper can add:
the chain in the right order with this session's own values, the checks that would
have caught each of the above, and the finished page saved where it can be found.

WHAT IT DOES
------------
 1. finds the session. Z: first, then F:, across all three rats -- whichever copy has
    the sort is the one used, and the roots that go into the page come from there.
 2. brings the cluster's four outputs into `outputs/`, comparing sha1 against Z and
    saying so when a local copy differs (Z wins; it is the archive).
 3. reads the session's own gain out of its `.rec` header.
 4. runs the chain: review data -> burst/ISI -> merge -> timeline -> merge
    candidates -> the patched page.
 5. checks what the previous builds got wrong: no null `trodes_channel`, every
    co-located pair analysed, the real gain in the metadata, the page under the
    artifact size limit, and the page's own sync command naming a folder that exists.
 6. copies the page and its merge data to Z, into
        <rec>\<sid>.kilosort\review\tool\build_<YYYYMMDD_HHMM>\
    with a MANIFEST naming the sha256 of each file, the clustering they were built
    against, and the exact publish parameters. Dated folders only: nothing on Z is
    overwritten or removed, so every build stays and the newest sorts last.
 7. prints the publish call.

Publishing is the one step that cannot be scripted -- it goes through the Artifact
tooling, which only Claude can reach. Everything up to it is here, so publishing is
a single call with no decisions left in it.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
OUT = os.path.join(ROOT, "outputs")

# Where recordings live, in the order they are searched. Z: is the archive and the
# only home; F: is a working copy that may or may not exist for any given session.
RAT_ROOTS = [
    ("Popper", r"Z:\Gil\Popper"),
    ("Shamir", r"Z:\Gil\Shamir_1"),
    ("Begin", r"Z:\Gil\Begin_1"),
    ("Shamir", r"F:\Gil\Shamir"),
    ("Popper", r"F:\Gil\Popper"),
    ("Begin", r"F:\Gil\Begin_1"),
]

# The cluster's post-KS outputs, which live beside the sort on Z and are the inputs
# the display half of the pipeline cannot compute without the raw .dat.
CLUSTER_INPUTS = [
    "corrected_classification_%s_with_waveforms.json",
    "raw_cluster_points_full_%s.json",
    "raw_separation_full_%s.npy",
    "unit_trodes_channel_lookup_%s.csv",
]

# From 2026-11-17 the cluster also computes the rest of the raw-bound work: the review
# data with everything merged in, the timeline, the merge candidates and the raw spike
# traces (ris_run_postks.sh steps 7-11). When they are on Z this script copies them in
# and skips recomputing them -- which is the difference between a session being ready
# to publish and twenty minutes of reading 300 GB back off Z. Sessions sorted before
# that date have none of them, which is not an error: the steps below just run.
CLUSTER_EXTRAS = [
    "pipeline_review_data_%s.json",
    "burst_isi_metrics_%s.json",
    "session_timeline_%s.json",
    "merge_candidates_%s.json",
    "raw_trace_snippets_%s.json",
]


def say(msg):
    print(msg, flush=True)


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha1_of(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def find_session(sid):
    """(rat, rat_root, rec_dir, ks_dir) for the copy that actually holds the sort."""
    partial = []
    for rat, rat_root in RAT_ROOTS:
        rec = os.path.join(rat_root, sid + ".rec")
        ks = os.path.join(rec, sid + ".kilosort", "kilosort4")
        if not os.path.isdir(rec):
            continue
        if os.path.exists(os.path.join(ks, "spike_clusters.npy")):
            return rat, rat_root, rec, ks
        partial.append(rec)
    if partial:
        raise SystemExit(
            "found %s but no sort in it (no kilosort4\\spike_clusters.npy):\n  %s\n"
            "Sort it first, or name a session that has been sorted."
            % (sid, "\n  ".join(partial)))
    raise SystemExit("no folder named %s.rec under any of:\n  %s"
                     % (sid, "\n  ".join(r for _, r in RAT_ROOTS)))


def bring_in_cluster_inputs(sid, rec):
    """Copy the cluster's four outputs into outputs/, with Z as the authority."""
    src_dir = os.path.join(rec, sid + ".kilosort", "cluster_outputs")
    if not os.path.isdir(src_dir):
        raise SystemExit(
            "no cluster_outputs beside the sort (%s).\nThe post-KS job either never "
            "ran for this session or its results were never brought home. Nothing "
            "here can substitute for them: they are computed from the raw .dat."
            % src_dir)
    missing, brought, already = [], [], []
    for pattern in CLUSTER_INPUTS:
        name = pattern % sid
        src, dst = os.path.join(src_dir, name), os.path.join(OUT, name)
        if not os.path.exists(src):
            missing.append(name)
            continue
        if os.path.exists(dst):
            if sha1_of(src) == sha1_of(dst):
                already.append(name)
                continue
            say("    local copy of %s DIFFERS from Z -- taking Z's, it is the archive"
                % name)
        shutil.copy2(src, dst)
        brought.append(name)
    if missing:
        raise SystemExit(
            "the post-KS job's output is incomplete for %s -- missing:\n  %s\n"
            "Re-run the missing step on the cluster while the raw .dat is still "
            "staged; see scripts/ris_run_postks.sh for the exact commands."
            % (sid, "\n  ".join(missing)))
    say("    %d already current, %d copied from Z" % (len(already), len(brought)))

    extras, absent = [], []
    for pattern in CLUSTER_EXTRAS:
        name = pattern % sid
        src, dst = os.path.join(src_dir, name), os.path.join(OUT, name)
        if not os.path.exists(src):
            absent.append(name)
            continue
        if not os.path.exists(dst) or sha1_of(src) != sha1_of(dst):
            shutil.copy2(src, dst)
        extras.append(name)
    if extras:
        say("    the cluster also built %d of the %d display inputs, taken from Z: %s"
            % (len(extras), len(CLUSTER_EXTRAS),
               ", ".join(n.replace("_%s" % sid, "") for n in extras)))
    if absent:
        say("    not built on the cluster (will be computed here): %s"
            % ", ".join(n.replace("_%s" % sid, "") for n in absent))


def read_gain(rec, sid):
    rec_file = os.path.join(rec, sid + ".rec")
    if not os.path.exists(rec_file):
        siblings = [f for f in sorted(os.listdir(rec)) if f.endswith(".rec")]
        if not siblings:
            raise SystemExit("no .rec file in %s, so no gain to read" % rec)
        rec_file = os.path.join(rec, siblings[0])
        say("    no .rec of its own (joined session); reading %s"
            % os.path.basename(rec_file))
    with open(rec_file, "rb") as fh:
        head = fh.read(4 * 1024 * 1024).decode("latin-1")
    found = sorted(set(re.findall(r'spikeScalingToUv="([^"]+)"', head)))
    if len(found) != 1:
        raise SystemExit("expected one spikeScalingToUv in %s, found %d (%s)"
                         % (rec_file, len(found), ", ".join(found) or "none"))
    return found[0]


RAW_SHARD_SIZE = 25          # must match RAW_SHARD_SIZE in build_review_html.py


def shard_raw_traces(sid):
    r"""Split the per-unit raw spike traces into the files the page fetches.

    These are the "raw spikes" panel, and they are the one part of the tool that is
    NOT baked into the HTML -- 97 kB per unit, 44 MB for a session, far past the
    16 MiB artifact limit. They used to be written into the artifact's `raw_traces`
    database one document per unit, which is ~50 separate write calls per session,
    so in practice a session got published without them and every unit read
    "no raw trace data ... doc does not exist". Shards of 25 units are ONE publish
    beside the page, and the page falls back to the database so the sessions already
    published that way keep working.
    """
    src = os.path.join(OUT, "raw_trace_snippets_%s.json" % sid)
    with open(src, encoding="utf-8") as fh:
        per_unit = json.load(fh)
    dest = os.path.join(OUT, "artifact_files_%s" % sid, "data", "raw_traces")
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    shards, with_traces = {}, 0
    for uid, doc in per_unit.items():
        if doc and doc.get("traces"):
            with_traces += 1
        shards.setdefault(int(uid) // RAW_SHARD_SIZE, {})[str(uid)] = doc
    files, biggest = {}, 0
    for n, payload in sorted(shards.items()):
        name = "shard_%d.json" % n
        path = os.path.join(dest, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        biggest = max(biggest, os.path.getsize(path))
        files["data/raw_traces/" + name] = os.path.relpath(path, ROOT).replace("\\", "/")
    return files, with_traces, len(per_unit), len(shards), biggest


def run(args, label):
    say("\n=== %s" % label)
    say("    " + " ".join(str(a) for a in args))
    r = subprocess.run([str(a) for a in args], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit("FAILED: %s (exit %d)" % (label, r.returncode))


def check(sid, page, gain, with_traces, n_trace_docs, biggest_shard):
    """The checks that would have caught every correction a past build needed."""
    problems = []

    review = json.load(open(os.path.join(OUT, "pipeline_review_data_%s.json" % sid),
                            encoding="utf-8"))

    # The raw spikes panel. Gil hit this on 20260914: the page said "no raw trace
    # data for unit 15 (doc does not exist)" for every unit, because this step was
    # never part of the build at all.
    want = sum(1 for r in review if r.get("n_spikes"))
    if with_traces < want:
        problems.append("raw spike traces exist for %d units but %d units have "
                        "spikes -- the raw spikes panel would read 'no raw trace "
                        "data' for the rest" % (with_traces, want))
    if n_trace_docs != len(review):
        problems.append("raw trace file covers %d units, the review data has %d"
                        % (n_trace_docs, len(review)))
    if biggest_shard > 12 * 1024 * 1024:
        problems.append("a raw-trace shard is %.1f MiB; shards this large make the "
                        "panel slow to open -- lower RAW_SHARD_SIZE (in both this "
                        "script and build_review_html.py)" % (biggest_shard / 1048576.0))
    nulls = [r["unit_id"] for r in review if r.get("trodes_channel") is None]
    if nulls:
        problems.append("%d units have trodes_channel = null (the 2026-09-16 bug: the "
                        "lookup was built but never merged in); e.g. %s"
                        % (len(nulls), nulls[:5]))
    for field in ("cluster_points", "sep_vs_noise", "burst_curve"):
        absent = sum(1 for r in review if field not in r)
        if absent:
            problems.append("%d units are missing %s" % (absent, field))

    cand_path = os.path.join(OUT, "merge_candidates_%s.json" % sid)
    cands = json.load(open(cand_path, encoding="utf-8"))
    meta = cands.get("meta", cands)
    in_range = meta.get("n_pairs_in_range")
    analysed = meta.get("n_pairs_analysed")
    if in_range is not None and analysed is not None and analysed < in_range:
        problems.append("only %s of %s co-located pairs were analysed -- a unit absent "
                        "from the merge panel would read as cleared when it was never "
                        "compared (RESEARCH_LOG 5bd.49)" % (analysed, in_range))
    if "uv_per_bit" in meta and abs(float(meta["uv_per_bit"]) - float(gain)) > 1e-12:
        problems.append("merge candidates were built with uv_per_bit=%s but this "
                        "session's .rec header says %s"
                        % (meta["uv_per_bit"], gain))

    size = os.path.getsize(page)
    limit = 16 * 1024 * 1024
    if size > limit:
        problems.append("the page is %.2f MiB, over the %d MiB artifact limit"
                        % (size / 1048576.0, limit // 1048576))

    # The command the page tells Gil to run. It used to hardcode F:\Gil\Shamir, which
    # does not exist for a session sorted on RIS and archived straight to Z:.
    html = open(page, encoding="utf-8").read()
    m = re.search(r"--rec \"(.*?)\\\\' \+", html)
    if not m:
        problems.append("cannot find the --rec path in the page's sync command")
    else:
        named = m.group(1).replace("\\\\", "\\")
        if not os.path.isdir(named):
            problems.append("the sync command on the page names %s, which is not a "
                            "folder on this machine" % named)
        else:
            say("    page's sync command points at %s" % named)
    return problems, size, len(review), in_range, analysed


def save_to_z(sid, rec, page, gain, extra):
    """Dated build folder on Z. Strictly additive: nothing is overwritten or removed."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    dest = os.path.join(rec, sid + ".kilosort", "review", "tool", "build_" + stamp)
    if os.path.exists(dest):
        raise SystemExit("%s already exists -- refusing to write into an existing "
                         "build folder" % dest)
    os.makedirs(dest)
    files = {}
    for src, name in ((page, os.path.basename(page)),
                      (os.path.join(OUT, "merge_candidates_%s.json" % sid),
                       "merge_candidates.json")):
        shutil.copy2(src, os.path.join(dest, name))
        files[name] = {"bytes": os.path.getsize(src), "sha256": sha256_of(src)}
    # the raw spike traces too, so this folder is everything a republish needs
    shard_src = os.path.join(OUT, "artifact_files_%s" % sid, "data", "raw_traces")
    if os.path.isdir(shard_src):
        shard_dst = os.path.join(dest, "data", "raw_traces")
        shutil.copytree(shard_src, shard_dst)
        files["data/raw_traces/"] = {
            "n_files": len(os.listdir(shard_src)),
            "bytes": sum(os.path.getsize(os.path.join(shard_src, f))
                         for f in os.listdir(shard_src))}
    manifest = {
        "session_id": sid,
        "built": datetime.datetime.now().isoformat(timespec="seconds"),
        "built_by": "scripts/build_review_for_session.py",
        "files": files,
        "uv_per_bit": gain,
        "publish": {
            "file_path": "outputs/%s" % os.path.basename(page),
            "files": {"data/merge_candidates.json":
                      "outputs/merge_candidates_%s.json" % sid},
            "capabilities": {"db": {}, "downloads": True},
            "note": "pass the EXISTING artifact url to keep its verdicts; publishing "
                    "without one makes a new artifact with an empty database",
        },
    }
    manifest.update(extra)
    with open(os.path.join(dest, "MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id")
    ap.add_argument("--force", action="store_true",
                    help="recompute every step even if its output is already there")
    ap.add_argument("--no_z_copy", action="store_true",
                    help="build, check and stop -- do not write the result to Z")
    ap.add_argument("--max_pairs", type=int, default=3000)
    args = ap.parse_args()
    sid = args.session_id
    py = sys.executable

    say("=" * 72)
    say("review tool for %s" % sid)
    say("=" * 72)

    rat, rat_root, rec, ks = find_session(sid)
    say("\n=== where it lives")
    say("    rat        %s" % rat)
    say("    recording  %s" % rec)
    say("    sort       %s" % ks)
    clustering = sha1_of(os.path.join(ks, "spike_clusters.npy"))
    say("    clustering spike_clusters.npy sha1 %s" % clustering)

    say("\n=== the cluster's post-KS outputs")
    bring_in_cluster_inputs(sid, rec)

    gain = read_gain(rec, sid)
    say("\n=== gain")
    say("    spikeScalingToUv = %s uV/bit, from the session's own .rec header" % gain)

    # cluster_info.tsv is the file sync_pipeline refuses to run without, and a
    # session sorted on the cluster has never had Phy opened on it, so nothing
    # creates it. 20260914_122553 got all the way to a finished review and then the
    # sync stopped dead on exactly this, after the verdicts had already been read
    # out of the artifact. It is built here from the classification brought in
    # above, and only when it is absent: if a real Phy curation pass ever writes
    # this file, that pass owns it and this must not overwrite it.
    say("\n=== cluster_info.tsv (sync_pipeline will not run without it)")
    cluster_info = os.path.join(ks, "cluster_info.tsv")
    if os.path.exists(cluster_info):
        say("    already there, left alone: %s" % cluster_info)
    else:
        run([py, os.path.join(SCRIPTS,
                              "build_cluster_info_from_classification_session.py"),
             sid, rec], "cluster_info.tsv from the corrected classification")
        if not os.path.exists(cluster_info):
            raise SystemExit(
                "cluster_info.tsv still is not at %s. The sync would refuse later; "
                "stopping here instead." % cluster_info)

    def have(name):
        p = os.path.join(OUT, name % sid)
        return (not args.force) and os.path.exists(p)

    if have("pipeline_review_data_%s.json") and have("burst_isi_metrics_%s.json"):
        say("\n=== review data: already built (use --force to rebuild)")
    else:
        run([py, os.path.join(SCRIPTS, "build_pipeline_review_data.py"), sid, rec],
            "review data")
        run([py, os.path.join(SCRIPTS, "build_burst_isi_metrics_session.py"), sid, rec],
            "burst and ISI metrics")
        run([py, os.path.join(SCRIPTS, "merge_review_data_session.py"), sid],
            "merge cluster points, separation, burst curves and trodes channels in")

    if have("session_timeline_%s.json"):
        say("\n=== session timeline: already built")
    else:
        run([py, os.path.join(SCRIPTS, "build_session_timeline_data.py"), sid, rec],
            "session timeline from the DIO")

    if have("merge_candidates_%s.json"):
        say("\n=== merge candidates: already built")
    else:
        run([py, os.path.join(SCRIPTS, "build_merge_candidates.py"), sid,
             "--rec_root", rat_root, "--max_pairs", args.max_pairs,
             "--uv_per_bit", gain], "merge candidates")

    if have("raw_trace_snippets_%s.json"):
        say("\n=== raw spike traces: already computed")
    else:
        run([py, os.path.join(SCRIPTS, "compute_raw_trace_snippets_session.py"),
             sid, rec, gain], "raw spike traces")

    run([py, os.path.join(SCRIPTS, "build_review_tool.py"), sid,
         "--rec_root", rat_root, "--rat_root", rat_root], "the page")

    page = os.path.join(OUT, "unit_review_tool_%s.html" % sid)

    say("\n=== raw spike traces -> the files published beside the page")
    pub_files, with_traces, n_trace_docs, n_shards, biggest = shard_raw_traces(sid)
    say("    %d units with traces, in %d shards of %d, largest %.1f MiB"
        % (with_traces, n_shards, RAW_SHARD_SIZE, biggest / 1048576.0))
    pub_files["data/merge_candidates.json"] = \
        "outputs/merge_candidates_%s.json" % sid
    # One publish carries at most 255 files and 64 MB. A 543-unit session's shards
    # plus its merge data come to about 64 MB, so the set is split into parts under
    # a safe 45 MB and published one after another to the same url -- files left out
    # of a publish are kept, so the parts accumulate.
    parts, cur, cur_bytes = [], {}, 0
    for pub_name in sorted(pub_files):
        local = pub_files[pub_name]
        n_bytes = os.path.getsize(os.path.join(ROOT, local))
        if cur and (cur_bytes + n_bytes > 45 * 1024 * 1024 or len(cur) >= 200):
            parts.append(cur)
            cur, cur_bytes = {}, 0
        cur[pub_name] = local
        cur_bytes += n_bytes
    if cur:
        parts.append(cur)
    files_paths = []
    for i, part in enumerate(parts):
        suffix = "" if len(parts) == 1 else "_part%d" % (i + 1)
        p = os.path.join(OUT, "publish_files_%s%s.json" % (sid, suffix))
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(part, fh, indent=2, sort_keys=True)
        files_paths.append(p)
    files_path = files_paths[0]

    problems, size, n_units, in_range, analysed = check(
        sid, page, gain, with_traces, n_trace_docs, biggest)

    say("\n=== checks")
    say("    units             %d" % n_units)
    say("    co-located pairs  %s analysed of %s in range" % (analysed, in_range))
    say("    raw spike traces  %d of %d units" % (with_traces, n_trace_docs))
    say("    page              %d bytes (%.2f MiB of the 16 MiB limit)"
        % (size, size / 1048576.0))
    if problems:
        say("")
        for p in problems:
            say("    FAIL  %s" % p)
        raise SystemExit("\n%d check(s) failed -- not saving this build." % len(problems))
    say("    all checks passed")

    if args.no_z_copy:
        say("\n--no_z_copy: built and checked, nothing written to Z")
    else:
        dest = save_to_z(sid, rec, page, gain,
                         {"clustering_spike_clusters_sha1": clustering,
                          "n_units": n_units,
                          "n_pairs_in_range": in_range,
                          "n_pairs_analysed": analysed,
                          "n_units_with_raw_traces": with_traces,
                          "n_raw_trace_shards": n_shards})
        say("\n=== saved on Z")
        say("    %s" % dest)
        say("    the page, merge_candidates.json and a MANIFEST.json naming both "
            "sha256s")

    say("\n=== publish it")
    say("    file_path    = outputs/unit_review_tool_%s.html" % sid)
    if len(files_paths) == 1:
        say("    files        = the %d entries in %s"
            % (len(pub_files), os.path.relpath(files_path, ROOT)))
    else:
        say("    files        = %d entries, split into %d publishes to the SAME url "
            "(a publish keeps the files it leaves out):" % (len(pub_files),
                                                            len(files_paths)))
        for p in files_paths:
            say("                   %s" % os.path.relpath(p, ROOT))
    say('    capabilities = {"db": {}, "downloads": true}')
    say("    pass the EXISTING artifact url to keep its verdicts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

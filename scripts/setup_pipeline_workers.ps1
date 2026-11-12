# Build N parallel worker roots for the agent pipeline.
#
#   .\scripts\setup_pipeline_workers.ps1 -Session 20260915_084812_103710_concat -N 4
#
# WHY SEPARATE ROOTS: run_pipeline() writes agent_processed\agent_report.json and
# cluster_<label>.tsv at FIXED names, and run_pipeline_batch_session.py archives them
# immediately afterwards. Two batches sharing a session root would race -- one process
# would archive the other's units under its own batch number, silently. Giving each
# worker its own root removes the race entirely; consolidate_batches_session.py just
# globs batch_*_report.json, so the per-worker archives gather back together cleanly.
#
# The raw .probe1.dat is HARDLINKED, not copied: it is 79-180 GB, read-only to the
# pipeline, and all roots are on the same NTFS volume. The kilosort4 folder is copied
# per worker rather than linked, so nothing the pipeline writes can reach across roots.
#
# -Session is REQUIRED on purpose. This script used to hardcode 20260915_103710, which
# is exactly the kind of default that silently runs a whole session's pipeline against
# the wrong raw file.
#
# NOT EVERY SESSION HAS A timestamps.dat. A joint (concatenated) session has one per
# part and none for the whole, since the Trodes clock restarts at the seam -- so that
# file is copied when it exists and skipped when it does not, rather than failing. The
# agent pipeline itself never reads it; only build_session_timeline_data.py does.

param(
    [Parameter(Mandatory = $true)][string]$Session,
    [int]$N = 4,
    [string]$SessionRoot = "F:\Gil\Shamir",
    [string]$Base = "F:\Gil\Shamir\par"
)

$SRC = Join-Path $SessionRoot "$Session.rec\$Session.kilosort"
if (-not (Test-Path $SRC)) { throw "no such session root: $SRC" }

$raw = Join-Path $SRC "$Session.probe1.dat"
if (-not (Test-Path $raw)) { throw "no raw file to link: $raw" }
$rawBytes = (Get-Item $raw).Length
if ($rawBytes % 768 -ne 0) {
    throw "raw file is $rawBytes bytes, not a multiple of 768 (2 bytes x 384 channels) -- it is truncated"
}
"raw: $raw  ($rawBytes bytes, divisible by 768)"

$big = @("$Session.probe1.dat", "$Session.timestamps.dat")
$small = @("$Session.channelmap_probe1.dat", "${Session}_custom_neuropixels_map.json")

for ($i = 1; $i -le $N; $i++) {
    $root = Join-Path $Base "w$i\$Session.rec\$Session.kilosort"
    New-Item -ItemType Directory -Force -Path $root | Out-Null

    foreach ($f in $big) {
        $s = Join-Path $SRC $f
        if (-not (Test-Path $s)) { "worker $i : no $f in the source, skipped"; continue }
        $t = Join-Path $root $f
        if (-not (Test-Path $t)) {
            New-Item -ItemType HardLink -Path $t -Target $s | Out-Null
        }
    }
    foreach ($f in $small) {
        Copy-Item (Join-Path $SRC $f) (Join-Path $root $f) -Force
    }
    # a clean kilosort4 per worker: the sort output only, no agent_processed carried over
    $ks = Join-Path $root "kilosort4"
    New-Item -ItemType Directory -Force -Path $ks | Out-Null
    Get-ChildItem (Join-Path $SRC "kilosort4") -File | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $ks $_.Name) -Force
    }
    "worker $i ready: $root"
}

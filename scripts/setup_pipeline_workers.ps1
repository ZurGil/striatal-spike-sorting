# Build N parallel worker roots for the agent pipeline.
#
# WHY SEPARATE ROOTS: run_pipeline() writes agent_processed\agent_report.json and
# cluster_<label>.tsv at FIXED names, and run_pipeline_batch_session.py archives them
# immediately afterwards. Two batches sharing a session root would race -- one process
# would archive the other's units under its own batch number, silently. Giving each
# worker its own root removes the race entirely; consolidate_batches_session.py just
# globs batch_*_report.json, so the per-worker archives gather back together cleanly.
#
# The raw .probe1.dat is HARDLINKED, not copied: it is 79 GB, read-only to the pipeline,
# and all roots are on the same NTFS volume. The kilosort4 folder is copied per worker
# (2.1 GB each) rather than linked, so nothing the pipeline writes can reach across roots.

param([int]$N = 4)

$SESSION = "20260915_103710"
$SRC  = "F:\Gil\Shamir\${SESSION}.rec\${SESSION}.kilosort"
$BASE = "F:\Gil\Shamir\par"

$big = @("${SESSION}.probe1.dat", "${SESSION}.timestamps.dat")
$small = @("${SESSION}.channelmap_probe1.dat", "${SESSION}_custom_neuropixels_map.json")

for ($i = 1; $i -le $N; $i++) {
    $root = Join-Path $BASE "w$i\${SESSION}.rec\${SESSION}.kilosort"
    New-Item -ItemType Directory -Force -Path $root | Out-Null

    foreach ($f in $big) {
        $t = Join-Path $root $f
        if (-not (Test-Path $t)) {
            New-Item -ItemType HardLink -Path $t -Target (Join-Path $SRC $f) | Out-Null
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

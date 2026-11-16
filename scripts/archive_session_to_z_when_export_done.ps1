# Wait for a Trodes export to finish, then copy that session to Z and verify it.
#
#   powershell -File scripts\archive_session_to_z_when_export_done.ps1 `
#       -Session 20261116_102517 -Source F:\Gil\Popper -Dest Z:\Gil\Popper
#
# WHY IT WAITS TWICE
# ------------------
# `trodesexport` leaving the process list is necessary but not sufficient: the
# last writes can still be in flight, and a .probe1.dat copied mid-write is a
# file that looks complete and is not. So after the process is gone this also
# waits for every file size under the session to stop changing.
#
# NOTHING IS EVER DELETED, on either side. Z is the archive: robocopy runs
# without /MIR and without /PURGE, so a file already on Z is left exactly as it
# is. Exit code 1 from robocopy means "files were copied" -- it is success.
param(
    [Parameter(Mandatory=$true)][string]$Session,
    [Parameter(Mandatory=$true)][string]$Source,
    [Parameter(Mandatory=$true)][string]$Dest,
    [int]$PollSeconds = 60,
    [int]$StableChecks = 3
)

$ErrorActionPreference = "Stop"
$src = Join-Path $Source "$Session.rec"
$dst = Join-Path $Dest   "$Session.rec"
$log = "D:\Gil\spike_sorting_agent\outputs\archive_${Session}_to_z.log"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Output $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

function Fingerprint($path) {
    # One string standing for every file's size under the session.
    $items = Get-ChildItem -Path $path -Recurse -File | Sort-Object FullName
    ($items | ForEach-Object { "$($_.FullName)=$($_.Length)" }) -join "`n"
}

Say "watching $src"
if (-not (Test-Path $src)) { Say "no such folder: $src"; exit 2 }

# 1. wait for the exporter to exit
while ($true) {
    $p = Get-Process -Name trodesexport -ErrorAction SilentlyContinue
    if (-not $p) { Say "trodesexport is no longer running"; break }
    Start-Sleep -Seconds $PollSeconds
}

# 2. wait for the files to stop growing
$stable = 0
$prev = ""
while ($stable -lt $StableChecks) {
    $now = Fingerprint $src
    if ($now -eq $prev) {
        $stable++
        Say "sizes unchanged ($stable/$StableChecks)"
    } else {
        $stable = 0
        $bytes = (Get-ChildItem -Path $src -Recurse -File | Measure-Object -Sum Length).Sum
        Say ("still changing -- {0:N1} GiB so far" -f ($bytes / 1GB))
    }
    $prev = $now
    if ($stable -lt $StableChecks) { Start-Sleep -Seconds $PollSeconds }
}

$srcFiles = Get-ChildItem -Path $src -Recurse -File
$srcBytes = ($srcFiles | Measure-Object -Sum Length).Sum
Say ("export finished: {0} file(s), {1:N1} GiB" -f $srcFiles.Count, ($srcBytes / 1GB))

# What a complete Popper export has. Named rather than assumed, so a missing
# piece stops the copy instead of being archived as if it were whole.
$need = @("$Session.rec", "$Session.kilosort\$Session.probe1.dat",
          "$Session.kilosort\$Session.timestamps.dat",
          "$Session.kilosort\$Session.channelmap_probe1.dat")
$missing = @()
foreach ($rel in $need) {
    if (-not (Test-Path (Join-Path $src $rel))) { $missing += $rel }
}
$dio = Get-ChildItem -Path (Join-Path $src "$Session.DIO") -Filter "*.dat" -ErrorAction SilentlyContinue
if (-not $dio -or $dio.Count -lt 1) { $missing += "$Session.DIO\*.dat" }
if ($missing.Count -gt 0) {
    Say "REFUSING TO COPY -- the export is missing: $($missing -join ', ')"
    exit 3
}
Say "all expected pieces present, including $($dio.Count) DIO file(s)"

$free = (Get-PSDrive -Name ($Dest.Substring(0,1))).Free
Say ("{0} free on {1}: {2:N1} GiB" -f "space", $Dest.Substring(0,1), ($free / 1GB))
if ($free -lt $srcBytes * 1.02) { Say "REFUSING TO COPY -- not enough room"; exit 4 }

Say "copying to $dst (no /MIR, no /PURGE: nothing on Z is removed or overwritten in place)"
$rc = Start-Process -FilePath robocopy.exe -Wait -PassThru -NoNewWindow -ArgumentList @(
    "`"$src`"", "`"$dst`"", "/E", "/COPY:DAT", "/DCOPY:DAT",
    "/R:3", "/W:10", "/XO", "/NFL", "/NDL", "/NP", "/NJH",
    "/LOG+:D:\Gil\spike_sorting_agent\outputs\robocopy_${Session}_to_z.log"
)
Say "robocopy exit code $($rc.ExitCode) (0 = nothing to do, 1 = files copied; both are success)"
if ($rc.ExitCode -ge 8) { Say "robocopy reported a real failure -- stopping"; exit 5 }

# 3. verify: every source file present on Z at the same size
$bad = @()
foreach ($f in $srcFiles) {
    $rel = $f.FullName.Substring($src.Length).TrimStart("\")
    $t = Join-Path $dst $rel
    if (-not (Test-Path $t)) { $bad += "missing on Z: $rel" }
    elseif ((Get-Item $t).Length -ne $f.Length) {
        $bad += "size differs: $rel ($($f.Length) on F, $((Get-Item $t).Length) on Z)"
    }
}
if ($bad.Count -gt 0) {
    Say "VERIFICATION FAILED:"
    $bad | ForEach-Object { Say "   $_" }
    exit 6
}
Say ("verified: {0} file(s), {1:N1} GiB, every size matches on Z" -f $srcFiles.Count, ($srcBytes / 1GB))
Say "F: is NOT touched. Freeing it is Gil's call."
exit 0

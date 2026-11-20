# Verify an archived .rec folder on Z really matches F.
#
# Size alone proves nothing here: robocopy /Z preallocates the destination to the
# full length the instant it starts, so a copy that died at 2% still reports the
# right number of bytes. Instead we hash sampled blocks at both ends and through
# the middle. A truncated or aborted copy leaves zeros where data should be, and
# the tail block catches that immediately.
#
# usage: verify_copy.ps1 <srcFolder> <dstFolder>

param(
    [Parameter(Mandatory = $true)][string]$Src,
    [Parameter(Mandatory = $true)][string]$Dst
)

$ErrorActionPreference = 'Stop'
$BLOCK = 16MB
$FRACTIONS = @(0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)   # 1.0 means the final block

function Block-Hash([string]$path, [long]$offset, [int]$len) {
    $fs = [System.IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
    try {
        $fs.Seek($offset, 'Begin') | Out-Null
        $buf = New-Object byte[] $len
        $read = 0
        while ($read -lt $len) {
            $n = $fs.Read($buf, $read, $len - $read)
            if ($n -le 0) { break }
            $read += $n
        }
        $md5 = [System.Security.Cryptography.MD5]::Create()
        return [BitConverter]::ToString($md5.ComputeHash($buf, 0, $read)).Replace('-', '')
    } finally { $fs.Dispose() }
}

$bad = 0
$checked = 0

Get-ChildItem -Path $Src -Recurse -File | ForEach-Object {
    $rel = $_.FullName.Substring($Src.Length).TrimStart('\')
    $there = Join-Path $Dst $rel

    if (-not (Test-Path $there)) {
        Write-Output "MISSING   $rel"
        $bad++
        return
    }

    $lenF = $_.Length
    $lenZ = (Get-Item $there).Length
    if ($lenF -ne $lenZ) {
        Write-Output ("SIZE      {0}  F={1:N0} Z={2:N0}" -f $rel, $lenF, $lenZ)
        $bad++
        return
    }

    # small files: hash the whole thing, no point sampling
    if ($lenF -le ($BLOCK * 4)) {
        $hF = (Get-FileHash -Path $_.FullName -Algorithm MD5).Hash
        $hZ = (Get-FileHash -Path $there -Algorithm MD5).Hash
        if ($hF -ne $hZ) { Write-Output "CONTENT   $rel (full hash)"; $bad++ }
        else { Write-Output ("ok        {0}  {1:N0} bytes, full hash" -f $rel, $lenF) }
        $checked++
        return
    }

    $offsets = @()
    foreach ($f in $FRACTIONS) {
        $o = if ($f -ge 1.0) { $lenF - $BLOCK } else { [long]([math]::Floor($lenF * $f)) }
        if ($o -gt ($lenF - $BLOCK)) { $o = $lenF - $BLOCK }
        if ($o -lt 0) { $o = 0 }
        $offsets += $o
    }
    $offsets = $offsets | Sort-Object -Unique

    $mismatch = @()
    foreach ($o in $offsets) {
        $hF = Block-Hash $_.FullName $o $BLOCK
        $hZ = Block-Hash $there      $o $BLOCK
        if ($hF -ne $hZ) { $mismatch += $o }
    }

    if ($mismatch.Count -gt 0) {
        Write-Output ("CONTENT   {0}  mismatched at offsets: {1}" -f $rel, ($mismatch -join ', '))
        $bad++
    } else {
        Write-Output ("ok        {0}  {1:N0} bytes, {2} sampled blocks incl. tail" -f $rel, $lenF, $offsets.Count)
    }
    $checked++
}

Write-Output ""
if ($bad -eq 0) {
    Write-Output "VERIFIED  $checked file(s), every sampled block identical"
} else {
    Write-Output "FAILED    $bad file(s) wrong"
}
exit $bad

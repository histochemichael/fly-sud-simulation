$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$sections = Get-Content -Raw (Join-Path $root 'output/narration.json') | ConvertFrom-Json
$voice = New-Object -ComObject SAPI.SpVoice
$voices = $voice.GetVoices()
foreach ($v in $voices) { if ($v.GetDescription() -like '*Zira*') { $voice.Voice = $v } }
$voice.Rate = 1
foreach ($s in $sections) {
    $stream = New-Object -ComObject SAPI.SpFileStream
    $path = Join-Path $root ('output/narration_' + $s.chapter + '.wav')
    $stream.Open($path,3,$false)
    $voice.AudioOutputStream = $stream
    [void] $voice.Speak($s.text)
    $stream.Close()
    Write-Output $path
}

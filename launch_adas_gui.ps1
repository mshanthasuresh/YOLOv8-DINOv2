param(
    [int]$Steps = 180,
    [string]$Device = "0"
)

$ErrorActionPreference = "Stop"
$projectWindows = Split-Path -Parent $MyInvocation.MyCommand.Path
$distribution = "Ubuntu-22.04"
$wslProject = (& wsl.exe -d $distribution -- wslpath -a "$projectWindows").Trim()

if (-not $wslProject) {
    throw "Could not convert the project path to WSL format."
}

$checkCommand = "test -x '$wslProject/.venv-wsl/bin/python'"
& wsl.exe -d $distribution -- bash -lc $checkCommand
if ($LASTEXITCODE -ne 0) {
    throw "WSL environment not found. Follow the WSL setup in README.md first."
}

$runCommand = "cd '$wslProject' && .venv-wsl/bin/python scripts/pybullet_adas_demo.py --device '$Device' --steps '$Steps' --gui"
Write-Host "Starting PyBullet ADAS GUI in $distribution"
Write-Host "Close the PyBullet window or wait for $Steps simulation steps to finish."
& wsl.exe -d $distribution -- bash -lc $runCommand
if ($LASTEXITCODE -ne 0) {
    throw "The PyBullet GUI demo exited with code $LASTEXITCODE."
}

$videoPath = Join-Path $projectWindows "runs\pybullet_adas\adas_replay.mp4"
if (Test-Path $videoPath) {
    Write-Host "Opening the generated ADAS replay: $videoPath"
    Start-Process $videoPath
}
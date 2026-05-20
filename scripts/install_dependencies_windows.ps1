param(
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Installer = Join-Path $Root "install_dependencies.py"
$AppData = Join-Path $env:LOCALAPPDATA "Scenespy"
$PrivatePythonDir = Join-Path $AppData "python311"
$PrivatePython = Join-Path $PrivatePythonDir "python.exe"
$DownloadsDir = Join-Path $AppData "downloads"
$PythonVersion = "3.11.9"
$PythonInstaller = Join-Path $DownloadsDir "python-$PythonVersion-amd64.exe"
$PythonUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"

if (-not (Test-Path $Installer)) {
    throw "install_dependencies.py was not found next to this script."
}

Write-Host "Scenespy dependency installer"
Write-Host ""
Write-Host "This installer prepares FFmpeg, FFprobe, Python, and the AI packages used by Scenespy."
Write-Host "It installs files only under your Windows user profile:"
Write-Host "  $AppData"
Write-Host "It does not change the system PATH and does not replace your system Python."
Write-Host ""

function Ensure-PrivatePython {
    if (Test-Path $PrivatePython) {
        Write-Host "Private Python is already installed for Scenespy."
        return
    }

    New-Item -ItemType Directory -Force -Path $DownloadsDir | Out-Null
    if (-not (Test-Path $PythonInstaller)) {
        Write-Host "Downloading private Python $PythonVersion from python.org..."
        Invoke-WebRequest -Uri $PythonUrl -OutFile $PythonInstaller
    }

    Write-Host "Installing private Python $PythonVersion for Scenespy. This can take a few minutes..."
    $args = @(
        "/quiet",
        "InstallAllUsers=0",
        "TargetDir=$PrivatePythonDir",
        "Include_pip=1",
        "Include_launcher=0",
        "Include_test=0",
        "PrependPath=0",
        "Shortcuts=0"
    )
    $process = Start-Process -FilePath $PythonInstaller -ArgumentList $args -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Python installer failed with exit code $($process.ExitCode)."
    }
    if (-not (Test-Path $PrivatePython)) {
        throw "Private Python was installed, but python.exe was not found in $PrivatePythonDir."
    }
}

if ($DryRun -and -not (Test-Path $PrivatePython)) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCommand) {
        $PythonCommand = Get-Command py -ErrorAction SilentlyContinue
    }
    if (-not $PythonCommand) {
        throw "Python was not found for dry-run validation."
    }
    $Python = $PythonCommand.Source
}
else {
    Ensure-PrivatePython
    $Python = $PrivatePython
}

Write-Host "Starting dependency installation. Large AI packages may take a while to download..."

if ($DryRun) {
    & $Python $Installer --dry-run
}
else {
    & $Python $Installer
}

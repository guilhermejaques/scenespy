param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$TorchMode = "auto"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Installer = Join-Path $Root "_internal\dependencies\install_runtime.py"
$AppData = Join-Path $env:LOCALAPPDATA "Scenespy"
$PythonVersion = "3.11.9"
$PrivatePythonDir = Join-Path $AppData "python311"
$PrivatePython = Join-Path $PrivatePythonDir "python.exe"
$DownloadsDir = Join-Path $AppData "downloads"
$PythonInstaller = Join-Path $DownloadsDir "python-$PythonVersion-amd64.exe"
$PythonUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"

if (-not (Test-Path $Installer)) {
    throw "install_runtime.py was not found in _internal\dependencies."
}

function Ensure-PrivatePython {
    if (Test-Path $PrivatePython) {
        return
    }
    New-Item -ItemType Directory -Force -Path $DownloadsDir | Out-Null
    if (-not (Test-Path $PythonInstaller)) {
        Write-Host "Downloading Python $PythonVersion for Scenespy..."
        Invoke-WebRequest -Uri $PythonUrl -OutFile $PythonInstaller
    }
    Write-Host "Installing private Python $PythonVersion for Scenespy..."
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
        throw "Private Python was installed, but python.exe was not found."
    }
}

Write-Host "Scenespy runtime installer"
Write-Host "This installs FFmpeg/FFprobe and Detect faces AI packages in your user profile."
Write-Host "Install target: $AppData"
Write-Host ""

Ensure-PrivatePython
& $PrivatePython $Installer --torch-mode $TorchMode

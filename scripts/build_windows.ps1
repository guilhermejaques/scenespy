param(
    [switch]$Zip,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$TorchMode = "auto"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$ReleaseAssets = Join-Path $Root "release-assets\windows\ffmpeg"
$DistApp = Join-Path $Root "dist\Scenespy"
$ReleaseDir = Join-Path $Root "release"
$ReleaseApp = Join-Path $ReleaseDir "Scenespy-windows-x64"
$RuntimeBin = Join-Path $ReleaseApp "_internal\bin\windows"
$ZipPath = Join-Path $ReleaseDir "Scenespy-windows-x64.zip"
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

Set-Location $Root

if (-not (Test-Path (Join-Path $ReleaseAssets "ffmpeg.exe"))) {
    throw "Missing ffmpeg.exe in $ReleaseAssets"
}
if (-not (Test-Path (Join-Path $ReleaseAssets "ffprobe.exe"))) {
    throw "Missing ffprobe.exe in $ReleaseAssets"
}
if (-not (Test-Path (Join-Path $Root "models\yolov8n-face.pt"))) {
    throw "Missing models\yolov8n-face.pt"
}

if (Test-Path $ReleaseApp) {
    Remove-Item -LiteralPath $ReleaseApp -Recurse -Force
}
if (Test-Path $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}
if (Test-Path (Join-Path $Root "build")) {
    Remove-Item -LiteralPath (Join-Path $Root "build") -Recurse -Force
}
if (Test-Path (Join-Path $Root "dist")) {
    Remove-Item -LiteralPath (Join-Path $Root "dist") -Recurse -Force
}

& $Python scripts\install_build_dependencies.py --torch-mode $TorchMode
& $Python -m PyInstaller --clean --noconfirm --distpath $ReleaseDir Scenespy.spec

if (-not (Test-Path (Join-Path $ReleaseApp "Scenespy.exe"))) {
    throw "Build finished, but Scenespy.exe was not found in $ReleaseApp"
}
if (-not (Test-Path (Join-Path $ReleaseApp "_internal\models\yolov8n-face.pt"))) {
    throw "Build finished, but yolov8n-face.pt was not found in $ReleaseApp\_internal\models"
}
if (-not (Test-Path (Join-Path $ReleaseApp "_internal\torch"))) {
    throw "Build finished, but torch was not bundled in $ReleaseApp\_internal"
}
if (-not (Test-Path (Join-Path $ReleaseApp "_internal\ultralytics"))) {
    throw "Build finished, but ultralytics was not bundled in $ReleaseApp\_internal"
}

New-Item -ItemType Directory -Force -Path $RuntimeBin | Out-Null
Copy-Item -Path (Join-Path $ReleaseAssets "*") -Destination $RuntimeBin -Force

if ($Zip) {
    Compress-Archive -Path $ReleaseApp -DestinationPath $ZipPath -CompressionLevel Optimal
    Write-Host "Windows ZIP ready: $ZipPath"
}

if (Test-Path (Join-Path $Root "build")) {
    Remove-Item -LiteralPath (Join-Path $Root "build") -Recurse -Force
}

Write-Host "Windows distribution folder ready: $ReleaseApp"

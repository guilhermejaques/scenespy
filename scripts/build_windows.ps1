param(
    [switch]$Zip
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$ReleaseAssets = Join-Path $Root "release-assets\windows\ffmpeg"
$DistApp = Join-Path $Root "dist\Scenespy"
$ReleaseDir = Join-Path $Root "release"
$ReleaseApp = Join-Path $ReleaseDir "Scenespy-windows-x64"
$DependenciesDir = Join-Path $ReleaseApp "_internal\dependencies"
$RuntimeAssets = Join-Path $DependenciesDir "runtime-assets\windows\ffmpeg"
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

& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements-base.txt pyinstaller
& $Python -m PyInstaller --clean --noconfirm --distpath $ReleaseDir Scenespy.spec

if (-not (Test-Path (Join-Path $ReleaseApp "Scenespy.exe"))) {
    throw "Build finished, but Scenespy.exe was not found in $ReleaseApp"
}
if (-not (Test-Path (Join-Path $ReleaseApp "_internal\models\yolov8n-face.pt"))) {
    throw "Build finished, but yolov8n-face.pt was not found in $ReleaseApp\_internal\models"
}

New-Item -ItemType Directory -Force -Path $RuntimeAssets | Out-Null
Copy-Item -Path (Join-Path $ReleaseAssets "*") -Destination $RuntimeAssets -Force
Copy-Item -Path (Join-Path $Root "scripts\install_dependencies.py") -Destination $DependenciesDir -Force
Copy-Item -Path (Join-Path $Root "scripts\install_dependencies_windows.ps1") -Destination $DependenciesDir -Force
Copy-Item -Path (Join-Path $Root "scripts\install_dependencies_windows.bat") -Destination $ReleaseApp -Force

if ($Zip) {
    Compress-Archive -Path $ReleaseApp -DestinationPath $ZipPath -CompressionLevel Optimal
    Write-Host "Windows ZIP ready: $ZipPath"
}

if (Test-Path (Join-Path $Root "build")) {
    Remove-Item -LiteralPath (Join-Path $Root "build") -Recurse -Force
}

Write-Host "Windows distribution folder ready: $ReleaseApp"

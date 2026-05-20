param(
    [switch]$Zip
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$DistApp = Join-Path $Root "dist\Scenespy"
$ReleaseDir = Join-Path $Root "release"
$ReleaseApp = Join-Path $ReleaseDir "Scenespy-windows-x64"
$DependenciesDir = Join-Path $ReleaseApp "_internal\dependencies"
$ZipPath = Join-Path $ReleaseDir "Scenespy-windows-x64.zip"
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

Set-Location $Root

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

& $Python scripts\install_build_dependencies.py
& $Python -m PyInstaller --clean --noconfirm --distpath $ReleaseDir Scenespy.spec

if (-not (Test-Path (Join-Path $ReleaseApp "Scenespy.exe"))) {
    throw "Build finished, but Scenespy.exe was not found in $ReleaseApp"
}
if (-not (Test-Path (Join-Path $ReleaseApp "_internal\models\yolov8n-face.pt"))) {
    throw "Build finished, but yolov8n-face.pt was not found in $ReleaseApp\_internal\models"
}

New-Item -ItemType Directory -Force -Path $DependenciesDir | Out-Null
Copy-Item -Path (Join-Path $Root "scripts\install_runtime.py") -Destination $DependenciesDir -Force
Copy-Item -Path (Join-Path $Root "scripts\install_runtime_windows.ps1") -Destination $ReleaseApp -Force
Copy-Item -Path (Join-Path $Root "scripts\install_runtime_windows.bat") -Destination $ReleaseApp -Force

if ($Zip) {
    Compress-Archive -Path $ReleaseApp -DestinationPath $ZipPath -CompressionLevel Optimal
    Write-Host "Windows ZIP ready: $ZipPath"
}

if (Test-Path (Join-Path $Root "build")) {
    Remove-Item -LiteralPath (Join-Path $Root "build") -Recurse -Force
}

Write-Host "Windows distribution folder ready: $ReleaseApp"

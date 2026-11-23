param(
    [switch]$Zip,
    [switch]$SkipInstaller
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$DistApp = Join-Path $Root "dist\Scenespy"
$ReleaseDir = Join-Path $Root "release"
$ReleaseApp = Join-Path $ReleaseDir "Scenespy-windows-x64"
$DependenciesDir = Join-Path $ReleaseApp "_internal\dependencies"
$ZipPath = Join-Path $ReleaseDir "Scenespy-windows-x64.zip"
$VersionMatch = Select-String -Path (Join-Path $Root "scenespy\__init__.py") -Pattern '^__version__\s*=\s*"([^"]+)"$'
if (-not $VersionMatch) { throw "Could not read Scenespy version." }
$AppVersion = $VersionMatch.Matches[0].Groups[1].Value
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

$PreviousQpaPlatform = $env:QT_QPA_PLATFORM
$PreviousAppSelfTest = $env:SCENESPY_APP_SELFTEST
try {
    $env:QT_QPA_PLATFORM = "offscreen"
    $env:SCENESPY_APP_SELFTEST = "1"
    & (Join-Path $ReleaseApp "Scenespy.exe")
    if ($LASTEXITCODE -ne 0) {
        throw "Packaged Windows application self-test failed with exit code $LASTEXITCODE"
    }
} finally {
    $env:QT_QPA_PLATFORM = $PreviousQpaPlatform
    $env:SCENESPY_APP_SELFTEST = $PreviousAppSelfTest
}

New-Item -ItemType Directory -Force -Path $DependenciesDir | Out-Null
Copy-Item -Path (Join-Path $Root "scripts\install_runtime.py") -Destination $DependenciesDir -Force
Copy-Item -Path (Join-Path $Root "scripts\install_runtime_windows.ps1") -Destination $ReleaseApp -Force
Copy-Item -Path (Join-Path $Root "scripts\install_runtime_windows.bat") -Destination $ReleaseApp -Force

if ($Zip) {
    Compress-Archive -Path $ReleaseApp -DestinationPath $ZipPath -CompressionLevel Optimal
    Write-Host "Windows ZIP ready: $ZipPath"
}

if (-not $SkipInstaller) {
    $InnoCandidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
    )
    $InnoCompiler = $InnoCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
    if (-not $InnoCompiler) {
        throw "Inno Setup 6 was not found. Install it or use -SkipInstaller for a portable build."
    }
    & $InnoCompiler "/DAppVersion=$AppVersion" "/DSourceDir=$ReleaseApp" "/DOutputDir=$ReleaseDir" (Join-Path $Root "installer\windows\Scenespy.iss")
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed with exit code $LASTEXITCODE" }
}

if (Test-Path (Join-Path $Root "build")) {
    Remove-Item -LiteralPath (Join-Path $Root "build") -Recurse -Force
}

Write-Host "Windows distribution folder ready: $ReleaseApp"

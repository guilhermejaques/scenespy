@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

echo Scenespy runtime installer
echo.
echo This installs FFmpeg, FFprobe, and the Detect faces AI packages.
echo Files are installed under your Windows user profile.
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install_runtime_windows.ps1"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Scenespy runtime installation failed. Error code: %EXIT_CODE%
  pause
  exit /b %EXIT_CODE%
)

echo.
echo Scenespy runtime installation finished.
pause

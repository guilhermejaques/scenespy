@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

echo Scenespy dependency installer
echo.
echo This will install the video and AI components required by Scenespy.
echo It may download Python and Python packages from official sources.
echo No system PATH changes are made. Files are installed under your user profile.
echo.
echo Please keep this window open until the installation finishes.
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%_internal\dependencies\install_dependencies_windows.ps1"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Scenespy dependency installation failed. Error code: %EXIT_CODE%
  pause
  exit /b %EXIT_CODE%
)

echo.
echo Scenespy dependency installation finished.
pause

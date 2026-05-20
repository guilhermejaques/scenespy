# Scenespy release builds

Build each platform on that same platform. Do not build Windows, Linux, and
macOS releases from one OS.

## Windows

Prerequisites:

- Python 3.11.
- The project virtual environment, or a Python that can install dependencies.
- FFmpeg files in `release-assets/windows/ffmpeg/`.
- `models/yolov8n-face.pt`.

Build the distributable folder:

```powershell
.\scripts\build_windows.ps1
```

The distributable folder is:

```text
release/Scenespy-windows-x64/
```

To also create a ZIP:

```powershell
.\scripts\build_windows.ps1 -Zip
```

The ZIP will be:

```text
release/Scenespy-windows-x64.zip
```

## Linux

Run this on the target Linux family:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
./scripts/build_linux.sh
```

The distributable folder is `release/Scenespy-linux-x64/`.

For broad compatibility, build on an older Ubuntu LTS rather than a very new
distribution.

## macOS

Run this on macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
./scripts/build_macos.sh
```

The distributable app bundle is `release/Scenespy-macos/Scenespy.app`.

Build Apple Silicon and Intel releases separately unless you intentionally set
up a universal Python environment.

## FFmpeg

Scenespy first checks the bundled paths:

```text
bin/windows/ffmpeg.exe
bin/windows/ffprobe.exe
bin/linux/ffmpeg
bin/linux/ffprobe
bin/macos/ffmpeg
bin/macos/ffprobe
```

If those files are absent, it falls back to the system `PATH`.

The Windows build script copies FFmpeg from `release-assets/windows/ffmpeg/` to
`bin/windows/` before PyInstaller runs. Keep FFmpeg license files in the final
release.

## What goes into the release

The Windows ZIP contains only:

```text
Scenespy-windows-x64/
  Scenespy.exe
  install_dependencies_windows.bat
  _internal/
    dependencies/
      install_dependencies.py
      install_dependencies_windows.ps1
      runtime-assets/windows/ffmpeg/
    models/yolov8n-face.pt
    scenespy/assets/
    Base Python libraries and DLLs collected by PyInstaller
```

The base release shows only the app and the dependency installer `.bat` at the
top level. The PowerShell/Python installer internals and local FFmpeg assets live
under `_internal/dependencies/`.

The release does not include:

```text
.git/
.venv/
build/
dist/
release-assets/
scripts/
docs/
source .py files as editable project files
settings.json
scenespy_crash.log
torch/torchvision/ultralytics/mediapipe in the base app _internal folder
```

User settings and crash logs are created on the user's machine, outside the
installed app folder.

## Dependency installer

Dependencies are installed by a script placed next to the app.

Windows:

Double-click:

```text
install_dependencies_windows.bat
```

Or run from PowerShell:

```powershell
.\install_dependencies_windows.ps1
```

The Windows dependency installer:

- installs FFmpeg/FFprobe from `runtime-assets/windows/ffmpeg/` into the user runtime folder;
- downloads and installs a private Python 3.11.9 for Scenespy if it is not already present;
- installs the AI dependencies into the AI Pack folder;
- does not change the user's system Python or PATH.

Linux/macOS:

```bash
./install_dependencies.sh
```

Dependencies install into user folders, not into the app folder:

```text
Windows FFmpeg runtime: %LOCALAPPDATA%/Scenespy/runtime/
Windows: %LOCALAPPDATA%/Scenespy/ai-pack/
Windows private Python: %LOCALAPPDATA%/Scenespy/python311/
macOS/Linux FFmpeg: installed through system package manager if needed
macOS: ~/Library/Application Support/Scenespy/ai-pack/
Linux: ~/.config/scenespy/ai-pack/
```

For AI, the installer chooses:

```text
NVIDIA detected with nvidia-smi on Windows/Linux: PyTorch CUDA 12.1
Everything else: PyTorch CPU
```

It still needs internet access to download Python on Windows when missing and
to download the AI dependencies. On Linux/macOS, it may need package-manager
access to install Python and FFmpeg if they are not already available.

## User-visible release layout

Windows:

```text
Scenespy-windows-x64/
  Scenespy.exe
  install_dependencies_windows.bat
  _internal/
    dependencies/
      install_dependencies.py
      install_dependencies_windows.ps1
      runtime-assets/windows/ffmpeg/
```

Linux:

```text
Scenespy-linux-x64/
  Scenespy
  install_dependencies.sh
  _internal/
    dependencies/
      install_dependencies.py
```

macOS:

```text
Scenespy-macos/
  Scenespy.app
  install_dependencies.sh
  _internal/
    dependencies/
      install_dependencies.py
```

AMD and Intel GPUs can still be used by FFmpeg for video encoding when the
encoder is available. They are not used for PyTorch face AI in this release.

## Smoke test

Test the generated app outside the repository and without relying on the source
virtual environment:

1. Open the app.
2. Before installing dependencies, confirm the app shows the missing FFmpeg message.
3. Run `install_dependencies_windows.bat`.
4. Restart Scenespy and confirm no missing FFmpeg warning appears.
5. Process a small video with `Every seconds`.
6. Process a small video with `Scene detection`.
7. Run `Detect faces` once to confirm the model loads.
6. Check `%APPDATA%/Scenespy/scenespy_crash.log` on Windows, or the equivalent
   user config folder on Linux/macOS.

## Size note

The base Windows release should be much smaller than the old all-in-one build
because it excludes `torch`, `torchvision`, `ultralytics`, and `mediapipe`.
The AI Pack remains large because it contains the face detection stack.

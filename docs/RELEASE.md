# Scenespy release builds

Build each platform on that same platform. Do not build Windows, Linux, and
macOS releases from one OS. Pushing a tag such as `v0.2.1` runs
`.github/workflows/release-installers.yml` and attaches all installers to the
GitHub Release. The tag and `scenespy.__version__` should match. SHA-256 files
are published beside every artifact and are required by the in-app updater.

Supported release targets:

- Windows 10 version 1809 or later, x64.
- Ubuntu 22.04 or later and Debian 12 or later, x64.
- macOS 12 Monterey or later, Intel x64 and Apple Silicon arm64.

The installers reject older operating systems where the pinned Qt and native
libraries are not covered by the release builds.

At startup, packaged apps check the latest GitHub Release without blocking the
interface. A newer platform-specific installer is offered in-app, downloaded
over HTTPS, verified against its release SHA-256 file, and opened only after
successful verification. Offline checks fail silently.

The application bundle is intentionally lightweight. Each platform installer
automatically downloads and installs the runtime during setup:

- Torch, TorchVision, Ultralytics, and MediaPipe are not bundled.
- FFmpeg and FFprobe are not bundled.
- The end user does not need to run a second script.
- Python is installed privately for Scenespy, so a system Python installation
  is not required.
- AI face detection is offered as a large optional download and selected by
  default. When selected, the installer automatically chooses CPU, NVIDIA/CUDA,
  or Apple MPS for the current hardware. If deselected, the core app and FFmpeg
  remain fully functional, but AI face detection is not installed.
- Installers display approximately 1 GB for the core setup, up to 3 GB for the
  CPU AI runtime, and up to 6 GB for the NVIDIA/CUDA runtime.

## Requirements files

```text
requirements-base.txt  Bundled into the lightweight app.
requirements-ai.txt    Used for source/dev installs and mirrored by runtime installer logic.
requirements.txt       Source/dev convenience install: base + AI.
```

Build scripts install only `requirements-base.txt` plus PyInstaller.

## Windows

Prerequisites:

- Python 3.11.
- The project virtual environment, or a Python that can install dependencies.
- `models/yolov8n-face.pt`.

Build the distributable folder and the Windows installer (requires Inno Setup 6):

```powershell
.\scripts\build_windows.ps1
```

To also create a ZIP:

```powershell
.\scripts\build_windows.ps1 -Zip
```

The release is:

```text
release/Scenespy-windows-x64/
release/Scenespy-windows-x64.zip
release/Scenespy-<version>-windows-x64-setup.exe
```

End users should run the `*-setup.exe`. The setup wizard automatically installs
FFmpeg and the AI runtime before launching Scenespy. The ZIP remains available
as a portable/developer alternative and does not perform automatic setup.

Portable installation:

```text
Scenespy-windows-x64/
  Scenespy.exe
  install_runtime_windows.bat
  install_runtime_windows.ps1
  _internal/dependencies/install_runtime.py
```

The user runs `install_runtime_windows.bat` once. It installs:

- an existing compatible Python 3.11, or a private Python 3.11 under
  `%LOCALAPPDATA%/Scenespy/python311/` when needed;
- FFmpeg/FFprobe from the Gyan.dev essentials build into `%LOCALAPPDATA%/Scenespy/runtime/`;
- Torch CPU or CUDA plus AI packages into `%LOCALAPPDATA%/Scenespy/ai-pack/`.

## Linux

Run this on the target Linux family:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
bash ./scripts/build_linux.sh
```

The build Python must have Tk support. On Debian/Ubuntu this usually means
installing `python3.11-tk` or `python3-tk` before building.

The release is:

```text
release/Scenespy-linux-x64/
release/Scenespy-linux-x64.tar.gz
release/Scenespy-<version>-linux-amd64.deb
release/Scenespy-<version>-linux-x64.run
```

For Ubuntu/Debian, users should open the `.deb` in the system Software app. It
installs Scenespy under `/opt/scenespy`, adds the application-menu entry, and
declares FFmpeg as a package dependency so the system package manager handles it.

The `.run` remains a manual per-user alternative and may ask for administrator
permission to install FFmpeg through the system package manager:

```bash
chmod +x Scenespy-<version>-linux-x64.run
./Scenespy-<version>-linux-x64.run
```

This installs the app in `~/.local/opt/scenespy`, creates a launcher under
`~/.local/share/applications`, links `scenespy` in `~/.local/bin`, and then
automatically installs FFmpeg and offers the optional AI runtime selected by
default. It can also be installed later from Detect faces inside Scenespy.
The tarball remains available as a portable alternative.

Portable installation:

```bash
tar -xzf Scenespy-linux-x64.tar.gz
cd Scenespy-linux-x64
./install_runtime.sh
./Scenespy
```

The runtime installer installs:

- an existing compatible Python 3.11, or a private Python 3.11 from
  `python-build-standalone` under `~/.config/scenespy/python311/` when needed;
- FFmpeg/FFprobe through `apt`, `dnf`, or `pacman`;
- Torch CPU or CUDA plus AI packages into `~/.config/scenespy/ai-pack/`.

## macOS

The packaged app targets macOS 12 Monterey or later. Keep the macOS-specific
PySide6 pin in `requirements-macos-build.txt`; newer Qt releases may raise the
minimum deployment target even when the application code is unchanged.

Run this on macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
bash ./scripts/build_macos.sh
```

The release is:

```text
release/Scenespy-macos/Scenespy.app
release/Scenespy-<version>-macos-x64.pkg
release/Scenespy-<version>-macos-arm64.pkg
```

User installation: open the PKG and follow the installer. It installs
`Scenespy.app` in Applications and automatically downloads FFmpeg for the
signed-in user. The optional AI runtime is selected by default and can also be
installed later from inside Scenespy.

Unsigned local builds may require Control-click > Open. Public distribution
should use an Apple Developer ID certificate and notarization.

Portable installation:

```bash
cd Scenespy-macos
./install_runtime.sh
open Scenespy.app
```

The runtime installer installs:

- an existing compatible Python 3.11, or a private Python 3.11 from
  `python-build-standalone` under
  `~/Library/Application Support/Scenespy/python311/` when needed;
- FFmpeg/FFprobe through Homebrew when they are not already in PATH;
- Torch CPU plus AI packages into `~/Library/Application Support/Scenespy/ai-pack/`.

On macOS 11 and 12, MediaPipe is skipped because current MediaPipe wheels require
macOS 13 or later. Detect faces still runs, but without MediaPipe landmark
validation.

## Torch selection

The runtime installer accepts:

```text
auto  Detect NVIDIA with nvidia-smi and try CUDA, otherwise CPU.
cpu   Force CPU PyTorch.
cuda  Force CUDA PyTorch and fail if CUDA is not usable.
```

Examples:

```powershell
.\install_runtime_windows.ps1 -TorchMode cpu
.\install_runtime_windows.ps1 -TorchMode cuda
```

```bash
TORCH_MODE=cpu ./install_runtime.sh
TORCH_MODE=cuda ./install_runtime.sh
```

In `auto` mode, if CUDA install or verification fails, the installer falls back
to CPU PyTorch.

## Smoke test

Test the generated app outside the repository:

1. Run the platform installer with an active internet connection and confirm
   runtime setup completes automatically.
2. Open the app without running any additional scripts.
3. Confirm no missing FFmpeg warning appears.
4. Process a small video with `Every seconds`.
5. Process a small video with `Scene detection`.
6. Run `Detect faces` once to confirm Torch, Ultralytics, MediaPipe, and the model load.

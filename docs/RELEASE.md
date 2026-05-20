# Scenespy release builds

Build each platform on that same platform. Do not build Windows, Linux, and
macOS releases from one OS.

The release app is intentionally lightweight:

- Torch, TorchVision, Ultralytics, and MediaPipe are not bundled.
- FFmpeg and FFprobe are not bundled.
- A runtime installer is shipped next to the app and installs those pieces on
  the user's machine.

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

Build the distributable folder:

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
```

User installation:

```text
Scenespy-windows-x64/
  Scenespy.exe
  install_runtime_windows.bat
  install_runtime_windows.ps1
  _internal/dependencies/install_runtime.py
```

The user runs `install_runtime_windows.bat` once. It installs:

- a private Python 3.11 under `%LOCALAPPDATA%/Scenespy/python311/`;
- FFmpeg/FFprobe from the Gyan.dev essentials build into `%LOCALAPPDATA%/Scenespy/runtime/`;
- Torch CPU or CUDA plus AI packages into `%LOCALAPPDATA%/Scenespy/ai-pack/`.

## Linux

Run this on the target Linux family:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
./scripts/build_linux.sh
```

The build Python must have Tk support. On Debian/Ubuntu this usually means
installing `python3.11-tk` or `python3-tk` before building.

The release is:

```text
release/Scenespy-linux-x64/
release/Scenespy-linux-x64.tar.gz
```

User installation:

```bash
tar -xzf Scenespy-linux-x64.tar.gz
cd Scenespy-linux-x64
./install_runtime.sh
./Scenespy
```

The runtime installer installs:

- a private Python 3.11 from `python-build-standalone` under `~/.config/scenespy/python311/`;
- FFmpeg/FFprobe through `apt`, `dnf`, or `pacman`;
- Torch CPU or CUDA plus AI packages into `~/.config/scenespy/ai-pack/`.

## macOS

Run this on macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
./scripts/build_macos.sh
```

The release is:

```text
release/Scenespy-macos/Scenespy.app
```

User installation:

```bash
cd Scenespy-macos
./install_runtime.sh
open Scenespy.app
```

The runtime installer installs:

- a private Python 3.11 from `python-build-standalone` under
  `~/Library/Application Support/Scenespy/python311/`;
- FFmpeg/FFprobe through Homebrew when they are not already in PATH;
- Torch CPU plus AI packages into `~/Library/Application Support/Scenespy/ai-pack/`.

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

1. Run the runtime installer.
2. Open the app.
3. Confirm no missing FFmpeg warning appears.
4. Process a small video with `Every seconds`.
5. Process a small video with `Scene detection`.
6. Run `Detect faces` once to confirm Torch, Ultralytics, MediaPipe, and the model load.

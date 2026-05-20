# Scenespy release builds

Build each platform on that same platform. Do not build Windows, Linux, and
macOS releases from one OS.

The release target is now a complete app bundle/folder. Users should not need
to run a second dependency installer to use `Scene detection`, `Every seconds`,
or `Detect faces`.

## PyTorch mode

Build scripts install the full runtime dependencies before PyInstaller runs.
PyTorch can be selected with:

```text
auto  Detect NVIDIA on the build machine and use CUDA there, otherwise CPU.
cpu   Force CPU PyTorch.
cuda  Force CUDA PyTorch and fail if CUDA is not usable during verification.
```

Windows:

```powershell
.\scripts\build_windows.ps1 -TorchMode auto
.\scripts\build_windows.ps1 -TorchMode cpu
.\scripts\build_windows.ps1 -TorchMode cuda
```

Linux/macOS:

```bash
TORCH_MODE=auto ./scripts/build_linux.sh
TORCH_MODE=cpu ./scripts/build_macos.sh
```

macOS defaults to CPU PyTorch. Apple VideoToolbox can still be used for video
encoding when FFmpeg supports it.

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

To also create a ZIP:

```powershell
.\scripts\build_windows.ps1 -Zip
```

The release is:

```text
release/Scenespy-windows-x64/
release/Scenespy-windows-x64.zip
```

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

For broad compatibility, build on an older Ubuntu LTS rather than a very new
distribution.

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

Build Apple Silicon and Intel releases separately unless you intentionally set
up a universal Python environment.

## FFmpeg

Scenespy first checks bundled paths under the frozen app:

```text
_internal/bin/windows/ffmpeg.exe
_internal/bin/windows/ffprobe.exe
_internal/bin/linux/ffmpeg
_internal/bin/linux/ffprobe
bin/macos/ffmpeg
bin/macos/ffprobe
```

If bundled binaries are absent, it falls back to the system `PATH`.

Windows builds require `release-assets/windows/ffmpeg/` and copy those files
into the generated release. Linux and macOS builds copy matching
`release-assets/<platform>/ffmpeg/` files when present.

## What goes into the release

The Windows ZIP contains:

```text
Scenespy-windows-x64/
  Scenespy.exe
  _internal/
    bin/windows/ffmpeg.exe
    bin/windows/ffprobe.exe
    models/yolov8n-face.pt
    scenespy/assets/
    torch/
    torchvision/
    ultralytics/
    mediapipe/
    Base Python libraries and DLLs collected by PyInstaller
```

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
```

User settings and crash logs are created on the user's machine, outside the
installed app folder.

## Smoke test

Test the generated app outside the repository and without relying on the source
virtual environment:

1. Open the app.
2. Confirm no missing FFmpeg warning appears.
3. Process a small video with `Every seconds`.
4. Process a small video with `Scene detection`.
5. Run `Detect faces` once to confirm Torch, Ultralytics, MediaPipe, and the model load.
6. Check `%APPDATA%/Scenespy/scenespy_crash.log` on Windows, or the equivalent
   user config folder on Linux/macOS.

## Size note

The release is intentionally larger because it includes the face detection stack.
CPU builds are more portable. CUDA builds are larger and should be built/tested
on a machine with a compatible NVIDIA driver.

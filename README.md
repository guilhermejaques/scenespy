# Scenespy

Scenespy is a desktop app designed to detect different scenes in a video and automatically separate them. It can detect scene changes, cut videos by fixed time intervals, or extract faces found in the video. It helps streamline the content creation workflow for people who work with video. Its core editing and cutting modes are lightweight, while optional AI packages are used for face detection.

The goal is simple: choose a video, choose an output folder, select the cutting mode, and let the app process it.

---
Face detection and cropping from video:

<img width="65%" alt="Image" src="https://github.com/user-attachments/assets/222c8977-bdea-41b0-8aa6-9ab088195f5f" />
<img width="65%" alt="Image" src="https://github.com/user-attachments/assets/8bdfd2b3-a0b2-4244-97c0-bc078a1ff509" />

---
Detection and cutting of video scenes:

<img width="65%" alt="Image" src="https://github.com/user-attachments/assets/649c86c9-1149-4148-a3ef-d1ecda397edd" />
<img width="65%" alt="Image" src="https://github.com/user-attachments/assets/ae2498f9-720c-4b2e-b678-757703e9219d" />

---

## What the app does

- Detects video scenes and places editable cut markers on a timeline.
- Splits videos by time interval using markers that can be adjusted before export.
- Detects faces and places reviewable face events at the moments where they appear.
- Plays video and audio inside the editor with play, pause, seek, and skip controls.
- Saves non-destructive `.scenespy` projects so analysis can be reviewed later.
- Processes one or multiple videos in queue.
- Creates an organized output folder for each process.
- Generates metadata files such as `scenes.json` and, when necessary, `cut_errors.json`.

## How to use
If you want to use scenespy directly in Google Colab or other cloud services: [Scenespy ipynb](https://github.com/guilhermejaques/scenespy/blob/main/Scenespy_Colab.ipynb)

1. Open Scenespy.
2. In **Source video**, select one or more videos.
3. In **Output folder**, choose where the files will be saved.
4. Select the mode:
   - **Scene detection**
   - **Every seconds**
   - **Detect faces**
5. Adjust sensitivity if necessary.
6. Choose the available hardware acceleration.
7. Click **Analyze video**.
8. Review the generated markers on the timeline. Move or remove cuts and include or exclude face events as needed.
9. Click **Export reviewed result** when the timeline is ready.

The source video is never modified. Final files are created only when the reviewed result is exported.

---

## Available modes

### Scene detection

Analyzes the video and tries to find natural scene changes to cut the video. Useful for movies, series, trailers, gameplay videos, with no size limit.  
This is the mode that requires the most from the machine because of the statistical pipeline that tries to identify the difference between one scene and another.

### Every seconds

Cuts the video into fixed-duration parts. It is the most predictable mode: you choose the interval in seconds and the app splits the video.

### Detect faces

Searches for faces in the video and saves images of the detected faces. It is a mode that can scan every frame of the video to find faces, even those difficult to visualize.  
It does not classify by person and saves the same person's face at different moments.

## Sensitivity

- **Low**: detects fewer cuts. Better for calm videos or when you want to avoid false and out-of-context cuts. Can work well for movies, documentaries, and videos where sequences are longer.
- **Normal**: balance between precision and number of cuts. Test it on your video and check the results.
- **High**: detects more cuts. Better for fast-paced videos, trailers, clips, and content with a lot of action.
- **Auto**: tries to automatically choose parameters based on the video. Not used in face mode.

IMPORTANT: Every video is unique, so if one sensitivity mode worked for a specific video, it does not mean it will work for another. Testing is always the best solution.

## Acceleration

- **CPU**: most compatible option (default). Works in all modes, but may be slower.
- **NVIDIA**: can accelerate encoding through FFmpeg/NVENC and can also accelerate face mode through CUDA if PyTorch with CUDA support is installed.
- **AMD**: can accelerate video encoding through FFmpeg/AMF on compatible systems. Does not accelerate face mode.
- **Intel**: can accelerate video encoding through FFmpeg/QSV on compatible systems. Does not accelerate face mode.
- **Apple**: can accelerate video encoding through FFmpeg/VideoToolbox on macOS. On compatible Apple Silicon systems, face detection can also use PyTorch MPS.

Today, the most relevant way to accelerate processing is with NVIDIA CUDA, but the app will work fine even if you do not use CUDA.

## Supported formats

The app accepts videos such as:

- `.mp4` The most compatible format.
- `.mkv`
- `.mov`
- `.avi`
- `.webm`
- `.m4v`

Invalid, temporary, or corrupted files may be ignored or automatically repaired when possible.  
MKV supports multiple audio tracks and may present container issues, so for difficult-to-process videos, the app will convert MKV to MP4 in an attempt to solve the problem.

---

# Installation

## Quick installation | GitHub Releases

Use the ready-to-use version of Scenespy from the GitHub **Releases** tab. Do not use the **Code > Download ZIP** button if you only want to install and use the app. Release `v0.2.1` provides an installer for each supported operating system and architecture:

- Windows x64: [`Scenespy-0.2.1-windows-x64-setup.exe`](https://github.com/guilhermejaques/scenespy/releases/download/v0.2.1/Scenespy-0.2.1-windows-x64-setup.exe)
- Linux Ubuntu/Debian x64: [`Scenespy-0.2.1-linux-amd64.deb`](https://github.com/guilhermejaques/scenespy/releases/download/v0.2.1/Scenespy-0.2.1-linux-amd64.deb)
- macOS Intel: [`Scenespy-0.2.1-macos-x64.pkg`](https://github.com/guilhermejaques/scenespy/releases/download/v0.2.1/Scenespy-0.2.1-macos-x64.pkg)
- macOS Apple Silicon: [`Scenespy-0.2.1-macos-arm64.pkg`](https://github.com/guilhermejaques/scenespy/releases/download/v0.2.1/Scenespy-0.2.1-macos-arm64.pkg)

Supported systems are Windows 10 version 1809 or later (x64), Ubuntu 22.04 or
later or Debian 12 or later (x64), and macOS 12 Monterey or later (Intel or
Apple Silicon).

All downloads and their SHA-256 checksum files are available on the [Scenespy v0.2.1 release page](https://github.com/guilhermejaques/scenespy/releases/tag/v0.2.1).

Download and open the installer for your system. The base setup installs
Scenespy and FFmpeg. AI face detection is a separate, optional download that
can be selected during setup or installed later from inside Scenespy.

The installer offers AI face detection as a large optional download and selects
it by default. When selected, it automatically chooses CPU, NVIDIA/CUDA, or
Apple MPS for the current hardware. If face detection is
opened later, Scenespy explains which packages are missing and can install them
inside its private environment without requiring a full app reinstall.

Allow approximately 1 GB for the core installation, up to 3 GB with CPU AI
features, or up to 6 GB when the NVIDIA/CUDA runtime is selected automatically.

- Windows: run the `*-setup.exe` installer.
- Linux Ubuntu/Debian: open the `.deb` with the system Software app. A `.run`
  and portable `.tar.gz` are also published for advanced/manual installation.
- macOS: open the `.pkg` for Intel or Apple Silicon.

After setup completes, launch Scenespy from the operating system's application
menu.

Packaged versions check GitHub Releases for updates when the app starts. If a
new version is available, Scenespy can download the correct installer for the
current system, verify its SHA-256 checksum, and open it. The check is silent
when the computer is offline.

---

## Run from source code

You are responsible for installing Python, Python dependencies, FFmpeg/FFprobe, and libraries.

Requirements for source code:

- [Python 3.11 or 3.12](https://www.python.org/downloads/)
- [FFmpeg](https://ffmpeg.org/download.html) and [FFprobe](https://ffmpeg.org/download.html).
- Dependencies from `requirements.txt`.
- On Windows, Microsoft Visual C++ Redistributable x64 may be required for PyTorch.

On distributions that ship a newer Python, use `pyenv` or another equivalent method to select Python 3.11 or 3.12.

Main Python dependencies installed by `requirements.txt`:

- `PySide6`: desktop editor interface and multimedia playback.
- `pillow`: images and previews.
- `numpy`: numerical processing.
- `opencv-contrib-python`: frame reading, visual analysis, and MediaPipe dependency.
- `av`: PyAV backend used by PySceneDetect.
- `scenedetect`: base scene change detection.
- `torch` and `torchvision`: required for **Detect faces** mode.
- `ultralytics`: loads the YOLO face model.
- `mediapipe`: facial validation and landmarks.

```bash
python -m pip install -r requirements.txt
```


---

### CPU-only installation 

On Windows and Linux, use the official CPU-only PyTorch index:

```bash
python -m pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

On macOS, use the regular PyPI wheels. They provide CPU support and MPS on
compatible Apple Silicon systems:

```bash
python -m pip install --force-reinstall torch==2.5.1 torchvision==0.20.1
```

With CPU:

- **Scene detection** works.
- **Every seconds** works.
- **Detect faces** works, but may be slower.
- NVIDIA/AMD/Intel/Apple acceleration may not appear or may not be used.

---

### Installation with NVIDIA CUDA

CUDA only affects **Detect faces** mode when PyTorch was installed with support for your CUDA version. It can also help with video cutting.

The `requirements.txt` pins `torch==2.5.1` and `torchvision==0.20.1` without choosing a specific CUDA build. For CUDA, install PyTorch from the official index for the desired version.

Example for CUDA 12.1:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

Then confirm whether CUDA was detected:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected result for active CUDA:

```text
True
```

If it returns `False`, the app will still work on CPU, but **Detect faces** mode will not use the GPU.

---

#### NVIDIA without CUDA

Even without PyTorch CUDA, the **NVIDIA** option can still accelerate video cutting if FFmpeg has the NVENC encoder (`h264_nvenc`) available.

This means:

- **Scene detection** and **Every seconds** can use NVIDIA for encoding.
- **Detect faces** falls back to CPU if `torch.cuda.is_available()` is `False`.

The installation process is standard (with Python 3.11, FFmpeg, and FFprobe installed).
```bash
python -m pip install -r requirements.txt
```

---

## More information

### AMD

In Scenespy, AMD is used for video encoding through FFmpeg/AMF (`h264_amf`) when available.

The installation process is standard (with Python 3.11, FFmpeg, and FFprobe installed).
```bash
python -m pip install -r requirements.txt
```

Practical requirements:

- Compatible AMD GPU.
- AMD driver installed.
- FFmpeg compiled with AMF support.

AMD does not accelerate **Detect faces** mode in this app. Face mode uses CPU or NVIDIA CUDA.

To test whether FFmpeg recognizes AMF:

```bash
ffmpeg -hide_banner -encoders | grep h264_amf
```

On Windows PowerShell:

```powershell
ffmpeg -hide_banner -encoders | Select-String h264_amf
```

---

### Intel

In Scenespy, Intel is used for video encoding through FFmpeg/QSV (`h264_qsv`) when available.

The installation process is standard (with Python 3.11, FFmpeg, and FFprobe installed).
```bash
python -m pip install -r requirements.txt
```

Practical requirements:

- Intel CPU/GPU with Quick Sync Video.
- Updated Intel driver.
- FFmpeg compiled with QSV support.

Intel does not accelerate **Detect faces** mode in this app.

Test:

```bash
ffmpeg -hide_banner -encoders | grep h264_qsv
```

On Windows PowerShell:

```powershell
ffmpeg -hide_banner -encoders | Select-String h264_qsv
```

---

### Apple Silicon and macOS

On macOS, the app can use VideoToolbox encoding (`h264_videotoolbox`) when the installed FFmpeg provides this encoder.

The installation process is standard (with Python 3.11, FFmpeg, and FFprobe installed).
```bash
python -m pip install -r requirements.txt
```

Encoder test:

```bash
ffmpeg -hide_banner -encoders | grep h264_videotoolbox
```

On compatible Apple Silicon systems, **Detect faces** can use the PyTorch MPS backend. If MPS is unavailable, face detection continues on CPU. The **Apple** option also enables VideoToolbox for compatible video-encoding operations.

# FFmpeg and FFprobe

Scenespy requires FFmpeg and FFprobe to read, validate, and cut videos.

In the ready-to-use installers from the **Releases** tab, setup automatically downloads or installs FFmpeg/FFprobe. A system-wide FFmpeg installation is not required before installing Scenespy.

On Windows, setup installs the FFmpeg essentials build binaries from Gyan.dev into Scenespy's private runtime under `%LOCALAPPDATA%/Scenespy/`.

See the [FFmpeg](https://ffmpeg.org/) website for more information.


When running from source on Linux or macOS, install FFmpeg through the system package manager if it is not already available:

```bash
sudo apt install ffmpeg
```

```bash
brew install ffmpeg
```

When starting, Scenespy searches its bundled/private locations and then the system `PATH`.

# Common problems

### `ffmpeg` or `ffprobe` not found

Install FFmpeg and make sure the executables are available in the `PATH`, or place the binaries in `bin/<system>/`.

### Detect faces mode does not open

The packaged app displays an **Install dependencies** option when optional AI packages are missing. For a source installation, check whether `torch`, `ultralytics`, `mediapipe`, and the `models/yolov8n-face.pt` model exist.

```bash
python -c "import torch, ultralytics, mediapipe; print('ok')"
```

### CUDA does not appear

Check whether you installed a CUDA build of PyTorch and whether the NVIDIA driver is updated.

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### AMD, Intel, or Apple do not appear

These options depend on FFmpeg, drivers, and the operating system. Check whether the corresponding encoder exists:

```bash
ffmpeg -hide_banner -encoders
```

### MediaPipe installation fails

Use Python 3.11 or 3.12. Other Python versions may not have compatible wheels for all dependencies.

### Both `opencv-python` and `opencv-contrib-python` are installed

The app declares `opencv-contrib-python` because MediaPipe depends on it. Ultralytics may also install `opencv-python`. If `python -m pip check` does not report conflicts and the app opens normally, no mandatory action is required.

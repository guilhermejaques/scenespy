# Scenespy

Scenespy is a desktop app designed to detect different scenes in a video and automatically separate them. It can detect scene changes, cut videos by fixed time intervals, or extract faces found in the video. It helps streamline the content creation workflow for people who work with video, requires very little from your computer, and does not run heavy AI models.

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

- Detects and cuts video scenes.
- Splits videos into segments by time interval in seconds.
- Detects faces and saves images of the best crops found.
- Processes one or multiple videos in queue.
- Creates an organized output folder for each process.
- Generates metadata files such as `scenes.json` and, when necessary, `cut_errors.json`.

## How to use

1. Open Scenespy.
2. In **Source video**, select one or more videos.
3. In **Output folder**, choose where the files will be saved.
4. Select the mode:
   - **Scene detection**
   - **Every seconds**
   - **Detect faces**
5. Adjust sensitivity if necessary.
6. Choose the available hardware acceleration.
7. Click **Start**.
8. Wait for the processing to finish.

The results will be created inside the selected folder, in a subfolder containing the date, mode, sensitivity, and acceleration used.

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
- **Apple**: can accelerate video encoding through FFmpeg/VideoToolbox on macOS. Does not accelerate face mode.

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

Use the ready-to-use version of Scenespy from the GitHub **Releases** tab. Do not use the **Code > Download ZIP** button if you only want to install and use the app. A release package already exists for each supported operating system:

- Windows: [Scenespy-Windows-x64](https://github.com/guilhermejaques/scenespy/releases/tag/0.1.0)
- Linux: [Scenespy-Linux-x64](https://github.com/guilhermejaques/scenespy/releases/tag/0.1.0)
- macOS: [Scenespy-MacOS-x64](https://github.com/guilhermejaques/scenespy/releases/tag/0.1.0)

Download the package for your system, extract the folder, and run the `install_runtime` installer included with the app. This installer configures and installs the external dependencies used by the app, such as FFmpeg/FFprobe, private Python, and AI packages that are required. Open the command line on your system and locate the app directory to run the installer, then launch the `Scenespy` app.

Windows (`.bat` installer may run as administrator, use right-click for that)

```bat
install_runtime_windows.bat 
Scenespy.exe
```

Linux and MacOS:

```bash
chmod +x install_runtime.sh  # Permission command
./install_runtime.sh 
./Scenespy # App 
```

---

For beginner users who do not know how to use the command line:  
you only need to locate the folder where the app is and run the installer before executing the app. Example:

`cd Downloads` > `cd Scenespy-Linux-x64` > `chmod +x install_runtime.sh` > `./install_runtime.sh`

### Commands to use in the terminal

Open terminal

| System | How to open |
|---|---|
| Windows | `Win + R` → `cmd` |
| PowerShell | Search for “PowerShell” |
| MacOS | `Command + Space` → `Terminal` |
| Linux | `Ctrl + Alt + T` |

Check which folder you are in

| System | Command |
|---|---|
| Windows CMD | `cd` |
| PowerShell | `pwd` |
| MacOS/Linux | `pwd` |

List files

| System | Command |
|---|---|
| Windows CMD | `dir` |
| PowerShell | `ls` |
| MacOS/Linux | `ls` |

Enter a folder

| System | Command |
|---|---|
| Windows | `cd Downloads` |
| MacOS/Linux | `cd Downloads` |

Go back one folder

| System | Command |
|---|---|
| All | `cd ..` |

Run a file

| System | Command |
|---|---|
| All | `./filename.sh` |

---

## Run from source code

You are responsible for installing Python, Python dependencies, FFmpeg/FFprobe, and libraries.

Requirements for source code:

- Python 3.11.X
- FFmpeg and FFprobe.
- Dependencies from `requirements.txt`.
- On Windows, Microsoft Visual C++ Redistributable x64 may be required for PyTorch.

On Arch, use `pyenv` or another equivalent method to ensure Python 3.11, because the `python` version in the repositories may be newer than what AI dependencies support.

```bash
python -m pip install -r requirements.txt
```

Main Python dependencies installed by `requirements.txt`:

- `customtkinter`: desktop interface.
- `pillow`: images and previews.
- `numpy`: numerical processing.
- `opencv-contrib-python`: frame reading, visual analysis, and MediaPipe dependency.
- `av`: PyAV backend used by PySceneDetect.
- `scenedetect`: base scene change detection.
- `torch` and `torchvision`: required for **Detect faces** mode.
- `ultralytics`: loads the YOLO face model.
- `mediapipe`: facial validation and landmarks.

---

### CPU-only installation 

If you want to guarantee a strictly CPU-only PyTorch installation:

```bash
python -m pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
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

Standard CPU-only installation process
```bash
python -m pip install -r requirements.txt
```

---

### AMD

In Scenespy, AMD is used for video encoding through FFmpeg/AMF (`h264_amf`) when available.

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

### Apple Silicon and MacOS

On MacOS, the app can use VideoToolbox encoding (`h264_videotoolbox`) when the installed FFmpeg provides this encoder.

Encoder test:

```bash
ffmpeg -hide_banner -encoders | grep h264_videotoolbox
```

Note: **Detect faces** mode uses PyTorch on CPU on macOS in this app version. The **Apple** option is for video encoding, not facial inference.

### FFmpeg and FFprobe

Scenespy requires FFmpeg and FFprobe to read, validate, and cut videos.

In the ready-to-use versions from the **Releases** tab, the runtime installer automatically downloads or installs FFmpeg/FFprobe.

On Windows, `install_runtime_windows.bat` installs the FFmpeg essentials build binaries from Gyan.dev into `%LOCALAPPDATA%/Scenespy/runtime/`.

On Linux and MacOS, the recommendation is to install through the system:

```bash
sudo apt install ffmpeg
```

```bash
brew install ffmpeg
```

When starting, Scenespy first searches in `bin/<system>/`. If not found, it searches in the system `PATH`.

## Verifying installation from source code

After installing from source, run:

```bash
python -m pip check
python -c "import customtkinter, PIL, numpy, cv2, av, scenedetect, ultralytics, mediapipe, torch; print('ok')"
ffmpeg -version
ffprobe -version
```

If all commands work, the basic installation is ready.

## Common problems

### `ffmpeg` or `ffprobe` not found

Install FFmpeg and make sure the executables are available in the `PATH`, or place the binaries in `bin/<system>/`.

### Detect faces mode does not open

Check whether `torch`, `ultralytics`, `mediapipe`, and the `models/yolov8n-face.pt` model exist.

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

Use Python 3.11. Some Python versions may not have compatible wheels for all dependencies.

### Both `opencv-python` and `opencv-contrib-python` are installed

The app declares `opencv-contrib-python` because MediaPipe depends on it. Ultralytics may also install `opencv-python`. If `python -m pip check` does not report conflicts and the app opens normally, no mandatory action is required.

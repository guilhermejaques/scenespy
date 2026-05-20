#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path


APP_NAME = "Scenespy"
PYTHON_VERSION = "3.11.9"
TORCH_VERSION = "2.5.1"
TORCHVISION_VERSION = "0.20.1"
AI_PACK_PACKAGES = [
    "ultralytics==8.4.9",
    "mediapipe==0.10.9",
]


def user_data_dir():
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA") or Path.home()) / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")) / "scenespy"


def local_data_dir():
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or Path.home()) / APP_NAME
    return user_data_dir()


def runtime_dir():
    return Path(os.environ.get("SCENESPY_RUNTIME") or (local_data_dir() / "runtime"))


def ai_pack_dir():
    return Path(os.environ.get("SCENESPY_AI_PACK") or (local_data_dir() / "ai-pack"))


def private_python_dir():
    return local_data_dir() / "python311"


def private_python():
    if sys.platform == "win32":
        return private_python_dir() / "python.exe"
    return Path(sys.executable)


def venv_python(pack_dir):
    if sys.platform == "win32":
        return pack_dir / "Scripts" / "python.exe"
    return pack_dir / "bin" / "python"


def platform_name():
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def run(cmd):
    print("+", " ".join(str(part) for part in cmd), flush=True)
    subprocess.check_call([str(part) for part in cmd])


def command_exists(name):
    return shutil.which(name) is not None


def local_asset_dir():
    return Path(__file__).resolve().parent / "runtime-assets" / platform_name() / "ffmpeg"


def install_ffmpeg_from_assets():
    src = local_asset_dir()
    if sys.platform == "win32":
        required = ["ffmpeg.exe", "ffprobe.exe"]
    else:
        required = ["ffmpeg", "ffprobe"]
    if not all((src / name).is_file() for name in required):
        return False

    dest = runtime_dir() / "bin" / platform_name()
    print(f"Copying FFmpeg and FFprobe from bundled Scenespy assets to: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            target = dest / item.name
            shutil.copy2(item, target)
            if item.name in required and sys.platform != "win32":
                target.chmod(target.stat().st_mode | 0o755)
    return True


def install_ffmpeg_with_package_manager():
    if command_exists("ffmpeg") and command_exists("ffprobe"):
        print("FFmpeg and FFprobe are already available on this system.")
        return True
    if sys.platform == "darwin" and command_exists("brew"):
        print("Installing FFmpeg with Homebrew. macOS may ask for permission or developer tools.")
        run(["brew", "install", "ffmpeg"])
        return True
    if sys.platform.startswith("linux"):
        if command_exists("apt"):
            print("Installing FFmpeg with apt. Linux may ask for your sudo password.")
            run(["sudo", "apt", "update"])
            run(["sudo", "apt", "install", "-y", "ffmpeg"])
            return True
        if command_exists("dnf"):
            print("Installing FFmpeg with dnf. Linux may ask for your sudo password.")
            run(["sudo", "dnf", "install", "-y", "ffmpeg"])
            return True
        if command_exists("pacman"):
            print("Installing FFmpeg with pacman. Linux may ask for your sudo password.")
            run(["sudo", "pacman", "-S", "--needed", "ffmpeg"])
            return True
    return False


def verify_ffmpeg():
    candidates = [
        runtime_dir() / "bin" / platform_name() / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"),
        shutil.which("ffmpeg"),
    ]
    ffmpeg = next((Path(p) for p in candidates if p and Path(p).exists()), None)
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not installed.")
    run([ffmpeg, "-version"])


def ensure_ffmpeg():
    print("")
    print("Step 1/2: Preparing FFmpeg and FFprobe")
    print("These tools are used by Scenespy to read, inspect, and cut video files.")
    if install_ffmpeg_from_assets():
        verify_ffmpeg()
        return
    if install_ffmpeg_with_package_manager():
        verify_ffmpeg()
        return
    raise RuntimeError(
        "Could not install FFmpeg automatically. On Windows, keep runtime-assets/windows/ffmpeg "
        "next to this installer. On macOS/Linux, install FFmpeg manually and run this again."
    )


def ensure_private_python_windows():
    py = private_python()
    if py.exists():
        return py

    downloads = local_data_dir() / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    installer = downloads / f"python-{PYTHON_VERSION}-amd64.exe"
    url = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-amd64.exe"
    if not installer.exists():
        print(f"Downloading private Python {PYTHON_VERSION} for Scenespy from python.org...")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, installer)
        except Exception as exc:
            raise RuntimeError(f"Could not download Python from {url}: {exc}") from exc

    print(f"Installing private Python {PYTHON_VERSION} for Scenespy...")
    args = [
        str(installer),
        "/quiet",
        "InstallAllUsers=0",
        f"TargetDir={private_python_dir()}",
        "Include_pip=1",
        "Include_launcher=0",
        "Include_test=0",
        "PrependPath=0",
        "Shortcuts=0",
    ]
    subprocess.check_call(args)
    if not py.exists():
        raise RuntimeError(f"Private Python was installed, but {py} was not found.")
    return py


def python_for_ai_pack():
    if sys.platform == "win32":
        return ensure_private_python_windows()
    return Path(sys.executable)


def has_nvidia_cuda():
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        result = subprocess.run(
            [nvidia_smi],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def torch_index_url():
    if sys.platform in {"win32", "linux"} and has_nvidia_cuda():
        return "https://download.pytorch.org/whl/cu121", "nvidia-cuda"
    if sys.platform == "darwin":
        return None, "cpu"
    return "https://download.pytorch.org/whl/cpu", "cpu"


def install_torch_command(py, index_url):
    cmd = [
        py,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
    ]
    if index_url:
        cmd += ["--index-url", index_url]
    return cmd


def ensure_ai_pack():
    pack_dir = ai_pack_dir()
    python_for_ai_pack()
    py = venv_python(pack_dir)
    index_url, mode = torch_index_url()

    print("")
    print("Step 2/2: Preparing the Scenespy AI packages")
    print("These packages are used only for Detect faces.")
    print(f"Scenespy AI Pack target: {pack_dir}")
    print(f"Selected AI mode: {mode}")
    if mode == "cpu":
        print("AMD/Intel GPUs are still usable by FFmpeg encoding, but face AI uses CPU in this pack.")
    else:
        print("NVIDIA was detected, so the CUDA-enabled PyTorch package will be installed.")

    if not py.exists():
        print(f"Creating isolated AI Python environment at: {pack_dir}")
        pack_dir.parent.mkdir(parents=True, exist_ok=True)
        venv.EnvBuilder(with_pip=True, clear=False).create(pack_dir)

    print("Upgrading pip inside the isolated AI environment...")
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    print("Installing PyTorch. This is a large download and can take several minutes...")
    run(install_torch_command(py, index_url))
    print("Installing Ultralytics and MediaPipe for face detection...")
    run([py, "-m", "pip", "install", "--force-reinstall", *AI_PACK_PACKAGES])
    print("Verifying the AI packages...")
    code = (
        "import torch, torchvision, ultralytics, mediapipe; "
        "print('torch', torch.__version__); "
        "print('cuda', torch.cuda.is_available())"
    )
    run([py, "-c", code])


def parse_args():
    parser = argparse.ArgumentParser(description="Install Scenespy runtime and AI dependencies.")
    parser.add_argument("--dry-run", action="store_true", help="show decisions without installing")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Scenespy dependency installer")
    print("")
    print("This installer prepares the external components Scenespy needs:")
    print("- FFmpeg and FFprobe for video processing")
    print("- A private Python/AI environment for Detect faces")
    print("")
    print("Files are installed in user-local Scenespy folders, not into system folders.")
    print("Internet access is required for Python/AI package downloads.")
    print("")
    print(f"Scenespy runtime target: {runtime_dir()}")
    print(f"Scenespy AI Pack target: {ai_pack_dir()}")
    index_url, mode = torch_index_url()
    print(f"Selected AI mode: {mode}")
    print(f"PyTorch index: {index_url or 'default PyPI'}")
    if args.dry_run:
        return

    ensure_ffmpeg()
    ensure_ai_pack()
    print("")
    print("Scenespy dependencies installed successfully.")
    print("Restart Scenespy before using Detect faces.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Install failed with exit code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)
    except Exception as exc:
        print(f"Install failed: {exc}", file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import venv
import zipfile
from pathlib import Path


APP_NAME = "Scenespy"
TORCH_VERSION = "2.5.1"
TORCHVISION_VERSION = "0.20.1"
CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu121"
CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
WINDOWS_FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
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


def platform_name():
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def venv_python(pack_dir):
    if sys.platform == "win32":
        return pack_dir / "Scripts" / "python.exe"
    return pack_dir / "bin" / "python"


def run(cmd):
    print("+", " ".join(str(part) for part in cmd), flush=True)
    subprocess.check_call([str(part) for part in cmd])


def command_output(cmd):
    return subprocess.check_output([str(part) for part in cmd], text=True).strip()


def command_exists(name):
    return shutil.which(name) is not None


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


def selected_torch_mode(requested):
    if requested != "auto":
        return requested
    if sys.platform in {"win32", "linux"} and has_nvidia_cuda():
        return "cuda"
    return "cpu"


def torch_index_url(mode):
    return CUDA_INDEX_URL if mode == "cuda" else CPU_INDEX_URL


def write_installed_torch_constraints(py, target):
    code = (
        "from importlib.metadata import version; "
        "print('torch==' + version('torch')); "
        "print('torchvision==' + version('torchvision'))"
    )
    target.write_text(command_output([py, "-c", code]) + "\n", encoding="utf-8")


def ensure_ai_pack(torch_mode):
    pack_dir = ai_pack_dir()
    py = venv_python(pack_dir)
    if not py.exists():
        print(f"Creating Scenespy AI environment: {pack_dir}")
        pack_dir.parent.mkdir(parents=True, exist_ok=True)
        venv.EnvBuilder(with_pip=True, clear=False).create(pack_dir)

    print("Installing AI packages for Detect faces.")
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    run([
        py,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--force-reinstall",
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        "--index-url",
        torch_index_url(torch_mode),
    ])
    with tempfile.TemporaryDirectory(prefix="scenespy-ai-constraints-") as tmp_name:
        constraints = Path(tmp_name) / "constraints.txt"
        write_installed_torch_constraints(py, constraints)
        run([py, "-m", "pip", "install", "--no-cache-dir", "-c", constraints, *AI_PACK_PACKAGES])

    code = (
        "import torch, torchvision, ultralytics, mediapipe; "
        "print('torch', torch.__version__); "
        "print('torchvision', torchvision.__version__); "
        "print('cuda', torch.cuda.is_available())"
    )
    run([py, "-c", code])
    if torch_mode == "cuda":
        run([py, "-c", "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"])


def ensure_ai_pack_with_fallback(requested_mode):
    torch_mode = selected_torch_mode(requested_mode)
    if requested_mode == "auto" and torch_mode == "cuda":
        try:
            ensure_ai_pack("cuda")
            return "cuda"
        except Exception as exc:
            print(f"CUDA PyTorch install failed, falling back to CPU PyTorch: {exc}")
            ensure_ai_pack("cpu")
            return "cpu"
    ensure_ai_pack(torch_mode)
    return torch_mode


def download_file(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url}")
    with urllib.request.urlopen(url) as response, open(target, "wb") as output:
        shutil.copyfileobj(response, output)


def install_ffmpeg_windows():
    dest = runtime_dir() / "bin" / "windows"
    ffmpeg = dest / "ffmpeg.exe"
    ffprobe = dest / "ffprobe.exe"
    if ffmpeg.exists() and ffprobe.exists():
        print(f"FFmpeg already installed: {dest}")
        return

    with tempfile.TemporaryDirectory(prefix="scenespy-ffmpeg-") as tmp_name:
        tmp = Path(tmp_name)
        archive = tmp / "ffmpeg-release-essentials.zip"
        download_file(WINDOWS_FFMPEG_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
        roots = [p for p in tmp.iterdir() if p.is_dir() and (p / "bin" / "ffmpeg.exe").exists()]
        if not roots:
            raise RuntimeError("Downloaded FFmpeg archive did not contain ffmpeg.exe.")
        src = roots[0]
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("ffmpeg.exe", "ffprobe.exe"):
            shutil.copy2(src / "bin" / name, dest / name)
        for name in ("LICENSE", "README.txt"):
            source = src / name
            if source.exists():
                shutil.copy2(source, dest / f"FFMPEG-{name}")


def install_ffmpeg_package_manager():
    if command_exists("ffmpeg") and command_exists("ffprobe"):
        print("FFmpeg and FFprobe are already available in PATH.")
        return
    if sys.platform == "darwin" and command_exists("brew"):
        run(["brew", "install", "ffmpeg"])
        return
    if sys.platform.startswith("linux"):
        if command_exists("apt"):
            run(["sudo", "apt", "update"])
            run(["sudo", "apt", "install", "-y", "ffmpeg"])
            return
        if command_exists("dnf"):
            run(["sudo", "dnf", "install", "-y", "ffmpeg"])
            return
        if command_exists("pacman"):
            run(["sudo", "pacman", "-S", "--needed", "ffmpeg"])
            return
    raise RuntimeError("Could not install FFmpeg automatically on this OS.")


def ensure_ffmpeg():
    print("Installing FFmpeg and FFprobe.")
    if sys.platform == "win32":
        install_ffmpeg_windows()
    else:
        install_ffmpeg_package_manager()


def parse_args():
    parser = argparse.ArgumentParser(description="Install Scenespy runtime dependencies.")
    parser.add_argument(
        "--torch-mode",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Choose PyTorch CPU, CUDA, or auto-detect.",
    )
    parser.add_argument("--skip-ffmpeg", action="store_true", help="Do not install FFmpeg/FFprobe.")
    parser.add_argument("--skip-ai", action="store_true", help="Do not install Detect faces AI packages.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch_mode = selected_torch_mode(args.torch_mode)
    print("Scenespy runtime installer")
    print(f"Runtime folder: {runtime_dir()}")
    print(f"AI folder: {ai_pack_dir()}")
    print(f"Selected PyTorch mode: {torch_mode}")
    if not args.skip_ffmpeg:
        ensure_ffmpeg()
    if not args.skip_ai:
        torch_mode = ensure_ai_pack_with_fallback(args.torch_mode)
        print(f"Installed PyTorch mode: {torch_mode}")
    print("Scenespy runtime dependencies installed.")
    print("Restart Scenespy before using the installed components.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Install failed with exit code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)
    except Exception as exc:
        print(f"Install failed: {exc}", file=sys.stderr)
        sys.exit(1)

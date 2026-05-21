#!/usr/bin/env python3
import argparse
import json
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
CUDA_MIN_DRIVER_LINUX = (530, 30, 2)
CUDA_MIN_DRIVER_WINDOWS = (531, 14)
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


def platform_bin_dir():
    return runtime_dir() / "bin" / platform_name()


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


def parse_driver_version(value):
    parts = []
    for chunk in str(value).strip().split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if digits == "":
            break
        parts.append(int(digits))
    return tuple(parts)


def driver_at_least(found, required):
    padded_found = tuple(found) + (0,) * max(0, len(required) - len(found))
    padded_required = tuple(required) + (0,) * max(0, len(found) - len(required))
    return padded_found >= padded_required


def cuda_visible_devices_allows_cuda():
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return True
    normalized = value.strip().lower()
    return normalized not in {"", "-1", "none", "nodevfiles", "void"}


def nvidia_smi_gpu_rows(nvidia_smi):
    result = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def has_nvidia_cuda():
    if sys.platform not in {"win32", "linux"}:
        return False
    if not cuda_visible_devices_allows_cuda():
        print("CUDA_VISIBLE_DEVICES disables CUDA; using CPU PyTorch.")
        return False
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        rows = nvidia_smi_gpu_rows(nvidia_smi)
        if not rows:
            return False
        required = CUDA_MIN_DRIVER_WINDOWS if sys.platform == "win32" else CUDA_MIN_DRIVER_LINUX
        for row in rows:
            parts = [part.strip() for part in row.rsplit(",", 1)]
            driver = parse_driver_version(parts[-1] if parts else "")
            if driver and driver_at_least(driver, required):
                gpu_name = parts[0] if len(parts) > 1 else "NVIDIA GPU"
                print(f"CUDA-capable GPU detected: {gpu_name}, driver {'.'.join(map(str, driver))}")
                return True
        print(
            "NVIDIA GPU detected, but the driver is too old for the bundled "
            f"CUDA 12.1 PyTorch wheels. Required driver: {'.'.join(map(str, required))}+."
        )
        return False
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


def installed_torch_info(py):
    code = r"""
import json
try:
    import torch
    import torchvision
    print(json.dumps({
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda": bool(getattr(torch.version, "cuda", None)),
        "cuda_available": bool(torch.cuda.is_available()),
    }))
except Exception:
    print("{}")
"""
    try:
        output = command_output([py, "-c", code])
        return json.loads(output or "{}")
    except Exception:
        return {}


def version_matches(installed, expected):
    return str(installed or "").split("+", 1)[0] == expected


def installed_torch_matches(py, torch_mode):
    info = installed_torch_info(py)
    if not (
        version_matches(info.get("torch"), TORCH_VERSION)
        and version_matches(info.get("torchvision"), TORCHVISION_VERSION)
    ):
        return False
    if torch_mode == "cuda":
        return bool(info.get("torch_cuda"))
    return not bool(info.get("torch_cuda"))


def install_torch_packages(py, torch_mode, force=False):
    cmd = [
        py,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
    ]
    if force:
        cmd.append("--force-reinstall")
    else:
        cmd.append("--upgrade")
    cmd.extend([
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        "--index-url",
        torch_index_url(torch_mode),
    ])
    run(cmd)


def ensure_ai_pack(torch_mode):
    pack_dir = ai_pack_dir()
    py = venv_python(pack_dir)
    if not py.exists():
        print(f"Creating Scenespy AI environment: {pack_dir}")
        pack_dir.parent.mkdir(parents=True, exist_ok=True)
        venv.EnvBuilder(with_pip=True, clear=False).create(pack_dir)

    print("Installing AI packages for Detect faces.")
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    if installed_torch_matches(py, torch_mode):
        print(f"PyTorch {TORCH_VERSION} for {torch_mode.upper()} is already installed.")
    else:
        install_torch_packages(py, torch_mode, force=False)
        if not installed_torch_matches(py, torch_mode):
            print("Existing PyTorch build did not match the requested mode; reinstalling it.")
            install_torch_packages(py, torch_mode, force=True)
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
    dest = platform_bin_dir()
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
        install_ffmpeg_runtime_links()
        return
    if sys.platform == "darwin" and command_exists("brew"):
        run(["brew", "install", "ffmpeg"])
        install_ffmpeg_runtime_links()
        return
    if sys.platform.startswith("linux"):
        if command_exists("apt"):
            run(["sudo", "apt", "update"])
            run(["sudo", "apt", "install", "-y", "ffmpeg"])
            install_ffmpeg_runtime_links()
            return
        if command_exists("dnf"):
            run(["sudo", "dnf", "install", "-y", "ffmpeg"])
            install_ffmpeg_runtime_links()
            return
        if command_exists("pacman"):
            run(["sudo", "pacman", "-S", "--needed", "ffmpeg"])
            install_ffmpeg_runtime_links()
            return
    raise RuntimeError("Could not install FFmpeg automatically on this OS.")


def install_ffmpeg_runtime_links():
    dest = platform_bin_dir()
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("ffmpeg", "ffprobe"):
        source_name = f"{name}.exe" if sys.platform == "win32" else name
        source = shutil.which(source_name) or shutil.which(name)
        if not source:
            raise RuntimeError(f"{name} was not found after installation.")
        target = dest / source_name
        if target.exists():
            continue
        try:
            target.symlink_to(Path(source).resolve())
        except Exception:
            shutil.copy2(source, target)
        try:
            target.chmod(target.stat().st_mode | 0o111)
        except Exception:
            pass
    print(f"FFmpeg runtime launchers are available in: {dest}")


def ensure_ffmpeg():
    print("Installing FFmpeg and FFprobe.")
    if sys.platform == "win32":
        install_ffmpeg_windows()
    else:
        install_ffmpeg_package_manager()


def runtime_executable(name):
    for filename in (f"{name}.exe", name) if sys.platform == "win32" else (name,):
        candidate = platform_bin_dir() / filename
        if candidate.exists():
            return candidate
    found = shutil.which(name)
    if found:
        return Path(found)
    return None


def verify_ffmpeg():
    ffmpeg = runtime_executable("ffmpeg")
    ffprobe = runtime_executable("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg/FFprobe verification failed: executables were not found.")
    command_output([ffmpeg, "-version"])
    command_output([ffprobe, "-version"])
    print(f"Verified FFmpeg: {ffmpeg}")
    print(f"Verified FFprobe: {ffprobe}")


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
        verify_ffmpeg()
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
        sys.exit(exc.returncode or 1)
    except Exception as exc:
        print(f"Install failed: {exc}", file=sys.stderr)
        sys.exit(1)

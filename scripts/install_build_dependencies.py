#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys


TORCH_VERSION = "2.5.1"
TORCHVISION_VERSION = "0.20.1"
CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu121"
CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"


def run(cmd):
    print("+", " ".join(str(part) for part in cmd), flush=True)
    subprocess.check_call([str(part) for part in cmd])


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


def install_torch(mode):
    if torch_matches_mode(mode):
        print(f"PyTorch {TORCH_VERSION} already matches requested mode: {mode}")
        return
    index_url = CUDA_INDEX_URL if mode == "cuda" else CPU_INDEX_URL
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        "--index-url",
        index_url,
    ])


def torch_matches_mode(mode):
    try:
        import torch
        import torchvision
    except Exception:
        return False

    torch_version = getattr(torch, "__version__", "")
    torchvision_version = getattr(torchvision, "__version__", "")
    if not torch_version.startswith(TORCH_VERSION):
        return False
    if not torchvision_version.startswith(TORCHVISION_VERSION):
        return False
    if mode == "cuda":
        return "+cu121" in torch_version and torch.cuda.is_available()
    return "+cu" not in torch_version


def verify_torch(expected_mode):
    code = (
        "import torch, torchvision; "
        "print('torch', torch.__version__); "
        "print('torchvision', torchvision.__version__); "
        "print('cuda', torch.cuda.is_available())"
    )
    run([sys.executable, "-c", code])
    if expected_mode == "cuda":
        check = "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
        run([sys.executable, "-c", check])


def verify_runtime_imports():
    code = (
        "import customtkinter, PIL, numpy, cv2, av, scenedetect, "
        "torch, torchvision, ultralytics, mediapipe, tkinter; "
        "print('runtime imports ok')"
    )
    run([sys.executable, "-c", code])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Install dependencies required to build a full Scenespy release."
    )
    parser.add_argument(
        "--torch-mode",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="PyTorch wheel flavor to include in the release.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode = selected_torch_mode(args.torch_mode)
    print(f"Selected PyTorch build mode: {mode}")
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-r", "requirements-base.txt"])
    install_torch(mode)
    run([sys.executable, "-m", "pip", "install", "-r", "requirements-ai.txt", "pyinstaller"])
    verify_torch(mode)
    verify_runtime_imports()


if __name__ == "__main__":
    main()

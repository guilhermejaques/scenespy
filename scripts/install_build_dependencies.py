#!/usr/bin/env python3
import subprocess
import sys


def runtime_requirements_file():
    if sys.platform == "darwin":
        return "requirements-macos-build.txt"
    return "requirements-base.txt"


def run(cmd):
    print("+", " ".join(str(part) for part in cmd), flush=True)
    subprocess.check_call([str(part) for part in cmd])


def verify_runtime_imports():
    code = (
        "import PySide6, PIL, numpy, cv2, av, scenedetect; "
        "print('base runtime imports ok')"
    )
    run([sys.executable, "-c", code])


def main():
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        runtime_requirements_file(),
        "pyinstaller",
    ]
    if sys.platform == "darwin":
        install_cmd.insert(4, "--only-binary=:all:")
    run(install_cmd)
    verify_runtime_imports()


if __name__ == "__main__":
    main()

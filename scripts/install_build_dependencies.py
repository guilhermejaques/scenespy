#!/usr/bin/env python3
import subprocess
import sys


def runtime_requirements_file():
    if sys.platform == "darwin":
        return "requirements-macos-build.txt"
    return "requirements-base.txt"


def ensure_tkinter():
    try:
        import tkinter  # noqa: F401
    except ModuleNotFoundError:
        if sys.platform.startswith("linux"):
            hint = "Install python3-tk or python3.11-tk, then rerun the build."
        elif sys.platform == "darwin":
            hint = "Install a Python build with Tcl/Tk support, then rerun the build."
        else:
            hint = "Install a Python build with tkinter support, then rerun the build."
        raise SystemExit(f"Python tkinter support is missing. {hint}")


def run(cmd):
    print("+", " ".join(str(part) for part in cmd), flush=True)
    subprocess.check_call([str(part) for part in cmd])


def verify_runtime_imports():
    code = (
        "import customtkinter, PIL, numpy, cv2, av, scenedetect, tkinter; "
        "print('base runtime imports ok')"
    )
    run([sys.executable, "-c", code])


def main():
    ensure_tkinter()
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

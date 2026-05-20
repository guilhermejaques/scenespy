#!/usr/bin/env python3
import subprocess
import sys


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
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-r", "requirements-base.txt", "pyinstaller"])
    verify_runtime_imports()


if __name__ == "__main__":
    main()

import os
import sys

from .shared import (AI_PACK_DIR, _ensure_mediapipe, _ensure_torch, _ensure_yolo,
                     configure_linux_virtual_machine_graphics,
                     install_crash_logging, mediapipe_required,
                     runtime_import_error_message, single_instance,
                     validate_runtime_dependencies)


def _runtime_selftest():
    print(f"AI pack folder: {AI_PACK_DIR}")
    checks = [("torch", _ensure_torch), ("ultralytics", lambda: _ensure_yolo() is not None)]
    if mediapipe_required():
        checks.append(("mediapipe", lambda: _ensure_mediapipe() is not None))
    failed = False
    for name, check in checks:
        ok = bool(check())
        print(f"{name}: {'ok' if ok else 'failed'}")
        failed = failed or not ok
        if not ok and runtime_import_error_message(name):
            print(runtime_import_error_message(name), file=sys.stderr)
    raise SystemExit(1 if failed else 0)


def main():
    install_crash_logging()
    configure_linux_virtual_machine_graphics()
    if os.environ.get("SCENESPY_RUNTIME_SELFTEST") == "1":
        _runtime_selftest()
    from PySide6.QtWidgets import QMessageBox
    from .qt_app import ScenespyApp, create_application
    application = create_application(sys.argv)
    if os.environ.get("SCENESPY_APP_SELFTEST") == "1":
        # Exercise the frozen Qt application, plugins, bundled resources and
        # main-window construction without requiring FFmpeg or a display. The
        # release jobs run this before an installer can be published.
        window = ScenespyApp()
        if window.menuBar().isNativeMenuBar():
            print("Application self-test failed: native menu bar is enabled.",
                  file=sys.stderr)
            raise SystemExit(1)
        print("Scenespy packaged application self-test passed.")
        window.deleteLater()
        application.processEvents()
        raise SystemExit(0)
    ok, message = validate_runtime_dependencies()
    if not ok:
        QMessageBox.critical(None, "Missing Required Dependencies", message)
        raise SystemExit(1)
    if not single_instance():
        QMessageBox.information(None, "Application Already Running", "This application is already running.")
        raise SystemExit(0)
    window = ScenespyApp()
    window.showMaximized()
    raise SystemExit(application.exec())

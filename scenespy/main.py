from .shared import (
    AI_PACK_DIR,
    ctk,
    mb,
    os,
    sys,
    _ensure_mediapipe,
    _ensure_torch,
    _ensure_yolo,
    install_crash_logging,
    mediapipe_required,
    register_bundled_fonts,
    runtime_import_error_message,
    single_instance,
    validate_runtime_dependencies,
)
from .app import ScenespyApp


def _show_error_or_print(title, message):
    try:
        mb.showerror(title, message)
    except Exception:
        print(f"{title}: {message}", file=sys.stderr)


def _runtime_selftest():
    print(f"AI pack folder: {AI_PACK_DIR}")
    checks = [
        ("torch", _ensure_torch),
        ("ultralytics", lambda: _ensure_yolo() is not None),
    ]
    if mediapipe_required():
        checks.append(("mediapipe", lambda: _ensure_mediapipe() is not None))
    else:
        print("mediapipe: skipped (optional on macOS before 13)")
    failed = False
    for name, check in checks:
        ok = bool(check())
        print(f"{name}: {'ok' if ok else 'failed'}")
        if not ok:
            failed = True
            detail = runtime_import_error_message(name)
            if detail:
                print(detail, file=sys.stderr)
    sys.exit(1 if failed else 0)


def main():
    install_crash_logging()

    if os.environ.get("SCENESPY_RUNTIME_SELFTEST") == "1":
        _runtime_selftest()

    deps_ok, deps_message = validate_runtime_dependencies()
    if not deps_ok:
        _show_error_or_print("Missing Required Dependencies", deps_message)
        sys.exit(1)

    if not single_instance():
        _show_error_or_print("Application Already Running",
                             "This application is already running.")
        sys.exit(0)

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    register_bundled_fonts()
    app = ScenespyApp()
    app.mainloop()

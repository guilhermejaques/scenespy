from .shared import (
    ctk,
    mb,
    sys,
    install_crash_logging,
    register_bundled_fonts,
    single_instance,
    validate_runtime_dependencies,
)
from .app import ScenespyApp


def _show_error_or_print(title, message):
    try:
        mb.showerror(title, message)
    except Exception:
        print(f"{title}: {message}", file=sys.stderr)


def main():
    install_crash_logging()

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


__version__ = "0.1.1"

__all__ = [
    "main",
    "ScenespyApp",
    "detect_scenes",
    "split_video",
    "extract_faces",
    "process_video",
    "process_videos",
    "__version__",
]


def main():
    from .main import main as run
    return run()


def __getattr__(name):
    if name == "ScenespyApp":
        from .app import ScenespyApp
        return ScenespyApp
    if name in {
        "detect_scenes",
        "split_video",
        "extract_faces",
        "process_video",
        "process_videos",
    }:
        from . import api
        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

Scenespy can use bundled FFmpeg binaries from this folder.

Expected layout:

```text
bin/
  windows/
    ffmpeg.exe
    ffprobe.exe
  macos/
    ffmpeg
    ffprobe
  linux/
    ffmpeg
    ffprobe
```

At startup, Scenespy checks this folder first. If the binaries are not found
here, it falls back to the system PATH.

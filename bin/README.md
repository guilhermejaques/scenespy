Scenespy can use bundled FFmpeg binaries from this folder.

Expected layout:

```text
bin/
  windows/
    ffmpeg.exe
    ffprobe.exe
    FFMPEG-LICENSE.txt
    FFMPEG-README.txt
    FFMPEG-SHA256.txt
  macos/
    ffmpeg
    ffprobe
  linux/
    ffmpeg
    ffprobe
```

At startup, Scenespy checks this folder first. If the binaries are not found
here, it falls back to the system PATH.

Keep this folder lightweight in Git. The Windows FFmpeg binaries used to build
release ZIPs should stay in the local ignored folder:

```text
release-assets/
  windows/
    ffmpeg/
      ffmpeg.exe
      ffprobe.exe
      FFMPEG-LICENSE.txt
      FFMPEG-README.txt
      FFMPEG-SHA256.txt
```

When preparing a Windows release, copy those files into `bin/windows/` inside
the final release folder before zipping it.

The recommended Windows source is the Gyan.dev FFmpeg essentials build:

https://www.gyan.dev/ffmpeg/builds/

For Linux and macOS, prefer installing FFmpeg through the system package
manager unless you are preparing dedicated platform builds.

# MKV Debug Context

Date: 2026-05-16

## Goal

Investigate and fix scene detection/cutting problems that happen only with some MKV files, without breaking the current behavior for MP4 files and MKVs that already work well.

Primary problematic file:

```text
D:\origemSceneDetect\Leon.94.Trial4KBD1080p.MemoriadaTV.Mini\Leon.94.Trial4KBD1080p.MemoriadaTV.Mini.mkv
```

Known-good comparison file:

```text
C:\Users\hvs\Downloads\sample_1280x720_surfing_with_audio.mkv
```

## Observed Symptoms

- Some MKV outputs had cuts with impossible/very long durations.
- Some cuts appeared to have the size/timeline of the full movie.
- Some players continued playing/advancing the timeline after the visible video had already ended.
- The issue did not affect all MKVs.
- The MP4 pipeline and many good MKVs should remain unchanged.

## Important Run Folders

Older run:

```text
D:\origemSceneDetect\Leon.94.Trial4KBD1080p.MemoriadaTV.Mini\SD_20260516_011034_301_Low_nvidia
```

Newer run observed while still being written:

```text
D:\origemSceneDetect\Leon.94.Trial4KBD1080p.MemoriadaTV.Mini\SD_20260516_184929_650_Normal_nvidia
```

The newer run was started before the final chapter/data-stream fix was loaded into the running app process, so it should not be treated as a valid post-fix test.

## Pipeline Shape

Relevant files:

```text
scenespy/app.py
scenespy/shared.py
scenespy/scene_engine.py
```

The app does:

1. `app.py` calls `prepare_video_for_processing(video, temp_files=temp_files)`.
2. `shared.py` remuxes MKV inputs into a temporary `_fixed.mkv`.
3. `SceneEngine` runs detection on the prepared video.
4. Scene boundaries are converted from frames to segments.
5. Each segment is exported with ffmpeg, usually through `_run_ffmpeg_precise_cut`.
6. Outputs are validated by `_validate_cut_output`.

Existing MKV remux logic in `shared.py`:

```text
ffmpeg -y -fflags +genpts+igndts -err_detect ignore_err -i input.mkv
  -map 0:v:0 -map 0:a? -c copy
  -max_interleave_delta 0 -avoid_negative_ts make_zero fixed.mkv
```

## Findings

The problem is not "MKV" as a format in general.

The known-good sample MKV is simple:

- 2 streams.
- Video + audio.
- Clean short duration.

The problematic Leon MKV is complex:

- 18 streams in the original file.
- Multiple audio tracks.
- Multiple SRT/PGS subtitle streams.
- Chapters.
- Different durations across video, audio, subtitles, and chapters.

Useful ffprobe finding for the original Leon MKV:

```text
video stream tags:
NUMBER_OF_FRAMES = 191283
DURATION = 02:12:58.095000000

format duration:
7978.095000
```

After automatic remux, an observed `_fixed.mkv` had:

```text
video stream tags:
NUMBER_OF_FRAMES = 191283
DURATION = 02:12:58.177000000

format duration:
7978.177000
```

This matters because using only `format=duration` can push the computed end slightly past the actual video frame count.

## First Fix Applied

File:

```text
scenespy/scene_engine.py
```

Changes:

- Added `_get_video_total_frames()`.
- Prefer explicit video frame counts from:
  - `stream.nb_frames`
  - `stream.tags.NUMBER_OF_FRAMES`
- Use stream duration only as fallback.
- Use format duration only as fallback.
- Use this safer frame count when composing scenes.
- Merge tiny final tail scenes into the previous scene.
- Strengthen `_validate_cut_output()` to remove and fail outputs that are too short or too long relative to the expected segment duration.

Validation command used:

```powershell
python -m compileall scenespy
```

Manual check for original Leon MKV:

```text
fps = 23.976023976023978
duration = 7978.095
safe_total_frames = 191283
safe_end_seconds = 7978.095125
```

Manual check for sample MKV:

```text
fps = 23.976023976023978
duration = 183.129
safe_total_frames = 4389
safe_end_seconds = 183.057875
```

## Second Finding: Real Cause Of "Player Keeps Running"

The major remaining issue was not video frames continuing after the cut.

Example output from the newer run:

```text
scene_043.mp4
```

`ffprobe -show_streams -show_format` showed:

```text
stream 0: video, duration 54.012s
stream 1: audio, duration 54.012s
stream 2: bin_data / SubtitleHandler, duration 7590.709s
format duration: 54.012s
```

Another example:

```text
scene_049.mp4
```

Showed:

```text
stream 0: video, duration 72.697625s
stream 1: audio, duration 72.697s
stream 2: bin_data / SubtitleHandler, duration 7472.132s
format duration: 72.697625s
```

Conclusion:

- The video and audio were cut correctly.
- The MP4 inherited chapter/data/subtitle metadata from the source MKV.
- That extra `bin_data`/`SubtitleHandler` stream had a huge duration.
- Some players used that stream to keep the timeline alive after video/audio ended.

## Final Fix Applied

File:

```text
scenespy/scene_engine.py
```

In `_run_ffmpeg_copy()`:

```text
-map 0:v:0 -map 0:a:0? -sn -dn -map_metadata -1 -map_chapters -1
```

In `_run_ffmpeg_precise_cut()` and CPU fallback:

```text
-sn -dn -map_metadata -1 -map_chapters -1
```

In `_validate_cut_output()`:

- It now reads full `ffprobe` JSON.
- It rejects and removes outputs containing streams other than:
  - `video`
  - `audio`

This is intended to prevent `bin_data`, subtitle streams, chapters, or other data streams from remaining in MP4 outputs.

## Manual Validation Of Final Fix

Command shape used through Python:

```text
SceneEngine(...)._run_ffmpeg_precise_cut(...)
SceneEngine(...)._validate_cut_output(...)
```

Temporary output:

```text
_tmp_chapterless_cut.mp4
```

`ffprobe` result:

```json
{
  "streams": [
    {
      "index": 0,
      "codec_name": "h264",
      "codec_type": "video",
      "duration": "4.963292"
    },
    {
      "index": 1,
      "codec_name": "aac",
      "codec_type": "audio",
      "duration": "4.963000"
    }
  ],
  "format": {
    "duration": "4.963292"
  }
}
```

This confirmed that the fixed cut contains only video/audio and no long `bin_data` stream.

## Important Runtime Note

Python does not reload modified modules inside an already-running app process.

If the app was already open when `scene_engine.py` was changed, the running process may still use the old logic.

For a valid test:

1. Stop the current processing.
2. Close the app completely.
3. Reopen the app.
4. Reprocess the Leon MKV into a fresh output folder.
5. Ignore older output folders when evaluating the fix.

## Commands Useful For Rechecking

List output folders:

```powershell
Get-ChildItem -Force -LiteralPath 'D:\origemSceneDetect\Leon.94.Trial4KBD1080p.MemoriadaTV.Mini' |
  Sort-Object LastWriteTime -Descending
```

Check streams in one output cut:

```powershell
ffprobe -v error -show_streams -show_format -of json 'D:\path\to\scene_043.mp4'
```

Quick check for any non-video/audio streams in outputs:

```powershell
$dir='D:\origemSceneDetect\Leon.94.Trial4KBD1080p.MemoriadaTV.Mini\OUTPUT_FOLDER'
Get-ChildItem -LiteralPath $dir -Filter 'scene_*.mp4' |
  ForEach-Object {
    $json = ffprobe -v error -show_streams -of json $_.FullName | ConvertFrom-Json
    $bad = @($json.streams | Where-Object { $_.codec_type -notin @('video','audio') })
    if ($bad.Count -gt 0) {
      [PSCustomObject]@{
        File = $_.Name
        BadStreams = ($bad | ForEach-Object { $_.codec_type + ':' + $_.codec_name }) -join ','
      }
    }
  }
```

Check biggest outputs by size and their format duration:

```powershell
$dir='D:\origemSceneDetect\Leon.94.Trial4KBD1080p.MemoriadaTV.Mini\OUTPUT_FOLDER'
Get-ChildItem -LiteralPath $dir -Filter 'scene_*.mp4' |
  Where-Object {$_.Length -gt 0} |
  Sort-Object Length -Descending |
  Select-Object -First 15 |
  ForEach-Object {
    $d = ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $_.FullName
    [PSCustomObject]@{
      Name = $_.Name
      MB = [math]::Round($_.Length/1MB,2)
      Duration = $d
    }
  } | Format-Table -AutoSize
```

Check source MKV video frame metadata:

```powershell
ffprobe -v error -select_streams v:0 `
  -show_entries stream=nb_frames:stream_tags=NUMBER_OF_FRAMES,DURATION `
  -show_entries format=duration `
  -of json 'D:\origemSceneDetect\Leon.94.Trial4KBD1080p.MemoriadaTV.Mini\Leon.94.Trial4KBD1080p.MemoriadaTV.Mini.mkv'
```

## Current Code State

Modified file:

```text
scenespy/scene_engine.py
```

Main changed areas:

- Safe video frame count:

```text
_get_video_total_frames()
_parse_duration_seconds()
```

- Scene list tail handling:

```text
_scenes_from_boundaries(..., fps=fps)
```

- Output validation:

```text
_validate_cut_output()
```

- ffmpeg output mapping:

```text
_run_ffmpeg_copy()
_run_ffmpeg_precise_cut()
```

## Next Steps

1. Stop any currently running app/process that was started before the final fix.
2. Restart the app.
3. Run the Leon MKV again.
4. Inspect a few output cuts with `ffprobe -show_streams -show_format`.
5. Confirm outputs contain only `video` and optionally `audio`.
6. Confirm player timeline ends when video/audio end.

If the issue persists after a fresh app restart, the next likely adjustment is to force stricter timestamp generation in the precise cut command:

```text
-fflags +genpts
-avoid_negative_ts make_zero
-shortest
-muxpreload 0
-muxdelay 0
```

And possibly change audio trimming from absolute `atrim=start=...:end=...` to duration-based trimming.

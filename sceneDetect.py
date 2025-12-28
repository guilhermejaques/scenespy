from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from tqdm import tqdm
import subprocess
import os
import shutil
import datetime
import sys

# ================= CONFIGURAÇÕES =================
VIDEO_PATH = r"D:\origemSceneDetect\blake.mp4"
OUTPUT_DIR = r"D:\saidaSceneDetect"

MODE = "scene_detect"   # scene_detect | fixed_interval
CUT_INTERVAL_SECONDS = 10.0  # usado apenas no fixed_interval

PROFILE = "mais_cortes"  # menos_cortes | normal | mais_cortes
DRY_RUN = False
# ================================================


# ===================== PERFIS ====================
PROFILES = {
    "menos_cortes": {
        "THRESHOLD": 45.0,
        "MIN_SCENE_LEN_FRAMES": 10,
        "DOWNSCALE": 4,
        "MIN_FINAL_DURATION": 5.5,
    },
    "normal": {
        "THRESHOLD": 28.0,
        "MIN_SCENE_LEN_FRAMES": 4,
        "DOWNSCALE": 3,
        "MIN_FINAL_DURATION": 1.8,
    },
    "mais_cortes": {
        "THRESHOLD": 18.0,
        "MIN_SCENE_LEN_FRAMES": 2,
        "DOWNSCALE": 2,
        "MIN_FINAL_DURATION": 0.9,
    },
}
# ================================================


def validate_environment(cfg):
    if not os.path.isfile(VIDEO_PATH):
        raise FileNotFoundError(f"Arquivo de vídeo não encontrado:\n{VIDEO_PATH}")

    if not shutil.which("ffmpeg"):
        raise EnvironmentError("FFmpeg não encontrado no PATH.")

    if MODE == "scene_detect":
        if PROFILE not in PROFILES:
            raise ValueError(f"Perfil inválido: {PROFILE}")
        if cfg["THRESHOLD"] <= 0 or cfg["MIN_SCENE_LEN_FRAMES"] <= 0:
            raise ValueError("Parâmetros inválidos")

    if MODE == "fixed_interval" and CUT_INTERVAL_SECONDS <= 0:
        raise ValueError("CUT_INTERVAL_SECONDS inválido")


def create_output_directory():
    video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = PROFILE if MODE == "scene_detect" else f"interval_{CUT_INTERVAL_SECONDS}s"
    path = os.path.join(OUTPUT_DIR, f"{video_name}_{suffix}_{timestamp}")
    os.makedirs(path, exist_ok=True)
    return path


def get_video_duration():
    video = open_video(VIDEO_PATH)
    return video.duration.get_seconds()


# ================= SCENE DETECT =================
def detect_scenes(cfg):
    video = open_video(VIDEO_PATH)
    video.downscale = cfg["DOWNSCALE"]

    manager = SceneManager()
    manager.add_detector(
        ContentDetector(
            threshold=cfg["THRESHOLD"],
            min_scene_len=cfg["MIN_SCENE_LEN_FRAMES"],
        )
    )

    print(f"\n🔍 Detectando cenas (perfil: {PROFILE})...\n")
    manager.detect_scenes(video, show_progress=True)

    scenes = manager.get_scene_list()
    print(f"\n✅ Cenas detectadas (brutas): {len(scenes)}\n")
    return scenes


# 🔑 Normalização contínua (NUNCA exclui tempo)
def normalize_scenes(scene_list, cfg):
    min_duration = cfg["MIN_FINAL_DURATION"]
    normalized = []

    buffer_start = None
    buffer_end = None

    for scene in scene_list:
        start = scene[0].get_seconds()
        end = scene[1].get_seconds()

        if buffer_start is None:
            buffer_start = start
            buffer_end = end
        else:
            buffer_end = end

        if (buffer_end - buffer_start) >= min_duration:
            normalized.append((buffer_start, buffer_end))
            buffer_start = None
            buffer_end = None

    if buffer_start is not None:
        normalized.append((buffer_start, buffer_end))

    print(f"✨ Cenas finais (timeline contínua): {len(normalized)}\n")
    return normalized


# ================= FIXED INTERVAL =================
def generate_fixed_interval_scenes():
    duration = get_video_duration()
    scenes = []

    t = 0.0
    while t < duration:
        end = min(t + CUT_INTERVAL_SECONDS, duration)
        scenes.append((t, end))
        t = end

    print(f"✂️ Cortes fixos gerados: {len(scenes)}\n")
    return scenes


# ================= CORTE =================
def cut_scenes(scene_list, output_base):
    for i, (start, end) in enumerate(
        tqdm(scene_list, desc="✂️ Cortando cenas", unit="cena", ncols=80)
    ):
        duration = end - start
        output_file = os.path.join(output_base, f"scene_{i+1:03d}.mp4")

        if DRY_RUN:
            print(f"[DRY] {i+1}: {start:.3f}s → {end:.3f}s ({duration:.3f}s)")
            continue

        cmd = [
            "ffmpeg",
            "-y",
            "-i", VIDEO_PATH,
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "copy",
            output_file,
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    try:
        cfg = PROFILES.get(PROFILE)
        validate_environment(cfg)

        output_base = create_output_directory()

        if MODE == "scene_detect":
            raw_scenes = detect_scenes(cfg)
            final_scenes = normalize_scenes(raw_scenes, cfg)
        else:
            final_scenes = generate_fixed_interval_scenes()

        cut_scenes(final_scenes, output_base)

        print("\n🎉 Finalizado com sucesso!")
        print("📁 Saída:", output_base)

    except Exception as e:
        print("\n❌ ERRO:")
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()

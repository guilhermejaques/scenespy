import os
import time
import threading
import customtkinter as ctk
from PIL import Image

from core.engine import SceneEngine
from gui.theme import COLORS
from gui.widgets import (
    Section,
    LabeledEntry,
    LogBox,
    ProgressBar,
    PreviewFrame,
    FileSelector,
    DirectorySelector
)

# ========================== Configurações ==========================
ENABLE_PREVIEW = True

PROFILES = {
    "menos_cortes": {
        "label": "Menos cortes",
        "THRESHOLD": 45.0,
        "MIN_SCENE_LEN_FRAMES": 10,
        "DOWNSCALE": 4,
        "MIN_FINAL_DURATION": 5.5,
    },
    "normal": {
        "label": "Normal",
        "THRESHOLD": 28.0,
        "MIN_SCENE_LEN_FRAMES": 4,
        "DOWNSCALE": 3,
        "MIN_FINAL_DURATION": 1.8,
    },
    "mais_cortes": {
        "label": "Mais cortes",
        "THRESHOLD": 18.0,
        "MIN_SCENE_LEN_FRAMES": 2,
        "DOWNSCALE": 2,
        "MIN_FINAL_DURATION": 0.9,
    },
}

# ========================== App ==========================
class SceneCutterApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Scene Cutter Pro")
        self.geometry("1000x650")
        self.configure(fg_color=COLORS["bg"])

        self.engine = None
        self.running = False

        self._last_preview_time = 0.0
        self._preview_min_interval = 0.12

        self._build_ui()

    # ========================== UI ==========================
    def _build_ui(self):
        self.left = ctk.CTkFrame(self, width=300, fg_color=COLORS["panel"])
        self.left.pack(side="left", fill="y", padx=10, pady=10)

        self.right = ctk.CTkFrame(self, fg_color=COLORS["panel"])
        self.right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self._build_controls()
        self._build_preview()

    def _build_controls(self):
        files = Section(self.left, "Arquivos")
        files.pack(fill="x", padx=10, pady=8)

        self.video_selector = FileSelector(files, "Vídeo de origem")
        self.video_selector.pack(fill="x", padx=12)

        self.output_selector = DirectorySelector(files)
        self.output_selector.pack(fill="x", padx=12)

        mode = Section(self.left, "Modo de corte")
        mode.pack(fill="x", padx=10, pady=8)

        self.cut_mode = ctk.StringVar(value="scene")
        self.mode_radios = []

        row = ctk.CTkFrame(mode, fg_color="transparent")
        row.pack(fill="x", padx=12)

        for text, value in [
            ("Detecção de cenas", "scene"),
            ("A cada X segundos", "interval")
        ]:
            rb = ctk.CTkRadioButton(
                row, text=text, variable=self.cut_mode,
                value=value, command=self._update_mode_ui
            )
            rb.pack(side="left", padx=6)
            self.mode_radios.append(rb)

        self.interval_entry = LabeledEntry(
            mode, "Intervalo (segundos)", placeholder="Ex: 10"
        )

        profile = Section(self.left, "Perfil de detecção")
        profile.pack(fill="x", padx=10, pady=8)

        self.profile = ctk.StringVar(value="normal")
        self.profile_radios = []

        row = ctk.CTkFrame(profile, fg_color="transparent")
        row.pack(fill="x", padx=12)

        for key, cfg in PROFILES.items():
            rb = ctk.CTkRadioButton(
                row, text=cfg["label"], variable=self.profile, value=key
            )
            rb.pack(side="left", padx=6)
            self.profile_radios.append(rb)

        self.start_btn = ctk.CTkButton(
            self.left, text="Iniciar", command=self.toggle_start
        )
        self.start_btn.pack(pady=20)

        self.status_label = ctk.CTkLabel(
            self.left,
            text="Aguardando início...",
            anchor="w",
            font=("Segoe UI", 11)
        )
        self.status_label.pack(fill="x", padx=12, pady=(0, 6))

        self.log = LogBox(self.left, height=200)
        self.log.pack(fill="x", padx=10, pady=10)

    def _build_preview(self):
        section = Section(self.right, "Processo")
        section.pack(fill="both", expand=True)

        self.preview_frame = PreviewFrame(section)
        self.preview_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.progress = ProgressBar(self.right)
        self.progress.pack(fill="x", padx=20, pady=10)

    # ========================== UI dinâmica ==========================
    def _update_mode_ui(self):
        if self.cut_mode.get() == "interval":
            if not self.interval_entry.winfo_ismapped():
                self.interval_entry.pack(fill="x", padx=12, pady=(4, 0))
        else:
            if self.interval_entry.winfo_ismapped():
                self.interval_entry.pack_forget()

    def set_ui_state(self, running: bool):
        state = "disabled" if running else "normal"

        for widget in [
            self.video_selector.entry, self.video_selector.button,
            self.output_selector.entry, self.output_selector.button
        ]:
            widget.configure(state=state)

        for rb in self.mode_radios + self.profile_radios:
            rb.configure(state=state)

        self.interval_entry.entry.configure(state=state)

    # ========================== Controle ==========================
    def toggle_start(self):
        if self.running:
            self.stop_process()
        else:
            self.start_process()

    def start_process(self):
        video = self.video_selector.get()
        output = self.output_selector.get()

        if not os.path.isfile(video) or not os.path.isdir(output):
            self.log.write("❌ Caminhos inválidos", color="red")
            return

        self.progress.update(0)
        self.status_label.configure(text="Iniciando...")

        self.running = True
        self.set_ui_state(True)
        self.start_btn.configure(text="Parar", fg_color="#dc2626")

        cfg = PROFILES[self.profile.get()].copy()
        cfg["ENABLE_PREVIEW"] = ENABLE_PREVIEW

        scene_mode = self.cut_mode.get() == "scene"
        if not scene_mode:
            try:
                cfg["FIXED_INTERVAL"] = float(self.interval_entry.get())
            except ValueError:
                self.log.write("❌ Intervalo inválido", color="red")
                self.reset_ui()
                return

        self.engine = SceneEngine(video, output, cfg, self.progress_callback)

        threading.Thread(
            target=self.run_engine,
            args=(scene_mode,),
            daemon=True
        ).start()

    def stop_process(self):
        self.running = False
        if self.engine:
            self.engine.stop()

        self.clear_preview()
        self.status_label.configure(text="Processo interrompido")
        self.log.write("⛔ Processo interrompido", color="#facc15")
        self.reset_ui()

    def run_engine(self, scene_mode):
        result = self.engine.run(scene_mode=scene_mode)
        if result:
            self.after(0, self.finish_process)
        else:
            self.after(0, self.reset_ui)

    # ========================== CALLBACK ==========================
    def progress_callback(self, msg=None, pct=None, img=None):
        if not self.running:
            return

        if msg is not None:
            self.after(0, lambda m=msg: self.status_label.configure(text=m))

        if pct is not None:
            self.after(0, lambda v=pct: self.progress.update(v / 100))

        if ENABLE_PREVIEW and isinstance(img, Image.Image):
            now = time.time()
            if now - self._last_preview_time >= self._preview_min_interval:
                self._last_preview_time = now
                self.after(0, lambda i=img: self.preview_frame.update_image(i))

    # ========================== Finalização ==========================
    def finish_process(self):
        self.running = False
        self.clear_preview()
        self.progress.update(1.0, finished=True)
        self.status_label.configure(text="Finalizado com sucesso")
        self.log.write("✅ Finalizado com sucesso", color="#22c55e")
        self.reset_ui(finished=True)

    def reset_ui(self, finished=False):
        self.running = False
        if self.engine:
            self.engine.stop()
            self.engine = None

        self.set_ui_state(False)
        self.start_btn.configure(text="Iniciar", fg_color="#3b82f6")

        if not finished:
            self.progress.update(0)
            self.status_label.configure(text="Aguardando início...")

    # ========================== Preview ==========================
    def clear_preview(self):
        self.preview_frame.clear()


if __name__ == "__main__":
    app = SceneCutterApp()
    app.mainloop()

import os
import threading
import time
import customtkinter as ctk
from PIL import Image

from core.engine import SceneEngine
from core.preview_ffmpeg import FFmpegPreview
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
PREVIEW_LIMIT = 5  # Quantas imagens mostrar durante o processo
PREVIEW_INTERVAL = 0.15  # Intervalo mínimo entre previews

PROFILES = {
    "menos_cortes": {"label": "Menos cortes", "THRESHOLD": 45.0, "MIN_SCENE_LEN_FRAMES": 10, "DOWNSCALE": 4, "MIN_FINAL_DURATION": 5.5},
    "normal": {"label": "Normal", "THRESHOLD": 28.0, "MIN_SCENE_LEN_FRAMES": 4, "DOWNSCALE": 3, "MIN_FINAL_DURATION": 1.8},
    "mais_cortes": {"label": "Mais cortes", "THRESHOLD": 18.0, "MIN_SCENE_LEN_FRAMES": 2, "DOWNSCALE": 2, "MIN_FINAL_DURATION": 0.9},
}

# ========================== App ==========================
class SceneCutterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Scene Cutter Pro")
        self.geometry("1000x650")

        self.engine = None
        self.previewer = None
        self.running = False

        self._build_ui()

    # ========================== UI ==========================
    def _build_ui(self):
        self.left = ctk.CTkFrame(self, width=300)
        self.left.pack(side="left", fill="y", padx=10, pady=10)
        self.right = ctk.CTkFrame(self)
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
        for text, value in [("Detecção de cenas", "scene"), ("A cada X segundos", "interval")]:
            rb = ctk.CTkRadioButton(row, text=text, variable=self.cut_mode, value=value, command=self._update_mode_ui)
            rb.pack(side="left", padx=6)
            self.mode_radios.append(rb)
        self.interval_entry = LabeledEntry(mode, "Intervalo (segundos)", placeholder="Ex: 10")

        profile = Section(self.left, "Perfil de detecção")
        profile.pack(fill="x", padx=10, pady=8)
        self.profile = ctk.StringVar(value="normal")
        self.profile_radios = []
        row = ctk.CTkFrame(profile, fg_color="transparent")
        row.pack(fill="x", padx=12)
        for key, cfg in PROFILES.items():
            rb = ctk.CTkRadioButton(row, text=cfg["label"], variable=self.profile, value=key)
            rb.pack(side="left", padx=6)
            self.profile_radios.append(rb)

        self.start_btn = ctk.CTkButton(self.left, text="Iniciar", command=self.toggle_start)
        self.start_btn.pack(pady=20)

        self.log = LogBox(self.left, height=220)
        self.log.pack(fill="x", padx=10, pady=10)

    def _build_preview(self):
        section = Section(self.right, "Processo")
        section.pack(fill="both", expand=True)
        self.preview_frame = PreviewFrame(section)
        self.preview_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.progress = ProgressBar(self.right)
        self.progress.pack(fill="x", padx=20, pady=10)

    def _update_mode_ui(self):
        if self.cut_mode.get() == "interval":
            if not self.interval_entry.winfo_ismapped():
                self.interval_entry.pack(fill="x", padx=12, pady=(4, 0))
        else:
            if self.interval_entry.winfo_ismapped():
                self.interval_entry.pack_forget()

    # ========================== Controle ==========================
    def toggle_start(self):
        try:
            if self.running:
                self.stop_process()
            else:
                self.start_process()
        except Exception as e:
            print("Erro toggle_start:", e)

    def start_process(self):
        video = self.video_selector.get()
        output = self.output_selector.get()
        if not os.path.isfile(video) or not os.path.isdir(output):
            self.log.write_message("❌ Caminhos inválidos", color="red")
            return

        self.progress.update(0)
        self.log.write_message("▶ Iniciando processamento...")

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
                self.log.write_message("❌ Intervalo inválido", color="red")
                self.reset_ui()
                return

        self.engine = SceneEngine(video, output, cfg, self.progress_callback)
        if ENABLE_PREVIEW:
            self.previewer = FFmpegPreview(video, min_interval=PREVIEW_INTERVAL)

        threading.Thread(target=self.run_engine, args=(scene_mode,), daemon=True).start()

    def stop_process(self):
        self.running = False
        try:
            if self.engine:
                self.engine.stop()
            if self.previewer:
                self.previewer.release()
            self.clear_preview()
            self.log.clear_status()
            self.log.write_message("⛔ Processo interrompido", color="#facc15")
        except Exception as e:
            print("Erro stop_process:", e)
        finally:
            self.reset_ui()

    def run_engine(self, scene_mode):
        try:
            result = self.engine.run(scene_mode=scene_mode)
            self.after(0, self.finish_process if result else self.reset_ui)
        except Exception as e:
            print("Erro run_engine:", e)
            self.after(0, self.reset_ui)

    # ========================== CALLBACK ==========================
    def progress_callback(self, msg=None, pct=None, idx=None, sec=None, status=None, **_):
        if not self.running:
            return

        # Atualiza linhas fixas do LogBox
        if status:
            try:
                self.after(0, lambda s=status: self.log.write_status(
                    detectadas=s.get("detectadas"),
                    cortadas=s.get("cortadas"),
                    eta=s.get("eta"),
                    corrido=s.get("corrido")
                ))
            except Exception as e:
                print("Erro atualizar status:", e)

        # Atualiza barra de progresso
        if pct is not None:
            self.after(0, lambda v=pct: self.progress.update(v / 100))

        # Atualiza preview limitado
        if sec is not None and self.previewer:
            try:
                img = self.previewer.get_frame_at(sec)
                if img:
                    self.after(0, lambda i=img: self.preview_frame.update_image(i))
            except Exception as e:
                print("Erro atualizar preview:", e)

    # ========================== Finalização ==========================
    def finish_process(self):
        self.running = False
        if self.previewer:
            self.previewer.release()
        self.clear_preview()
        self.log.clear_status()
        self.progress.update(1.0)

        elapsed = self.engine.total_time() if self.engine else "--:--"
        self.log.write_message(f"✅ Processo finalizado em {elapsed}", color="#22c55e")

        self.reset_ui(finished=True)

    def reset_ui(self, finished=False):
        self.running = False
        try:
            if self.engine:
                self.engine.stop()
                self.engine = None
            if self.previewer:
                self.previewer.release()
                self.previewer = None
        except Exception:
            pass
        self.set_ui_state(False)
        self.start_btn.configure(text="Iniciar", fg_color="#3b82f6")
        if not finished:
            self.progress.update(0)

    # ========================== UI Helpers ==========================
    def set_ui_state(self, running: bool):
        state = "disabled" if running else "normal"
        for widget in [self.video_selector.entry, self.video_selector.button,
                       self.output_selector.entry, self.output_selector.button]:
            widget.configure(state=state)
        for rb in self.mode_radios + self.profile_radios:
            rb.configure(state=state)
        self.interval_entry.entry.configure(state=state)

    def clear_preview(self):
        self.preview_frame.clear()


if __name__ == "__main__":
    app = SceneCutterApp()
    app.mainloop()

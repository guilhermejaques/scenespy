import os
import subprocess
import threading
import time
import datetime
import customtkinter as ctk
from PIL import Image
import av

# ======================= Configurações =======================
PROFILES = {
    "menos_cortes": {"label": "Menos cortes", "THRESHOLD": 45.0, "MIN_FINAL_DURATION": 5.5},
    "normal": {"label": "Normal", "THRESHOLD": 28.0, "MIN_FINAL_DURATION": 1.8},
    "mais_cortes": {"label": "Mais cortes", "THRESHOLD": 18.0, "MIN_FINAL_DURATION": 0.9},
}

ACCEL_OPTIONS = ["cpu", "nvidia", "amd", "intel"]
ENABLE_PREVIEW_DEFAULT = True
PREVIEW_INTERVAL = 0.15
PREVIEW_FPS = 1  # apenas 1 frame a cada N segundos para preview

# ======================= Widgets =======================
class Section(ctk.CTkFrame):
    def __init__(self, master, title, **kwargs):
        super().__init__(master, fg_color="#16161e", corner_radius=10, **kwargs)
        ctk.CTkLabel(self, text=title, font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=12, pady=(8,4))

class LabeledEntry(ctk.CTkFrame):
    def __init__(self, master, label, placeholder="", width=160):
        super().__init__(master, fg_color="transparent")
        ctk.CTkLabel(self, text=label, font=("Segoe UI", 12)).pack(anchor="w")
        self.entry = ctk.CTkEntry(self, placeholder_text=placeholder, width=width)
        self.entry.pack(pady=(2,8), fill="x")
    def get(self):
        return self.entry.get()

class LogBox(ctk.CTkTextbox):
    def __init__(self, master, height=140):
        super().__init__(master, height=height, fg_color="#0f0f14", corner_radius=8)
        self.configure(state="disabled", font=("Segoe UI", 11))
        self.status_lines = [""] * 4
        self._init_status_lines()
    def write_status(self, detectadas=None, cortadas=None, eta=None, corrido=None):
        self.configure(state="normal")
        if detectadas is not None:
            self.status_lines[0] = f"Analisando cenas: {detectadas}"
        if cortadas is not None:
            self.status_lines[1] = f"Cenas cortadas: {cortadas}"
        if eta is not None:
            self.status_lines[2] = f"Tempo estimado: {eta}"
        if corrido is not None:
            self.status_lines[3] = f"Tempo corrido: {corrido}"
        self.delete("1.0","end")
        for line in self.status_lines:
            self.insert("end", line+"\n")
        self.see("end")
        self.configure(state="disabled")
    def write_message(self, text, color=None):
        self.configure(state="normal")
        self.insert("end", text + "\n")
        if color:
            tag = f"msg_{color}"
            self.tag_add(tag, "end-2l", "end-1l")
            self.tag_config(tag, foreground=color)
        self.see("end")
        self.configure(state="disabled")
    def clear_status(self):
        self.configure(state="normal")
        self.status_lines = [""] * 4
        self.delete("1.0","end")
        for line in self.status_lines:
            self.insert("end", line+"\n")
        self.configure(state="disabled")
    def _init_status_lines(self):
        self.configure(state="normal")
        for _ in self.status_lines:
            self.insert("end","\n")
        self.configure(state="disabled")

class ProgressBar(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.bar = ctk.CTkProgressBar(self)
        self.bar.pack(fill="x", pady=4)
        self.bar.set(0)
        self.label = ctk.CTkLabel(self, text="0%", font=("Segoe UI", 11))
        self.label.pack(anchor="e")
    def update(self,value):
        self.bar.set(value)
        self.label.configure(text=f"{int(value*100)}%")

class PreviewFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="#0f0f14", corner_radius=10)
        self.info_label = ctk.CTkLabel(self, text="", font=("Segoe UI",10))
        self.info_label.pack(anchor="n", pady=4)
        self.label = ctk.CTkLabel(self, text="")
        self.label.pack(expand=True)
        self._img_ref = None
    def update_image(self,image):
        self._img_ref = ctk.CTkImage(light_image=image,size=image.size)
        self.label.configure(image=self._img_ref)
    def update_info(self,text):
        self.info_label.configure(text=text)
    def clear_image(self):
        self.label.configure(image=None)
        self._img_ref = None
    def clear_all(self):
        self.clear_image()
        self.info_label.configure(text="")


class FileSelector(ctk.CTkFrame):
    def __init__(self, master, label="Arquivo", width=420):
        super().__init__(master, fg_color="transparent")
        ctk.CTkLabel(self, text=label, font=("Segoe UI", 12)).pack(anchor="w")
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", pady=(4,8))
        self.entry = ctk.CTkEntry(row, width=width)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.configure(state="disabled")
        self.button = ctk.CTkButton(row, text="Selecionar", width=100, command=self.select)
        self.button.pack(side="right", padx=(6,0))
    def select(self):
        import tkinter.filedialog as fd
        path = fd.askopenfilename()
        if path:
            self.entry.configure(state="normal")
            self.entry.delete(0,"end")
            self.entry.insert(0,path)
            self.entry.configure(state="disabled")
    def get(self):
        return self.entry.get()

class DirectorySelector(FileSelector):
    def select(self):
        import tkinter.filedialog as fd
        path = fd.askdirectory()
        if path:
            self.entry.configure(state="normal")
            self.entry.delete(0,"end")
            self.entry.insert(0,path)
            self.entry.configure(state="disabled")

# ======================= Engine =======================
class SceneEngine:
    def __init__(self, video, output, cfg, logbox=None, progressbar=None, previewer=None, preview_enabled=True):
        self.video = video
        self.output = output
        self.cfg = cfg
        self.log = logbox
        self.progress = progressbar
        self.previewer = previewer
        self.preview_enabled = preview_enabled
        self._stop = False
        self.detected = 0
        self.total = 0
        self.done = 0
        self._start_time = None
        self._end_time = None
        self._video_info_shown = False

    def stop(self):
        self._stop = True

    def total_time(self):
        if not self._start_time: return "--:--"
        end = self._end_time or time.time()
        elapsed = int(end - self._start_time)
        m,s = divmod(elapsed,60)
        h,m = divmod(m,60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def run(self, scene_mode=True):
        self._start_time = time.time()
        if self.log:
            self.log.write_message("▶ Iniciando processo...")
            self.log.write_message("🔍 Analisando vídeo...")

        # Mostra infos do vídeo no preview
        if self.previewer and not self._video_info_shown:
            info_text = self._get_video_info_text()
            self.previewer.update_info(info_text)
            self._video_info_shown = True

        scenes = self._detect_scenes_progressive() if scene_mode else self._fixed_interval()
        if not scenes or self._stop: return False
        if self.log:
            self.log.write_message(f"🎬 Cenas detectadas: {len(scenes)}")
        self._cut_scenes(scenes)
        self._end_time = time.time()
        return not self._stop

    def _get_video_info_text(self):
        cmd = ["ffprobe","-v","error","-select_streams","v:0","-show_entries",
               "stream=width,height,r_frame_rate,bit_rate","-of","default=noprint_wrappers=1:nokey=1", self.video]
        try:
            out = subprocess.check_output(cmd).decode().splitlines()
            width,height,fps,bitrate = out
            num,den = fps.split('/')
            fps_float = round(int(num)/int(den),2)
            return f"{width}x{height} | FPS: {fps_float} | Bitrate: {int(bitrate)/1000:.0f} kbps"
        except Exception:
            return "Info vídeo indisponível"

    def _detect_scenes_progressive(self):
        container = av.open(self.video)
        stream = container.streams.video[0]
        min_dur = self.cfg["MIN_FINAL_DURATION"]
        scenes=[]
        frame_idx=0
        for packet in container.demux(stream):
            if self._stop: break
            for frame in packet.decode():
                frame_idx+=1
                if frame_idx % 10 == 0:
                    scenes.append(frame.time)
                    if self.log:
                        self.log.write_status(detectadas=len(scenes), cortadas=self.done)
                    if self.previewer and self.preview_enabled and frame_idx % int(PREVIEW_FPS/ PREVIEW_INTERVAL)==0:
                        img = frame.to_image().resize((420, int(420*frame.height/frame.width)))
                        self.previewer.update_image(img)
        # Agrupa cenas
        result=[]
        last=0
        for t in scenes:
            if t-last>=min_dur:
                result.append((last,t))
                last=t
        self.detected=len(result)
        return result

    def _fixed_interval(self):
        interval=self.cfg.get("FIXED_INTERVAL",10)
        cmd=["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1", self.video]
        duration=float(subprocess.check_output(cmd).decode().strip())
        scenes,t=[],0.0
        while t<duration:
            scenes.append((t,min(t+interval,duration)))
            t+=interval
        self.detected=len(scenes)
        return scenes

    def _cut_scenes(self, scenes):
        outdir=os.path.join(self.output, datetime.datetime.now().strftime("scenes_%Y%m%d_%H%M%S"))
        os.makedirs(outdir, exist_ok=True)
        self.total, self.done=len(scenes),0
        accel=self.cfg.get("ACCEL","cpu")
        for idx,(start,end) in enumerate(scenes,1):
            if self._stop: break

            # --- PREVIEW DURANTE CORTE (miniatura) ---
            if self.previewer and self.preview_enabled:
                mid_time = (start + end) / 2
                thumb = self._generate_thumbnail(mid_time)
                if thumb:
                    self.previewer.update_image(thumb)

            cmd=["ffmpeg","-y","-ss",f"{start:.3f}","-i",self.video,"-t",f"{end-start:.3f}"]
            if accel=="nvidia": cmd+=["-c:v","h264_nvenc"]
            elif accel=="amd": cmd+=["-c:v","h264_amf"]
            elif accel=="intel": cmd+=["-c:v","h264_qsv"]
            else: cmd+=["-c:v","libx264","-preset","veryfast"]
            cmd+=["-crf","23","-c:a","copy",os.path.join(outdir,f"scene_{idx:03d}.mp4")]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.done+=1
            if self.log:
                self.log.write_status(detectadas=self.detected, cortadas=self.done)
            if self.progress:
                self.progress.update(self.done/self.total)



    def _generate_thumbnail(self, timestamp):
        try:
            container = av.open(self.video)
            stream = container.streams.video[0]
            container.seek(int(timestamp * av.time_base))

            for frame in container.decode(video=0):
                img = frame.to_image()
                img = img.resize((420, int(420 * frame.height / frame.width)))
                return img
        except Exception:
            return None


# ======================= App =======================
class SceneCutterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Scene Cutter Pro")
        self.geometry("1000x650")
        self.engine = None
        self.running = False
        self.preview_enabled = ENABLE_PREVIEW_DEFAULT
        self._build_ui()

    def _build_ui(self):
        # Lateral esquerda
        self.left = ctk.CTkFrame(self,width=300)
        self.left.pack(side="left", fill="y", padx=10, pady=10)
        files = Section(self.left,"Arquivos")
        files.pack(fill="x", padx=10, pady=8)
        self.video_selector=FileSelector(files,"Vídeo de origem")
        self.video_selector.pack(fill="x", padx=12)
        self.output_selector=DirectorySelector(files)
        self.output_selector.pack(fill="x", padx=12)

        mode = Section(self.left, "Modo de corte")
        mode.pack(fill="x", padx=10, pady=8)

        self.cut_mode = ctk.StringVar(value="scene")
        self.cut_mode.trace_add("write", self._on_cut_mode_change)

        row = ctk.CTkFrame(mode, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=4)

        self.mode_radios = []

        # Radio: Detecção de cenas (primeiro)
        rb_scene = ctk.CTkRadioButton(
            row,
            text="Detecção de cenas",
            variable=self.cut_mode,
            value="scene"
        )
        rb_scene.pack(side="left", padx=(0, 16))
        self.mode_radios.append(rb_scene)

        # Radio: A cada X segundos
        rb_interval = ctk.CTkRadioButton(
            row,
            text="A cada X segundos",
            variable=self.cut_mode,
            value="interval"
        )
        rb_interval.pack(side="left")
        self.mode_radios.append(rb_interval)

        # Entry pequena ao lado (≈ 6 dígitos)
        self.interval_entry = ctk.CTkEntry(
            row,
            width=80,
            placeholder_text="seg"
        )

        profile=Section(self.left,"Perfil de detecção")
        profile.pack(fill="x", padx=10, pady=8)
        self.profile=ctk.StringVar(value="normal")
        row=ctk.CTkFrame(profile, fg_color="transparent")
        row.pack(fill="x", padx=12)
        self.profile_radios=[]
        for key,cfg in PROFILES.items():
            rb=ctk.CTkRadioButton(row,text=cfg["label"],variable=self.profile,value=key)
            rb.pack(side="left", padx=6)
            self.profile_radios.append(rb)

        accel_section=Section(self.left,"Aceleração de hardware")
        accel_section.pack(fill="x", padx=10, pady=8)
        self.accel=ctk.StringVar(value="cpu")
        row=ctk.CTkFrame(accel_section, fg_color="transparent")
        row.pack(fill="x", padx=12)
        self.accel_radios = []

        for val in ACCEL_OPTIONS:
            rb = ctk.CTkRadioButton(
                row,
                text=val.upper(),
                variable=self.accel,
                value=val
            )
            rb.pack(side="left", padx=6)
            self.accel_radios.append(rb)

        self.start_btn=ctk.CTkButton(self.left,text="Iniciar",command=self.toggle_start)
        self.start_btn.pack(pady=20)
        self.log=LogBox(self.left,height=220)
        self.log.pack(fill="x", padx=10, pady=10)

        # Preview direita
        self.right=ctk.CTkFrame(self)
        self.right.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        section=Section(self.right,"Preview")
        section.pack(fill="x")
        # switch preview
        self.preview_switch = ctk.CTkSwitch(
            section,
            text="Preview",
            command=self.toggle_preview
        )
        self.preview_switch.pack(anchor="e", padx=10)

        # garante que o preview inicia ligado visualmente
        if self.preview_enabled:
            self.preview_switch.select()
            self.toggle_preview()
        self.preview_frame=PreviewFrame(self.right)
        self.preview_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.progress=ProgressBar(self.right)
        self.progress.pack(fill="x", padx=20, pady=10)
        self._on_cut_mode_change()

    def toggle_preview(self):
        self.preview_enabled=self.preview_switch.get()
        if not self.preview_enabled:
            self.preview_frame.clear_image()

    def toggle_start(self):
        if self.running:
            self.stop_process()
        else:
            self.start_process()

    def start_process(self):
        video=self.video_selector.get()
        output=self.output_selector.get()
        if not os.path.isfile(video) or not os.path.isdir(output):
            self.log.write_message("❌ Caminhos inválidos", color="red")
            return
        self.progress.update(0)
        self.running=True
        self.set_ui_state(True)
        self.start_btn.configure(text="Parar", fg_color="#dc2626")
        cfg=PROFILES[self.profile.get()].copy()
        cfg["ACCEL"]=self.accel.get()
        scene_mode=self.cut_mode.get()=="scene"
        if not scene_mode:
            try:
                cfg["FIXED_INTERVAL"]=float(self.interval_entry.get())
            except ValueError:
                self.log.write_message("❌ Intervalo inválido", color="red")
                self.reset_ui()
                return
        self.engine=SceneEngine(video,output,cfg,logbox=self.log,progressbar=self.progress,
                                previewer=self.preview_frame,preview_enabled=self.preview_enabled)
        threading.Thread(target=self.run_engine,args=(scene_mode,),daemon=True).start()

    def stop_process(self):
        self.running=False
        if self.engine: self.engine.stop()
        self.preview_frame.clear_all()
        self.log.clear_status()
        self.log.write_message("⛔ Processo interrompido",color="#facc15")
        self.reset_ui()

    def run_engine(self, scene_mode):
        result=False
        try:
            result=self.engine.run(scene_mode=scene_mode)
        except Exception as e:
            print("Erro:",e)
        finally:
            self.after(0,self.reset_ui if not result else lambda: self.reset_ui(finished=True))

    def reset_ui(self, finished=False):
        self.running=False
        self.start_btn.configure(text="Iniciar", fg_color="#4ade80")
        self.set_ui_state(False)
        if finished:
            self.log.write_message(f"✅ Processo finalizado em {self.engine.total_time()}",color="#22c55e")
            self.progress.update(1.0)

    def set_ui_state(self,disabled):
        state="disabled" if disabled else "normal"
        for widget in [
            self.video_selector.button,
            self.output_selector.button,
            *self.mode_radios,
            *self.profile_radios,
            *self.accel_radios
        ]:
            widget.configure(state=state)

        if self.cut_mode.get() == "interval":
            self.interval_entry.configure(state=state)
        self.preview_switch.configure(state="disabled" if self.running else "normal")

    def _on_cut_mode_change(self, *args):
        if self.cut_mode.get() == "interval":
            self.interval_entry.pack(side="left", padx=(8, 0))
        else:
            self.interval_entry.pack_forget()


# ======================= Main =======================
if __name__=="__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    app=SceneCutterApp()
    app.mainloop()

import tkinter as tk

from .shared import *

class Section(ctk.CTkFrame):
    """Container with a title used to group related controls."""
    def __init__(self, master, title, **kwargs):
        super().__init__(master, fg_color=BG_CARD, border_width=1,
                         border_color=BORDER_SOFT2, corner_radius=0, **kwargs)
        self.title_label = ctk.CTkLabel(self, text=title, font=ui_font(14, "bold"))
        self.title_label.pack(anchor="w", padx=12, pady=(8, 4))


class ToolTip:
    """Small delayed hover popup for explaining compact controls."""
    def __init__(self, widget, text, delay=450):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._after_id = None
        self._window = None

        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Destroy>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after_id:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        self._after_id = None
        if self._window or not self.text:
            return
        try:
            if not self.widget.winfo_exists():
                return
            x = self.widget.winfo_rootx() + 16
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
            self._window = tk.Toplevel(self.widget)
            self._window.wm_overrideredirect(True)
            self._window.wm_attributes("-topmost", True)
            self._window.configure(bg=BORDER_SOFT2)
            label = tk.Label(
                self._window, text=self.text, justify="left",
                bg=BG_MAIN, fg=TEXT_MAIN, bd=0, padx=9, pady=6,
                font=ui_font(10), wraplength=260
            )
            label.pack(padx=1, pady=1)
            self._window.wm_geometry(f"+{x}+{y}")
        except Exception:
            self._window = None

    def _hide(self, _event=None):
        self._cancel()
        if self._window:
            try:
                self._window.destroy()
            except Exception:
                pass
            self._window = None


class LabeledEntry(ctk.CTkFrame):
    """Text entry paired with a label."""
    def __init__(self, master, label, placeholder="", width=160):
        super().__init__(master, fg_color="transparent")
        ctk.CTkLabel(self, text=label, font=ui_font(12)).pack(anchor="w")
        self.entry = ctk.CTkEntry(
            width=width, corner_radius=15, fg_color=BG_MAIN,
            border_width=1, border_color=BORDER_SOFT,
            text_color=TEXT_MAIN, placeholder_text_color=TEXT_MUTED,
            placeholder_text=placeholder
        )
        self.entry.pack(pady=(2, 8), fill="x")

    def get(self):
        return self.entry.get()


class LogBox(ctk.CTkTextbox):
    """Status console for processing progress and user-facing messages."""
    def __init__(self, master, height=140):
        super().__init__(master, height=height, fg_color=BG_MAIN,
                         corner_radius=15, border_color=BORDER_SOFT2, border_width=1)
        self.configure(state="disabled", font=ui_font(12), wrap="char")
        self.pack_propagate(False)
        self.status_lines = []
        self.initialized = False
        self.status_labels = ("Scenes detected", "Scenes cut")
        self._eta_label = "Estimated time"

    def _terminal_wrap_width(self):
        width_px = self.winfo_width()
        if width_px <= 1:
            width_px = 640
        return max(42, min(96, int(width_px / 8)))

    def _format_terminal_message(self, text, max_chars=900):
        cleaned = " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars].rstrip() + "..."
        return "\n".join(textwrap.wrap(
            cleaned,
            width=self._terminal_wrap_width(),
            break_long_words=True,
            break_on_hyphens=False,
        )) or "Unknown error"

    def _ensure_initialized(self):
        if self.initialized:
            return
        self.delete("1.0", "end")
        self.status_lines = self._status_lines()
        for line in self.status_lines:
            self.insert("end", line + "\n")
        self.initialized = True

    def set_mode(self, mode):
        self.status_labels = {
            "scene": ("Scenes detected", "Scenes cut"),
            "interval": ("Segments total", "Segments cut"),
            "faces": ("Faces detected", "Faces saved"),
        }.get(mode, ("Items detected", "Items done"))

    def write_status(self, detected=None, cut=None, eta=None):
        lines = self._status_lines(detected=detected, cut=cut, eta=eta)
        self.status_lines = lines
        self.configure(state="normal")
        if not self.initialized:
            self.delete("1.0", "end")
            self.status_lines = lines
            for line in lines:
                self.insert("end", line + "\n")
            self.initialized = True
        else:
            for i, line in enumerate(lines):
                self.delete(f"{i + 1}.0", f"{i + 1}.end")
                self.insert(f"{i + 1}.0", line)
        self.configure(state="disabled")

    def _status_label_width(self):
        return max(len(self._eta_label), *(len(label) for label in self.status_labels))

    def _format_status_line(self, label, value):
        return f"{label:<{self._status_label_width()}} : {value}"

    def _status_lines(self, detected=None, cut=None, eta=None):
        detected_label, cut_label = self.status_labels
        return [
            self._format_status_line(detected_label, detected if detected is not None else "-"),
            self._format_status_line(cut_label, cut if cut is not None else "-"),
            self._format_status_line(self._eta_label, eta if eta is not None else "--:--"),
        ]

    def clear_status(self):
        self.status_lines = ["Processing..."]
        self.initialized = False
        self._render()

    def show_message(self, text):
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.insert("end", self._format_terminal_message(text) + "\n")
        self.configure(state="disabled")
        self.status_lines = [str(text or "")]
        self.initialized = False

    def append_message(self, text, kind="info"):
        self.configure(state="normal")
        self._ensure_initialized()
        tag = f"msg_{kind}"
        message = self._format_terminal_message(text)
        if self.index("end-1c") != "1.0":
            self.insert("end", "\n")
        start = self.index("end")
        self.insert("end", message)
        end = self.index("end")
        self.tag_add(tag, start, end)
        self.tag_config(tag, foreground=TEXT_MAIN)
        self.configure(state="disabled")

    def write_finished(self, text):
        self.configure(state="normal")
        self._ensure_initialized()

        current = self.get("3.0", "3.end").strip()
        if not current:
            current = self._format_status_line(self._eta_label, "--:--")
        self.delete("3.0", "3.end")
        self.insert("3.0", current + " ")
        start = self.index("3.end")
        self.insert("3.end", f"({text})")
        end = self.index("3.end")
        self.tag_add("finished", start, end)
        self.tag_config("finished", foreground="#22c55e")
        self.configure(state="disabled")

    def _render(self):
        self.configure(state="normal")
        self.delete("1.0", "end")
        for line in self.status_lines:
            self.insert("end", line + "\n")
        self.configure(state="disabled")


class ProgressBar(ctk.CTkFrame):
    """Progress bar with smoothed visual updates."""
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.bar = ctk.CTkProgressBar(self)
        self.bar.pack(fill="x", pady=4)
        self.bar.set(0)
        self._after_id = None
        self._enabled = True
        self.label = ctk.CTkLabel(self, text="0%", font=ui_font(11))
        self.label.pack(anchor="e")
        self._normal_color = self.bar.cget("progress_color")
        self._logical_value = 0.0
        self._visual_value = 0.0
        self._animating = False
        self._speed = 0.03

    def update(self, value):
        if not self._enabled:
            return
        value = max(0.0, min(1.0, value))
        if abs(value - self._logical_value) < 0.005 or value < self._logical_value:
            return
        if not self.winfo_exists():
            return
        self._logical_value = value
        if not self._animating:
            self._animating = True
            self._after_id = self.after(10, self._animate_step)

    def _animate_step(self):
        if self._logical_value - self._visual_value < 0.01:
            self._visual_value = self._logical_value
            self.bar.set(self._visual_value)
            self.label.configure(text=f"{int(self._visual_value * 100)}%")
            self._animating = False
            return
        delta = max((self._logical_value - self._visual_value) * self._speed, 0.004)
        self._visual_value += delta
        self.bar.set(self._visual_value)
        self.label.configure(text=f"{int(self._visual_value * 100)}%")
        self._after_id = self.after(16, self._animate_step)

    def mark_finished(self):
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        self._animating = False
        self._logical_value = 1.0
        self._visual_value = 1.0
        self.bar.configure(progress_color="#22c55e")
        self.bar.set(1.0)
        self.label.configure(text="100%")

    def reset(self):
        self._enabled = False
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        self._animating = False
        self._logical_value = 0.0
        self._visual_value = 0.0
        self.bar.configure(progress_color=self._normal_color)
        self.bar.set(0)
        self.label.configure(text="0%")
        self._enabled = True


class PreviewFrame(ctk.CTkFrame):
    """Video preview area with optional loading animation."""
    def __init__(self, master):
        super().__init__(master, fg_color=BG_MAIN, border_width=1,
                         border_color=BORDER_SOFT2, corner_radius=15)
        self.info_label = ctk.CTkLabel(self, text="", font=ui_font(10))
        self.info_label.pack(anchor="n", pady=4)
        self.label = ctk.CTkLabel(self, text="")
        self.label.pack(expand=True, anchor="center")
        self.loading_label = ctk.CTkLabel(self, text="", fg_color="transparent")
        self.loading_label.place_forget()
        self._img_ref = None
        self._enabled = True
        self._loading_frames = []
        self._loading_durations = []
        self._loading_index = 0
        self._loading_after_id = None
        self._loading_visible = False

    def set_enabled(self, enabled):
        self._enabled = bool(enabled)
        if not self._enabled:
            self.clear_image()

    def update_image(self, image):
        if not self._enabled or not image or not self.winfo_exists():
            return
        self.hide_loading()
        self._img_ref = ctk.CTkImage(light_image=image, size=image.size)
        self.label.configure(image=self._img_ref)

    def update_info(self, text):
        if not self._enabled:
            return
        self.info_label.configure(text=text)

    def _load_loading_frames(self):
        if self._loading_frames:
            return True
        if not os.path.exists(LOADING_GIF_FILE):
            return False
        try:
            with Image.open(LOADING_GIF_FILE) as gif:
                frame_count = max(1, getattr(gif, "n_frames", 1))
                stride = max(1, frame_count // 24)
                for index in range(0, frame_count, stride):
                    gif.seek(index)
                    duration = max(70, int(gif.info.get("duration", 80)) * stride)
                    img = gif.convert("RGBA")
                    img.thumbnail((34, 34), Image.Resampling.LANCZOS)
                    self._loading_frames.append(ctk.CTkImage(light_image=img.copy(), size=img.size))
                    self._loading_durations.append(duration)
        except Exception as e:
            print(f"[DEBUG] Failed to load loading animation: {e}")
            self._loading_frames = []
            self._loading_durations = []
        return bool(self._loading_frames)

    def show_loading(self):
        if self._loading_visible or not self.winfo_exists():
            return
        if not self._load_loading_frames():
            return
        self._loading_visible = True
        self._loading_index = 0
        self.loading_label.place(relx=0.5, rely=0.5, anchor="center")
        self.loading_label.lift()
        self._animate_loading()

    def hide_loading(self):
        self._loading_visible = False
        if self._loading_after_id:
            try:
                self.after_cancel(self._loading_after_id)
            except Exception:
                pass
            self._loading_after_id = None
        self.loading_label.configure(image=None)
        self.loading_label.place_forget()

    def _animate_loading(self):
        if not self._loading_visible or not self.winfo_exists() or not self._loading_frames:
            self.hide_loading()
            return
        frame = self._loading_frames[self._loading_index]
        duration = self._loading_durations[self._loading_index]
        self.loading_label.configure(image=frame)
        self.loading_label.lift()
        self._loading_index = (self._loading_index + 1) % len(self._loading_frames)
        self._loading_after_id = self.after(duration, self._animate_loading)

    def clear_image(self):
        self.label.configure(image=None)
        self._img_ref = None

    def clear_all(self):
        self.hide_loading()
        self.clear_image()
        self.info_label.configure(text="")


class FileSelector(ctk.CTkFrame):
    """File picker that supports one or more selected videos."""
    def __init__(self, master, label="File", width=400):
        super().__init__(master, fg_color="transparent")
        self.paths = []
        ctk.CTkLabel(self, text=label, font=ui_font(12)).pack(anchor="w")
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", pady=(4, 8))
        self.entry = ctk.CTkEntry(
            row, width=width, corner_radius=15, fg_color=BG_MAIN,
            border_width=1, border_color=BORDER_SOFT,
            text_color="#ededed", font=ui_font(11),
            placeholder_text_color=TEXT_MUTED
        )
        self.entry.pack(side="left")
        self.button = ctk.CTkButton(
            row, text="...", width=10, height=10, corner_radius=15,
            fg_color=BG_CARD, hover_color="#615f5f", border_width=1,
            border_color=BORDER_SOFT, text_color=TEXT_MUTED,
            command=self.select
        )
        self.button.pack(side="right", padx=(6, 0))
        self.entry.pack_propagate(False)
        self.button.pack_propagate(False)

    def select(self):
        paths = list(fd.askopenfilenames(
            filetypes=[("Video files", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v"),
                       ("All files", "*.*")]
        ))
        if paths:
            self.paths = paths
            self.entry.delete(0, "end")
            if len(paths) == 1:
                self.entry.insert(0, paths[0])
            else:
                self.entry.insert(0, f"{len(paths)} videos selected")

    def get(self):
        return self.entry.get()

    def get_paths(self):
        value = self.get().strip()
        if self.paths:
            if len(self.paths) > 1 and value == f"{len(self.paths)} videos selected":
                return list(self.paths)
            if len(self.paths) == 1 and value == self.paths[0]:
                return list(self.paths)
        return [value] if value else []


class DirectorySelector(FileSelector):
    """Folder picker for the output directory."""
    def select(self):
        path = fd.askdirectory()
        if path:
            self.entry.delete(0, "end")
            self.entry.insert(0, path)


class RadioGroup(ctk.CTkFrame):
    """Horizontal group of radio buttons bound to one variable."""
    def __init__(self, master, variable, options, columns=4, radio_width=120, height=32):
        super().__init__(master, fg_color="transparent", height=height)
        self.grid_propagate(False)
        self.radios = []
        for i, (label, value) in enumerate(options):
            rb = ctk.CTkRadioButton(
                self, text=label, variable=variable, value=value,
                width=radio_width, radiobutton_width=10, radiobutton_height=10,
                fg_color=ACCENT, border_color="#4b5563", hover_color="#6366f1",
                text_color=TEXT_MAIN, text_color_disabled=TEXT_MUTED,
                bg_color="transparent", font=ui_font(12)
            )
            rb.grid(row=0, column=i, padx=(0, 12), pady=0, sticky="w")
            self.radios.append(rb)

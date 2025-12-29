# ========================= gui/widgets.py =========================
import customtkinter as ctk

class Section(ctk.CTkFrame):
    def __init__(self, master, title, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="#16161e", corner_radius=10)
        ctk.CTkLabel(
            self, text=title, font=("Segoe UI", 14, "bold")
        ).pack(anchor="w", padx=12, pady=(8, 4))


class LabeledEntry(ctk.CTkFrame):
    def __init__(self, master, label, placeholder="", width=160):
        super().__init__(master, fg_color="transparent")
        ctk.CTkLabel(self, text=label, font=("Segoe UI", 12)).pack(anchor="w")
        self.entry = ctk.CTkEntry(self, placeholder_text=placeholder, width=width)
        self.entry.pack(pady=(2, 8), fill="x")

    def get(self):
        return self.entry.get()


class LogBox(ctk.CTkTextbox):
    def __init__(self, master, height=120):
        super().__init__(master, height=height, fg_color="#0f0f14", corner_radius=8)
        self.configure(state="disabled", font=("Segoe UI", 11))

    def write(self, text, color=None):
        self.configure(state="normal")
        if color:
            tag = f"c_{color}"
            self.insert("end", text + "\n", tag)
            self.tag_config(tag, foreground=color)
        else:
            self.insert("end", text + "\n")
        self.see("end")
        self.configure(state="disabled")


class ProgressBar(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.bar = ctk.CTkProgressBar(self)
        self.bar.pack(fill="x", pady=4)
        self.bar.set(0)
        self.label = ctk.CTkLabel(self, text="0%", font=("Segoe UI", 11))
        self.label.pack(anchor="e")

    def update(self, value, finished=False):
        self.bar.set(value)
        self.label.configure(text=f"{int(value * 100)}%")


class PreviewFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="#0f0f14", corner_radius=10)
        self.label = ctk.CTkLabel(self, text="")
        self.label.pack(expand=True)
        self._img_ref = None

    def update_image(self, image):
        self._img_ref = ctk.CTkImage(light_image=image, size=image.size)
        self.label.configure(image=self._img_ref)

    def clear(self):
        # NÃO limpa a imagem para evitar flicker
        pass


class FileSelector(ctk.CTkFrame):
    def __init__(self, master, label="Arquivo", width=420):
        super().__init__(master, fg_color="transparent")
        ctk.CTkLabel(self, text=label, font=("Segoe UI", 12)).pack(anchor="w")
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", pady=(4, 8))
        self.entry = ctk.CTkEntry(row, width=width)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.configure(state="disabled")
        self.button = ctk.CTkButton(row, text="Selecionar", width=100, command=self.select)
        self.button.pack(side="right", padx=(6, 0))

    def select(self):
        import tkinter.filedialog as fd
        path = fd.askopenfilename()
        if path:
            self.entry.configure(state="normal")
            self.entry.delete(0, "end")
            self.entry.insert(0, path)
            self.entry.configure(state="disabled")

    def get(self):
        return self.entry.get()


class DirectorySelector(ctk.CTkFrame):
    def __init__(self, master, label="Diretório de saída", width=420):
        super().__init__(master, fg_color="transparent")
        ctk.CTkLabel(self, text=label, font=("Segoe UI", 12)).pack(anchor="w")
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", pady=(4, 8))
        self.entry = ctk.CTkEntry(row, width=width)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.configure(state="disabled")
        self.button = ctk.CTkButton(row, text="Selecionar", width=100, command=self.select)
        self.button.pack(side="right", padx=(6, 0))

    def select(self):
        import tkinter.filedialog as fd
        path = fd.askdirectory()
        if path:
            self.entry.configure(state="normal")
            self.entry.delete(0, "end")
            self.entry.insert(0, path)
            self.entry.configure(state="disabled")

    def get(self):
        return self.entry.get()

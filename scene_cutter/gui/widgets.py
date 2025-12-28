import customtkinter as ctk

# ===========================
# Blocos e labels
# ===========================
class Section(ctk.CTkFrame):
    """Bloco visual com título"""
    def __init__(self, master, title, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="#16161e", corner_radius=10)
        self.title = ctk.CTkLabel(
            self,
            text=title,
            font=("Segoe UI", 14, "bold")
        )
        self.title.pack(anchor="w", padx=12, pady=(8, 4))


class LabeledEntry(ctk.CTkFrame):
    """Label + Entry padronizado"""
    def __init__(self, master, label, placeholder="", width=160):
        super().__init__(master, fg_color="transparent")
        ctk.CTkLabel(
            self,
            text=label,
            font=("Segoe UI", 12)
        ).pack(anchor="w")

        self.entry = ctk.CTkEntry(
            self,
            placeholder_text=placeholder,
            width=width
        )
        self.entry.pack(pady=(2, 8), fill="x")

    def get(self):
        return self.entry.get()


# ===========================
# Log e progresso
# ===========================
class LogBox(ctk.CTkTextbox):
    """Área de log estilo software profissional"""
    def __init__(self, master, height=120):
        super().__init__(
            master,
            height=height,
            fg_color="#0f0f14",
            corner_radius=8
        )
        self.configure(state="disabled", font=("Segoe UI", 11))

    def write(self, text, color=None):
        self.configure(state="normal")
        if color:
            tag = f"color_{color}"
            self.insert("end", text + "\n", (tag,))
            self.tag_config(tag, foreground=color)
        else:
            self.insert("end", text + "\n")
        self.see("end")
        self.configure(state="disabled")

    def clear(self):
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.configure(state="disabled")


class ProgressBar(ctk.CTkFrame):
    """Barra de progresso customizada (SEM BUG DE COR)"""
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")

        self.default_color = "#3b82f6"  # azul padrão
        self.finished_color = "#22c55e" # verde final

        self.bar = ctk.CTkProgressBar(
            self,
            progress_color=self.default_color
        )
        self.bar.pack(fill="x", pady=4)
        self.bar.set(0)

        self.label = ctk.CTkLabel(self, text="0%", font=("Segoe UI", 11))
        self.label.pack(anchor="e")

    def update(self, value, finished=False):
        self.bar.set(value)
        self.label.configure(text=f"{int(value * 100)}%")

        if finished:
            self.bar.configure(progress_color=self.finished_color)
        else:
            # 🔒 garante que NUNCA será None
            self.bar.configure(progress_color=self.default_color)


# ===========================
# Preview
# ===========================
class PreviewFrame(ctk.CTkFrame):
    """Frame de preview de vídeo"""
    def __init__(self, master):
        super().__init__(master, fg_color="#0f0f14", corner_radius=10)
        self.label = ctk.CTkLabel(self, text="")
        self.label.pack(expand=True)
        self._img_ref = None  # evita GC

    def update_image(self, image):
        import customtkinter as ctk
        self._img_ref = ctk.CTkImage(light_image=image, size=image.size)
        self.label.configure(image=self._img_ref)

    def clear(self):
        self.label.configure(image="")
        self._img_ref = None


# ===========================
# Seletores
# ===========================
class FileSelector(ctk.CTkFrame):
    """Campo de seleção de arquivo"""
    def __init__(self, master, label="Arquivo", filetypes=None, width=420):
        super().__init__(master, fg_color="transparent")

        self.filetypes = filetypes or [
            ("Vídeos", "*.mp4 *.mkv *.avi *.mov")
        ]

        ctk.CTkLabel(self, text=label, font=("Segoe UI", 12)).pack(anchor="w")

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", pady=(4, 8))

        self.entry = ctk.CTkEntry(row, width=width)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.configure(state="disabled")

        self.button = ctk.CTkButton(
            row,
            text="Selecionar",
            width=100,
            command=self.select_file
        )
        self.button.pack(side="right", padx=(6, 0))

    def select_file(self):
        import tkinter.filedialog as fd
        path = fd.askopenfilename(filetypes=self.filetypes)
        if path:
            self.entry.configure(state="normal")
            self.entry.delete(0, "end")
            self.entry.insert(0, path)
            self.entry.configure(state="disabled")

    def get(self):
        return self.entry.get()


class DirectorySelector(ctk.CTkFrame):
    """Campo de seleção de diretório"""
    def __init__(self, master, label="Diretório de saída", width=420):
        super().__init__(master, fg_color="transparent")

        ctk.CTkLabel(self, text=label, font=("Segoe UI", 12)).pack(anchor="w")

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", pady=(4, 8))

        self.entry = ctk.CTkEntry(row, width=width)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.configure(state="disabled")

        self.button = ctk.CTkButton(
            row,
            text="Selecionar",
            width=100,
            command=self.select_directory
        )
        self.button.pack(side="right", padx=(6, 0))

    def select_directory(self):
        import tkinter.filedialog as fd
        path = fd.askdirectory()
        if path:
            self.entry.configure(state="normal")
            self.entry.delete(0, "end")
            self.entry.insert(0, path)
            self.entry.configure(state="disabled")

    def get(self):
        return self.entry.get()

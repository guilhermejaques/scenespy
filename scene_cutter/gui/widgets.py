import customtkinter as ctk


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
            self.status_lines[0] = f"Detectando cenas: {detectadas}"
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
        """Escreve mensagem adicional abaixo das 4 linhas de status."""
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
        self.label = ctk.CTkLabel(self, text="")
        self.label.pack(expand=True)
        self._img_ref = None
    def update_image(self,image):
        self._img_ref = ctk.CTkImage(light_image=image,size=image.size)
        self.label.configure(image=self._img_ref)
    def clear(self):
        self.label.configure(image=None)
        self._img_ref = None


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

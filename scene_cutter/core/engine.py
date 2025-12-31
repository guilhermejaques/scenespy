import os, subprocess, datetime, time
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

class SceneEngine:
    def __init__(self, video, output, cfg, logbox=None, progressbar=None, previewer=None):
        self.video = video
        self.output = output
        self.cfg = cfg
        self.log = logbox
        self.progress = progressbar
        self.previewer = previewer
        self._stop = False
        self.detected = 0
        self.total = 0
        self.done = 0
        self._start_time = None
        self._end_time = None

    def stop(self):
        self._stop = True

    def total_time(self):
        if not self._start_time:
            return "--:--"
        end = self._end_time or time.time()
        elapsed = int(end - self._start_time)
        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def run(self, scene_mode=True):
        self._start_time = time.time()
        if self.log:
            self.log.write_message("▶ Iniciando processo...")
            self.log.write_message("🔍 Analisando vídeo...")

        scenes = self._detect_scenes_progressive() if scene_mode else self._fixed_interval()
        if not scenes or self._stop:
            return False
        if self.log:
            self.log.write_message(f"🎬 Cenas detectadas: {len(scenes)}")
        self._cut_scenes(scenes)
        self._end_time = time.time()
        return not self._stop

    def _detect_scenes_progressive(self):
        video = open_video(self.video)
        video.downscale = self.cfg["DOWNSCALE"]
        sm = SceneManager()
        detector = ContentDetector(threshold=self.cfg["THRESHOLD"], min_scene_len=self.cfg["MIN_SCENE_LEN_FRAMES"])
        sm.add_detector(detector)
        for i, _ in enumerate(range(int(video.frame_count))):
            if self._stop:
                break
            detector.process_frame(video.frame_index)
            if self.log:
                self.log.write_status(detectadas=len(detector.get_scene_list()))
        scenes = detector.get_scene_list()
        # aplicar MIN_FINAL_DURATION
        min_dur = self.cfg["MIN_FINAL_DURATION"]
        result, buf_s, buf_e = [], None, None
        for s, e in scenes:
            start, end = s.get_seconds(), e.get_seconds()
            if buf_s is None: buf_s = start
            buf_e = end
            if buf_e-buf_s >= min_dur:
                result.append((buf_s, buf_e))
                buf_s = None
        if buf_s is not None: result.append((buf_s, buf_e))
        self.detected = len(result)
        return result

    def _fixed_interval(self):
        interval = self.cfg["FIXED_INTERVAL"]
        cmd = ["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1", self.video]
        duration = float(subprocess.check_output(cmd).decode().strip())
        scenes, t = [], 0.0
        while t < duration:
            scenes.append((t,min(t+interval,duration)))
            t += interval
        self.detected = len(scenes)
        return scenes

    def _cut_scenes(self, scenes):
        outdir = os.path.join(self.output, datetime.datetime.now().strftime("scenes_%Y%m%d_%H%M%S"))
        os.makedirs(outdir, exist_ok=True)
        self.total, self.done = len(scenes), 0
        for idx, (start, end) in enumerate(scenes, 1):
            if self._stop:
                break
            elapsed = int(time.time()-self._start_time)
            pct_done = (idx-1)/self.total
            eta_str = "--:--"
            if pct_done>0:
                total_est=elapsed/pct_done
                rem=max(0,int(total_est-elapsed))
                mh, ms = divmod(rem,60)
                hh, mh = divmod(mh,60)
                eta_str = f"{hh:02d}:{mh:02d}:{ms:02d}" if hh else f"{mh:02d}:{ms:02d}"
            if self.log:
                self.log.write_status(detectadas=self.detected, cortadas=idx-1, eta=eta_str)
            if self.progress:
                self.progress.update(pct_done)
            outfile = os.path.join(outdir,f"scene_{idx:03d}.mp4")
            subprocess.run(["ffmpeg","-y","-ss",f"{start:.3f}","-i",self.video,"-t",f"{end-start:.3f}",
                            "-c:v","libx264","-preset","veryfast","-crf","23","-c:a","copy",outfile],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.done += 1
            if self.progress:
                self.progress.update(self.done/self.total)
        if self.log:
            self.log.write_message(f"✅ Processo finalizado em {self.total_time()}", color="#22c55e")
        if self.progress:
            self.progress.update(1.0)

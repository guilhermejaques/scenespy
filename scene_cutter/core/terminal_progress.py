import sys
import time
import threading

class TerminalProgressReporter:
    def __init__(self):
        self.start_time = time.time()
        self._lock = threading.Lock()
        self._last_len = 0
        self._active = True

        # linha inicial limpa
        sys.stdout.write("\n")
        sys.stdout.flush()

    def update(self, percent: float, text: str = ""):
        if not self._active:
            return

        if percent is None:
            percent = 0.0

        percent = max(0.0, min(100.0, float(percent)))
        text = (text or "").strip()

        elapsed = time.time() - self.start_time

        eta = ""
        if percent > 0:
            total_est = elapsed / (percent / 100.0)
            remaining = max(0, int(total_est - elapsed))
            eta = f" | ETA {self._fmt(remaining)}"

        msg = f"{text} | {percent:6.2f}%{eta}"

        with self._lock:
            # sobrescreve a linha anterior
            sys.stdout.write("\r")
            sys.stdout.write(" " * max(self._last_len, len(msg)))
            sys.stdout.write("\r")
            sys.stdout.write(msg)
            sys.stdout.flush()
            self._last_len = len(msg)

    def finish(self, final_text: str = "Concluído"):
        with self._lock:
            if not self._active:
                return

            total = int(time.time() - self.start_time)
            total_fmt = self._fmt(total)

            sys.stdout.write("\r")
            sys.stdout.write(" " * self._last_len)
            sys.stdout.write("\r")
            sys.stdout.write(f"{final_text} | Tempo total: {total_fmt}\n")
            sys.stdout.flush()

            self._active = False

    @staticmethod
    def _fmt(seconds: int) -> str:
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

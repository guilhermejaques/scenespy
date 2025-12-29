import time
import sys
import threading


class TerminalProgressReporter:
    def __init__(self):
        self.start_time = time.time()
        self.last_len = 0
        self._lock = threading.Lock()

    def _fmt(self, sec):
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m:02d}:{s:02d}"

    def update(self, current=0, total=0, detected=None, phase=None):
        with self._lock:
            elapsed = time.time() - self.start_time

            parts = []
            if phase:
                parts.append(f"[{phase.upper()}]")
            if detected is not None:
                parts.append(f"Detected: {detected}")
            if total:
                parts.append(f"Progress: {current}/{total}")
            else:
                parts.append(f"Progress: {current}")
            parts.append(f"Time: {self._fmt(elapsed)}")

            msg = " | ".join(parts)

            pad = max(0, self.last_len - len(msg))
            sys.stdout.write("\r" + msg + (" " * pad))
            sys.stdout.flush()
            self.last_len = len(msg)

    def finish(self):
        with self._lock:
            sys.stdout.write("\n")
            sys.stdout.flush()

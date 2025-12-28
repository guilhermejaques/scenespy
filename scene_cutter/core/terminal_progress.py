import time
import sys


class TerminalProgressReporter:
    def __init__(self, update_interval=1.5):
        self.update_interval = update_interval
        self._last = 0
        self._start = time.time()

    def update(self, phase, current, total):
        now = time.time()
        if now - self._last < self.update_interval:
            return

        self._last = now

        elapsed = now - self._start
        avg = elapsed / max(current, 1)
        eta = avg * (total - current)

        pct = (current / total) * 100
        eta_min = eta / 60

        msg = (
            f"[{phase.upper():>4}] "
            f"{current}/{total} | "
            f"{pct:5.1f}% | "
            f"ETA ~{eta_min:4.1f} min"
        )

        sys.stdout.write("\r" + msg)
        sys.stdout.flush()

    def finish(self):
        sys.stdout.write("\n")
        sys.stdout.flush()

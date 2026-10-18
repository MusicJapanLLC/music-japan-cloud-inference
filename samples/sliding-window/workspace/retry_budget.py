"""An intentionally defective rolling-window limiter for agent evaluation."""

class Budget:
    def __init__(self, limit, window):
        self.limit = limit
        self.window = window
        self.timestamps = []

    def try_acquire(self, now):
        self.timestamps = [stamp for stamp in self.timestamps if now - stamp > self.window]
        if len(self.timestamps) <= self.limit:
            self.timestamps.append(now)
            return True
        return False

    def reset(self):
        self.timestamps.clear()

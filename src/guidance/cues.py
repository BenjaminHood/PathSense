"""Policy action id -> spoken words. The ONLY place that decides what is said.

WARNING: in the current sim, CUE_LEFT does heading -= 30 deg, which under the
contract's +y = left turns the walker RIGHT (and CUE_RIGHT turns left).
Until Track A fixes it, the words follow what the action actually DOES in sim.
After the fix, swap the two strings back and re-train/re-test.
"""
import time

# 0 STAY_SILENT, 1 CUE_LEFT, 2 CUE_RIGHT, 3 CUE_STOP, 4 CUE_STRAIGHT
PHRASES = {1: "bear right", 2: "bear left", 3: "stop", 4: "go straight"}


class Speaker:
    def __init__(self, min_gap_s=1.5):
        import pyttsx3
        self.engine = pyttsx3.init()
        self.min_gap_s, self.last = min_gap_s, 0.0

    def say(self, action: int):
        phrase = PHRASES.get(action)
        if phrase is None or time.time() - self.last < self.min_gap_s:
            return
        self.last = time.time()
        self.engine.say(phrase)
        self.engine.runAndWait()      # blocking; move to a thread if it costs latency

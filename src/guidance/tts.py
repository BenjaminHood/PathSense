"""Non-blocking speech. The control loop must never wait on audio.

Only the LATEST pending cue is kept: if the loop issues "left" then
"stop" while the engine is still talking, "left" is dropped. A stale
navigation instruction is worse than none.
"""
import threading


class PrintSpeaker:
    def say(self, text: str):
        print(f"[cue] {text}", flush=True)

    def close(self):
        pass


class Speaker:
    def __init__(self, rate: int = 190):
        self._pending = None
        self._cv = threading.Condition()
        self._stop = False
        self._rate = rate
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def say(self, text: str):
        with self._cv:
            self._pending = text
            self._cv.notify()

    def _run(self):
        import pyttsx3   # imported in the worker; some backends need it

        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                text, self._pending = self._pending, None
            engine.say(text)
            engine.runAndWait()

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()


def make_speaker(enabled: bool = True):
    if not enabled:
        return PrintSpeaker()
    try:
        import pyttsx3  # noqa: F401
    except ImportError:
        print("pyttsx3 not installed; printing cues instead")
        return PrintSpeaker()
    return Speaker()

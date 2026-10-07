
import threading
import time


class ThreadedCamera:
    def __init__(self, source=0, width: int | None = 640, height: int | None = 480,
                 clock=time.monotonic):
        import cv2

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera source {source!r}")
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.clock = clock
        self._frame, self._t = None, 0.0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)          # dropout: keep the old frame; it'll go stale
                continue
            with self._lock:
                self._frame, self._t = frame, self.clock()

    def latest(self):
        with self._lock:
            return self._frame, self._t

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


class StaticCamera:
 

    def __init__(self, frame=None, clock=time.monotonic):
        self.frame = frame
        self.clock = clock
        self.frozen_at = None             # set to simulate a camera that stopped updating

    def latest(self):
        if self.frame is None:
            return None, 0.0
        return self.frame, (self.frozen_at if self.frozen_at is not None else self.clock())

    def close(self):
        pass

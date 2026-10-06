import matplotlib
import matplotlib.pyplot as plt

from .plots import (_style, _save, _smooth,
                    C_BLUE, C_BLUE_LIGHT, AXIS, MUTED, INK)

PANELS = [
    ("return",    "Mean episodic return", "20-episode moving average"),
    ("entropy",   "Policy entropy",       "ln 5 = 1.609 at initialisation"),
    ("ev",        "Explained variance",   "value-function fit, 1.0 = perfect"),
    ("approx_kl", "Approximate KL",       "policy movement per update"),
    ("clip_frac", "Clip fraction",        "share of samples clipped"),
]


class LiveTrainingPlot:
    """Call update() once per PPO update; close() when training ends."""

    def __init__(self, title="PPO training", every=1):
        self.history = []
        self.every = every
        self.n = 0

        # Degrade gracefully if there's no GUI backend (CI, headless, pytest).
        self.live = "agg" not in matplotlib.get_backend().lower()
        if not self.live:
            print("[live] non-interactive backend — figure saved at the end only")

        _style()
        if self.live:
            plt.ion()

        self.fig, self.axes = plt.subplots(
            len(PANELS), 1, figsize=(7, 11), sharex=True)
        self.lines = {}

        for ax, (key, heading, sub) in zip(self.axes, PANELS):
            if key == "return":
                raw, = ax.plot([], [], color=C_BLUE_LIGHT, lw=1.0,
                               label="per update")
                smooth, = ax.plot([], [], color=C_BLUE, lw=2.0,
                                  label="smoothed")
                self.lines[key] = (raw, smooth)
                ax.legend(loc="lower right")
            else:
                line, = ax.plot([], [], color=C_BLUE, lw=1.6)
                self.lines[key] = (line,)

            if key in ("return", "ev"):
                ax.axhline(0, color=AXIS, lw=0.8, zorder=0)

            ax.set_title(heading, loc="left", pad=8)
            ax.text(0.0, 1.0, sub, transform=ax.transAxes, va="bottom",
                    ha="left", fontsize=8, color=MUTED)
            ax.grid(axis="y")
            ax.set_axisbelow(True)

        self.axes[-1].set_xlabel("environment steps")
        self.fig.suptitle(title, x=0.0, ha="left", fontsize=12, color=INK)
        self.fig.tight_layout()

        if self.live:
            self.fig.show()

    def update(self, record):
        self.history.append(record)
        self.n += 1
        if self.n % self.every:
            return
        self._refresh()
        if self.live:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def _refresh(self):
        steps = [h["steps"] for h in self.history]
        for key, lines in self.lines.items():
            y = [h[key] for h in self.history]
            lines[0].set_data(steps, y)
            if len(lines) > 1:
                lines[1].set_data(steps, _smooth(y))
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()

    def close(self, name="training_nav.png", block=False):
        self._refresh()              # always — so the saved figure has data
        if self.live:
            plt.ioff()
        self.fig.canvas.draw_idle()
        _save(self.fig, name)
        if block and self.live:
            plt.show()
        return self.history
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np


class FastPlotLib:
    def __init__(self, nrows=1, ncols=1, figsize=(8, 6), dpi=100):
        self.nrows = nrows
        self.ncols = ncols

        self.fig, self.axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            dpi=dpi,
        )

        self.fig.tight_layout()
        self.fig.canvas.draw()

    def _get_axis(self, row=0, col=0):
        if isinstance(self.axes, np.ndarray):
            if self.nrows == 1 and self.ncols == 1:
                ax = self.axes
            elif self.nrows == 1:
                ax = self.axes[col]
            else:
                ax = self.axes[row, col]
        else:
            ax = self.axes
        return ax

    @contextmanager
    def get_axis(self, row=0, col=0):
        ax = self._get_axis(row, col)
        yield ax
        self.fig.canvas.blit(ax.bbox)

    def clear(self):
        if isinstance(self.axes, np.ndarray):
            for ax in self.axes.flat:
                ax.clear()
        else:
            self.axes.clear()

    def get_canvas_as_numpy(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        s, (width, height) = self.fig.canvas.print_to_buffer()
        return np.frombuffer(s, np.uint8).reshape((height, width, 4))[:, :, :3]

    def __del__(self):
        plt.close(self.fig)

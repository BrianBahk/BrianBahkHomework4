from pathlib import Path

import imageio
import ipywidgets as widgets
from IPython.display import display
from PIL import Image


class JupyterAnimationDisplay:
    def __init__(self, frames):
        self.frames = frames
        print(f"Total frames: {len(frames)}")

    def display_interactive(
        self,
        start_frame: int = 0,
        fps: int = 10,
        gif_path: str = "tmp/animation.gif",
    ):
        frames = self.frames
        slider = widgets.IntSlider(
            value=start_frame,
            min=0,
            max=len(frames) - 1,
            step=1,
            description="Frame:",
            continuous_update=False,
        )
        play = widgets.Play(
            value=0,
            min=0,
            max=len(frames) - 1,
            step=1,
            interval=1000 // fps,
            description="Play",
            repeat=True,
        )

        def save_gif(b):
            imageio.mimsave(gif_path, frames, fps=fps)
            print(f"Saved to {Path(gif_path).resolve()}.")

        download = widgets.Button(description="Save")
        download.on_click(save_gif)

        # widgets.jslink((play, "value"), (slider, "value"))
        widgets.jslink((slider, "value"), (play, "value"))

        output = widgets.Output()

        def update_slider(change):
            idx = change["new"]
            with output:
                output.clear_output(wait=True)
                display(Image.fromarray(frames[idx]))

        slider.observe(update_slider, names="value")

        display(
            widgets.VBox(
                [
                    widgets.HBox([play, slider, download]),
                    output,
                ]
            )
        )

        update_slider({"new": start_frame})

from pathlib import Path

import fire
import numpy as np
from PIL import Image

try:
    from supertux.agents.custom_autopilot import CustomAutopilot
    from supertux.env import Rollout
except ImportError:
    from .supertux.agents.custom_autopilot import CustomAutopilot
    from .supertux.env import Rollout


def collect(
    track: str,
    agent: CustomAutopilot,
    output_path: Path,
    max_steps: int,
    frame_skip: int,
    render: bool = True,
    render_depth: bool = True,
):
    output_path.mkdir(parents=False, exist_ok=True)

    rollout = Rollout(
        render=render,
        render_depth=render_depth,
        agent=agent,
        track=track,
    )

    # meta
    track_info = None
    frames = {}

    for i, frame in enumerate(rollout.rollout(max_steps=max_steps, frame_skip=frame_skip)):
        if i == 0:
            track_info = frame
        else:
            idx = i - 1
            img = frame.pop("image")
            raw_depth = frame.pop("depth")
            depth = (raw_depth * 65535).astype(np.uint16)

            Image.fromarray(img).save(output_path / f"{idx:05d}_im.jpg")
            Image.fromarray(depth).save(output_path / f"{idx:05d}_depth.png")

            # np.savez(output_path / f"{idx:05d}_depth.npz", raw_depth)

            for k, v in frame.items():
                if k not in frames:
                    frames[k] = []

                frames[k].append(v)

    info = {"track": track_info, "frames": frames}

    np.savez(output_path / "info.npz", **info)

    del rollout


def collect_single(
    output_dir: str = "tmp",
    track: str = "lighthouse",
    max_steps: int = 500,
    frame_skip: int = 4,
    num_repeat: int = 8,
    seed: int = 2024,
):
    np.random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=False, exist_ok=True)

    print(f"Saving to {output_dir.resolve()}")

    for repeat in range(num_repeat):
        # slight randomization
        p = 7.5 + 5.0 * (2 * np.random.rand() - 1)
        i = 2.0 + 2.0 * (2 * np.random.rand() - 1)
        target_speed = 10.0 + 5.0 * (2 * np.random.rand() - 1)
        agent = CustomAutopilot(p=p, i=i, target_speed=target_speed)

        episode_path = output_dir / f"{track}_{repeat:02d}"

        collect(
            track=track,
            agent=agent,
            output_path=episode_path,
            max_steps=max_steps,
            frame_skip=frame_skip,
        )

        print(episode_path.resolve())


if __name__ == "__main__":
    fire.Fire(collect_single)

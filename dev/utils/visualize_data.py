from pathlib import Path

import cv2
import fire
import numpy as np
from solution.datasets.road_dataset import RoadDataset
from supertux.utils import VideoWriter

# one hot to colors
COLORS = np.uint8(
    [
        [0, 0, 0],
        [0, 0, 255],
        [255, 0, 0],
    ]
)


def depth_to_color_cv2(depth, cmap=cv2.COLORMAP_VIRIDIS):
    depth_clipped = depth.clip(0, 1)
    depth_uint8 = (255 * depth_clipped).astype(np.uint8)

    return cv2.applyColorMap(depth_uint8, cmap)


def visualize_numpy(image, track, depth):
    """
    Args:
        image (np.float32): (3, h, w) in [0, 1]
        track (np.float32): (h, w) in {0, 1, 2}
        depth (np.float32): (1, h, w) in [0, 1]
    """
    image = (image * 255).astype(np.uint8).transpose(1, 2, 0)
    track = COLORS[track]
    depth = depth_to_color_cv2(depth)

    return np.hstack([image, track, depth])


def visualize_dataset(dataset):
    writer = VideoWriter(fps=15)

    for i in range(len(dataset)):
        batch = dataset[i]
        viz = visualize_numpy(**batch)
        writer.append_data(viz)


def visualize_multiple(datasets, name=""):
    writer = VideoWriter(fps=15, name=name)

    for i in range(min(map(len, datasets))):
        grid = []

        for dataset in datasets:
            viz = visualize_numpy(**dataset[i])
            grid.append(viz)

        writer.append_data(np.vstack(grid))


def visualize_dataset_main(path: str):
    dataset = RoadDataset(path)
    visualize_dataset(dataset)


def visualize_multiple_main(root_dir: str, map_name: str, name: str):
    paths = list(Path(root_dir).glob(f"{map_name}_*"))
    datasets = [RoadDataset(path) for path in paths]

    visualize_multiple(datasets, name=map_name)


if __name__ == "__main__":
    fire.Fire(
        {
            "visualize_dataset": visualize_dataset_main,
            "visualize_multiple": visualize_multiple_main,
        }
    )

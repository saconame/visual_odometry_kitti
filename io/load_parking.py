from pathlib import Path

import cv2
import numpy as np


def load_parking_intrinsics(dataset_dir: Path) -> np.ndarray:
    k_path = dataset_dir / "images" / "K.txt"

    first_values = None
    with k_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            first_values = np.fromstring(line, sep=" ", dtype=np.float64)
            break

    fx, fy, cx, cy = first_values[:4]

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class ParkingSequence:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.image_paths = sorted((self.dataset_dir / "images").glob("img_*.png"))

    def __iter__(self):
        for frame_idx, image_path in enumerate(self.image_paths):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            yield frame_idx, image

    def __len__(self):
        return len(self.image_paths)


def load_parking_poses(dataset_dir: Path) -> np.ndarray:
    poses_path = dataset_dir / "poses.txt"
    raw_poses = np.loadtxt(str(poses_path))
    return raw_poses.reshape(-1, 3, 4)


from pathlib import Path

import cv2
import numpy as np


def load_kitti_intrinsics(dataset_dir: Path) -> np.ndarray:
    calib_path = dataset_dir / "05" / "calib.txt"
    
    with calib_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or not line.startswith("P0:"):
                continue
            
            # P0 is the left grayscale camera. It gives a 3x4 projection matrix.
            # The intrinsic matrix K is the left 3x3 block.
            P = np.fromstring(line.split(":", 1)[1], sep=" ").reshape(3, 4)
            return P[:, :3]

    raise ValueError("P0 not found in KITTI calibration file.")


class KittiSequence:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.image_paths = sorted((self.dataset_dir / "05" / "image_0").glob("*.png"))

    def __iter__(self):
        for frame_idx, image_path in enumerate(self.image_paths):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            yield frame_idx, image

    def __len__(self):
        return len(self.image_paths)


def load_kitti_poses(dataset_dir: Path) -> np.ndarray:
    poses_path = dataset_dir / "poses" / "05.txt"
    raw_poses = np.loadtxt(str(poses_path))
    return raw_poses.reshape(-1, 3, 4)


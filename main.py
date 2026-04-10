import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from viz.draw_landmarks import draw_landmark_count_panel
from viz.draw_tracks import draw_frame_panel
from viz.draw_trajectory import draw_trajectory_panel
from vo.initialize_vo import initialize_vo
from vo.process_frame import process_frame

ROOT_DIR = Path(__file__).resolve().parent

sys.path.append(str(ROOT_DIR / "io"))
from load_kitti import KittiSequence, load_kitti_intrinsics, load_kitti_poses
from load_parking import ParkingSequence, load_parking_intrinsics, load_parking_poses

# 0 = Parking, 1 = KITTI
DATASET = 1
SEED = 5

def main():
    np.random.seed(SEED)

    if DATASET == 0:
        dataset_dir = ROOT_DIR / "data" / "parking"
        K = load_parking_intrinsics(dataset_dir)
        sequence = ParkingSequence(dataset_dir)
        gt_poses = load_parking_poses(dataset_dir)
    else:
        dataset_dir = ROOT_DIR / "data" / "kitti"
        K = load_kitti_intrinsics(dataset_dir)
        sequence = KittiSequence(dataset_dir)
        gt_poses = load_kitti_poses(dataset_dir)

    startup_frames = []
    for _, image in sequence:
        startup_frames.append(image)
        if len(startup_frames) == 4:
            break

    state, init_info = initialize_vo(startup_frames, K, gt_poses)
    init_frame_idx = init_info["frame_b_index"]

    print("K:")
    print(K)
    print("Initialization stats:")
    print(init_info["stats"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Visual odometry", fontsize=14)

    t_wc = state["T_wc"][:3, 3]
    trajectory = [(0.0, 0.0), (float(t_wc[0]), float(t_wc[2]))]
    landmark_count = [0, int(state["X"].shape[1])]

    previous_image = startup_frames[init_frame_idx]
    draw_frame_panel(axes[0], previous_image, init_frame_idx, state)
    draw_trajectory_panel(axes[1], trajectory)
    draw_landmark_count_panel(axes[2], landmark_count)
    plt.pause(0.001)

    for frame_idx, image in sequence:
        if frame_idx <= init_frame_idx:
            continue

        state = process_frame(image, previous_image, state)

        t_wc = state["T_wc"][:3, 3]
        trajectory.append((float(t_wc[0]), float(t_wc[2])))
        landmark_count.append(int(state["X"].shape[1]))

        draw_frame_panel(axes[0], image, frame_idx, state)
        draw_trajectory_panel(axes[1], trajectory)
        draw_landmark_count_panel(axes[2], landmark_count)

        plt.pause(0.001)
        previous_image = image

    plt.show()


if __name__ == "__main__":
    main()

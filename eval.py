import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from metrics.align import align_trajectory
from metrics.ate import compute_ate
from metrics.rte import compute_rte
from viz.draw_trajectory import plot_trajectory
from vo.initialize_vo import initialize_vo
from vo.process_frame import process_frame

ROOT_DIR = Path(__file__).resolve().parent

sys.path.append(str(ROOT_DIR / "io"))
from load_kitti import KittiSequence, load_kitti_intrinsics, load_kitti_poses
from load_parking import ParkingSequence, load_parking_intrinsics, load_parking_poses

# 0 = Parking, 1 = KITTI
DATASET = 1
RTE_DELTA = 10
ANALYSIS_FRAMES = 240
SEED = 19


class SequenceSlice:
    def __init__(self, base_sequence, max_frames):
        self.base_sequence = base_sequence
        self.max_frames = int(max_frames)

    def __iter__(self):
        for frame_idx, image in self.base_sequence:
            if frame_idx >= self.max_frames:
                break
            yield frame_idx, image

    def __len__(self):
        return min(len(self.base_sequence), self.max_frames)


def load_dataset(dataset):
    if dataset == 0:
        dataset_dir = ROOT_DIR / "data" / "parking"
        K = load_parking_intrinsics(dataset_dir)
        sequence = ParkingSequence(dataset_dir)
        gt_poses = load_parking_poses(dataset_dir)
    else:
        dataset_dir = ROOT_DIR / "data" / "kitti"
        K = load_kitti_intrinsics(dataset_dir)
        sequence = KittiSequence(dataset_dir)
        gt_poses = load_kitti_poses(dataset_dir)

    return K, sequence, gt_poses


def run_vo_headless(sequence, K, gt_poses, analysis_frames=None, show_progress=False):
    if analysis_frames is not None:
        sequence = SequenceSlice(sequence, analysis_frames)
        gt_poses = gt_poses[:analysis_frames]

    startup_frames = []
    for _, image in sequence:
        startup_frames.append(image)
        if len(startup_frames) == 4:
            break

    state, init_info = initialize_vo(startup_frames, K, gt_poses)
    init_frame_idx = init_info["frame_b_index"]

    est_poses = [state["T_wc"].copy()]
    previous_image = startup_frames[init_frame_idx]

    total_steps = max(len(sequence) - (init_frame_idx + 1), 0)

    iterator = sequence
    if show_progress:
        progress_ctx = tqdm(total=total_steps, desc="VO progress", unit="frame")
    else:
        progress_ctx = None

    processed_steps = 0
    try:
        if progress_ctx is None:
            for frame_idx, image in iterator:
                if frame_idx <= init_frame_idx:
                    continue
                if processed_steps >= total_steps:
                    break

                state = process_frame(image, previous_image, state)
                est_poses.append(state["T_wc"].copy())
                previous_image = image
                processed_steps += 1
        else:
            for frame_idx, image in iterator:
                if frame_idx <= init_frame_idx:
                    continue
                if processed_steps >= total_steps:
                    break

                state = process_frame(image, previous_image, state)
                est_poses.append(state["T_wc"].copy())
                previous_image = image
                processed_steps += 1
                progress_ctx.update(1)
    finally:
        if progress_ctx is not None:
            progress_ctx.close()

    return est_poses, init_frame_idx, init_info


def main():
    K, sequence, gt_poses = load_dataset(DATASET)
    np.random.seed(SEED)

    if ANALYSIS_FRAMES is not None:
        sequence = SequenceSlice(sequence, ANALYSIS_FRAMES)
        gt_poses = gt_poses[:ANALYSIS_FRAMES]

    est_poses, init_frame_idx, init_info = run_vo_headless(
        sequence,
        K,
        gt_poses,
        analysis_frames=None,
        show_progress=True,
    )

    gt_eval = gt_poses[init_frame_idx:]
    n = min(gt_eval.shape[0], len(est_poses))

    gt_eval = gt_eval[:n].copy()
    est_eval = est_poses[:n]

    est_aligned = align_trajectory(gt_eval, est_eval, constrain_first_state=False)

    ate_rmse = compute_ate(gt_eval, est_aligned)
    rte_rmse = compute_rte(gt_eval, est_aligned, delta=RTE_DELTA)

    print("Initialization stats:")
    print(init_info["stats"])
    print(f"Evaluated poses: {n}")
    print(f"ATE RMSE: {ate_rmse:.4f} m")
    print(f"RTE RMSE (delta={RTE_DELTA}): {rte_rmse:.4f} m")

    plot_trajectory(gt_eval, est_aligned, title="Aligned trajectories (SE(3) aligned)")


if __name__ == "__main__":
    main()

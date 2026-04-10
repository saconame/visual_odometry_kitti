import numpy as np


def compute_rte(gt_poses, est_poses_aligned, delta=10):
    gt_xyz = np.asarray(gt_poses, dtype=np.float64)[:, :3, 3]
    est_xyz = np.asarray(est_poses_aligned, dtype=np.float64)[:, :3, 3]

    n = min(gt_xyz.shape[0], est_xyz.shape[0])
    gt_step = gt_xyz[delta:n] - gt_xyz[: n - delta]
    est_step = est_xyz[delta:n] - est_xyz[: n - delta]

    errors = np.linalg.norm(est_step - gt_step, axis=1)
    return float(np.sqrt(np.mean(errors * errors)))

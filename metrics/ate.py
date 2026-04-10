import numpy as np


def compute_ate(gt_poses, est_poses_aligned):
    gt_xyz = np.asarray(gt_poses, dtype=np.float64)[:, :3, 3]
    est_xyz = np.asarray(est_poses_aligned, dtype=np.float64)[:, :3, 3]

    n = min(gt_xyz.shape[0], est_xyz.shape[0])
    errors = np.linalg.norm(gt_xyz[:n] - est_xyz[:n], axis=1)
    return float(np.sqrt(np.mean(errors * errors)))

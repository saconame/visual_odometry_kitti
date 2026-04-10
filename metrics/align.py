import numpy as np


def _as_homogeneous_batch(poses):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape[1:] == (3, 4):
        out = np.zeros((poses.shape[0], 4, 4), dtype=np.float64)
        out[:, :3, :] = poses
        out[:, 3, 3] = 1.0
        return out
    return poses


def _estimate_similarity_transform(src_xyz, dst_xyz):
    n = src_xyz.shape[0]
    mu_src = src_xyz.mean(axis=0)
    mu_dst = dst_xyz.mean(axis=0)

    src_centered = src_xyz - mu_src
    dst_centered = dst_xyz - mu_dst

    covariance = (dst_centered.T @ src_centered) / float(n)
    U, singular_values, Vt = np.linalg.svd(covariance)

    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0.0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    src_var = np.mean(np.sum(src_centered * src_centered, axis=1))
    scale = np.trace(np.diag(singular_values) @ S) / src_var

    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def align_trajectory(gt_poses, est_poses, constrain_first_state=True):
    raw_est = np.asarray(est_poses)
    gt_h = _as_homogeneous_batch(gt_poses)
    est_h = _as_homogeneous_batch(est_poses)

    n = min(gt_h.shape[0], est_h.shape[0])
    gt_h = gt_h[:n]
    est_h = est_h[:n]

    gt_xyz = gt_h[:, :3, 3]
    est_xyz = est_h[:, :3, 3]

    scale, R, t = _estimate_similarity_transform(est_xyz, gt_xyz)
    if constrain_first_state:
        t = gt_xyz[0] - scale * (R @ est_xyz[0])

    aligned = est_h.copy()
    aligned[:, :3, :3] = np.einsum("ij,njk->nik", R, est_h[:, :3, :3])
    aligned[:, :3, 3] = (scale * (R @ est_xyz.T)).T + t

    if raw_est.ndim == 3 and raw_est.shape[1:] == (3, 4):
        return aligned[:, :3, :]

    return aligned

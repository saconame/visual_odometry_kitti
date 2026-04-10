import matplotlib.pyplot as plt
import numpy as np


def draw_trajectory_panel(ax, trajectory_xz):
    ax.clear()

    points = np.asarray(trajectory_xz, dtype=np.float64)
    ax.plot(points[:, 0], points[:, 1], linewidth=2.0, color="#1f77b4")
    ax.scatter(points[-1, 0], points[-1, 1], s=30, color="#d62728")

    ax.set_title("Trajectory (x-z)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.6)


def _poses_to_xz(poses):
    poses = np.asarray(poses, dtype=np.float64)
    return poses[:, 0, 3], poses[:, 2, 3]


def plot_trajectory(gt_poses, est_poses_aligned, title=None):
    gt_x, gt_z = _poses_to_xz(gt_poses)
    est_x, est_z = _poses_to_xz(est_poses_aligned)

    all_x = np.concatenate((gt_x, est_x))
    all_z = np.concatenate((gt_z, est_z))
    span = max(float(np.ptp(all_x)), float(np.ptp(all_z)), 1.0)
    pad = 0.05 * span

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gt_x, gt_z, linewidth=2.0, color="#222222", label="ground truth")
    ax.plot(est_x, est_z, linewidth=1.8, color="#1f77b4", label="estimate (aligned)")
    ax.set_title("Trajectory (x-z)" if title is None else title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_xlim(float(all_x.min() - pad), float(all_x.max() + pad))
    ax.set_ylim(float(all_z.min() - pad), float(all_z.max() + pad))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.6)
    ax.legend()
    plt.show()

    return fig, ax

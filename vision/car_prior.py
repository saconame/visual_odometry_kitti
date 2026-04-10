import cv2
import numpy as np
from numba import njit


CAR_PRIOR_TAU_THETA_DEG = 3.0
CAR_PRIOR_MAX_ITERS = 200


@njit(fastmath=True)
def _compute_1pt_ransac_inliers(u1, v1, u2, v2, sample_indices, tau_theta_rad):
    N = u1.shape[0]
    theta_all = 2.0 * np.arctan2(v2 - v1, u2 + u1)
    iters = sample_indices.shape[0]
    
    best_inlier_count = -1
    best_idx = 0
    
    for i in range(iters):
        idx = sample_indices[i]
        theta_hat = theta_all[idx]
        
        inlier_count = 0
        for j in range(N):
            diff = theta_all[j] - theta_hat
            diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
            if np.abs(diff) < tau_theta_rad:
                inlier_count += 1
                
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_idx = i
            
    best_theta = theta_all[sample_indices[best_idx]]
    best_mask = np.zeros(N, dtype=np.bool_)
    for j in range(N):
        diff = theta_all[j] - best_theta
        diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
        if np.abs(diff) < tau_theta_rad:
            best_mask[j] = True
            
    return best_mask


def filter_car_prior_1pt_ransac(points_1, points_2, K, tau_theta_deg=None, max_iters=None):
    """
    Filters feature  under a planar-circular car motion prior
    """
    tau_theta_deg = float(CAR_PRIOR_TAU_THETA_DEG)
    max_iters = int(CAR_PRIOR_MAX_ITERS)

    N = points_1.shape[0]
    if N < 2:
        return np.ones(N, dtype=bool)

    p1_norm = cv2.undistortPoints(points_1.reshape(-1, 1, 2), cameraMatrix=K, distCoeffs=None).reshape(-1, 2)
    p2_norm = cv2.undistortPoints(points_2.reshape(-1, 1, 2), cameraMatrix=K, distCoeffs=None).reshape(-1, 2)

    u1, v1 = p1_norm[:, 0], p1_norm[:, 1]
    u2, v2 = p2_norm[:, 0], p2_norm[:, 1]
    
    tau_theta_rad = np.radians(tau_theta_deg)
    iters = min(N, max_iters)
    sample_indices = np.random.choice(N, iters, replace=False) if iters < N else np.arange(N)

    return _compute_1pt_ransac_inliers(u1, v1, u2, v2, sample_indices, tau_theta_rad)

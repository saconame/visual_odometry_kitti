import cv2
import numpy as np


PNP_ITERATIONS = 100
PNP_REPROJECTION_ERROR = 2.5
PNP_CONFIDENCE = 0.99
PNP_FLAGS = cv2.SOLVEPNP_AP3P


def estimate_pose_pnp_ransac(points_3d, points_2d, K):
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d,
        points_2d,
        K,
        distCoeffs=None,
        iterationsCount=PNP_ITERATIONS,
        reprojectionError=PNP_REPROJECTION_ERROR,
        confidence=PNP_CONFIDENCE,
        flags=PNP_FLAGS,
    )

    inlier_mask = np.zeros((points_2d.shape[0],), dtype=bool)
    if not success or inliers is None:
        return None, inlier_mask

    inlier_mask[inliers.reshape(-1)] = True

    R, _ = cv2.Rodrigues(rvec)
    T_cw = np.eye(4, dtype=np.float64)
    T_cw[:3, :3] = R
    T_cw[:3, 3] = tvec.reshape(3)
    T_wc = np.linalg.inv(T_cw)

    return T_wc, inlier_mask

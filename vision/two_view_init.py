import cv2
import numpy as np

from vision.detect_features import detect_features
from vision.track_klt import track_klt
from vision.triangulation import triangulate_points
from vision.car_prior import filter_car_prior_1pt_ransac


ESSENTIAL_RANSAC_PROB = 0.99
ESSENTIAL_RANSAC_THRESHOLD = 1.2
ESSENTIAL_RANSAC_MAX_ITERS = 2000


def two_view_initialization(image_a, image_b, K, scale=1.0):
    detected_points = detect_features(image_a)
    points_a, points_b, _ = track_klt(image_a, image_b, detected_points)

    prior_inliers = filter_car_prior_1pt_ransac(points_a, points_b, K)
    points_a = points_a[prior_inliers]
    points_b = points_b[prior_inliers]

    E, ransac_mask = cv2.findEssentialMat(
        points_a,
        points_b,
        cameraMatrix=K,
        method=cv2.RANSAC,
        prob=ESSENTIAL_RANSAC_PROB,
        threshold=ESSENTIAL_RANSAC_THRESHOLD,
        maxIters=ESSENTIAL_RANSAC_MAX_ITERS,
    )

    if E.shape[0] > 3:
        E = E[:3, :]

    ransac_inliers = ransac_mask.reshape(-1).astype(bool)
    points_a = points_a[ransac_inliers]
    points_b = points_b[ransac_inliers]

    _, R, t, pose_mask = cv2.recoverPose(E, points_a, points_b, K)
    t = t * float(scale)
    pose_inliers = pose_mask.reshape(-1).astype(bool)
    points_a = points_a[pose_inliers]
    points_b = points_b[pose_inliers]

    P_a = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P_b = K @ np.hstack((R, t))
    points_3d = triangulate_points(P_a, P_b, points_a, points_b)

    depth_a = points_3d[:, 2]
    points_3d_camera_b = (R @ points_3d.T + t).T
    depth_b = points_3d_camera_b[:, 2]
    valid_depth = (depth_a > 0.0) & (depth_b > 0.0)

    points_b = points_b[valid_depth]
    points_3d = points_3d[valid_depth]

    T_cw = np.eye(4, dtype=np.float64)
    T_cw[:3, :3] = R
    T_cw[:3, 3] = t.reshape(3)
    T_wc = np.linalg.inv(T_cw)

    stats = {
        "detected": int(detected_points.shape[0]),
        "tracked": int(ransac_inliers.shape[0]),
        "ransac_inliers": int(ransac_inliers.sum()),
        "pose_inliers": int(pose_inliers.sum()),
        "triangulated": int(points_3d.shape[0]),
    }

    return {
        "P": points_b.T.astype(np.float32),
        "X": points_3d.T.astype(np.float32),
        "T_wc": T_wc,
        "stats": stats,
    }

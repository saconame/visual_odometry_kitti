import numpy as np

from vision.pnp_ransac import estimate_pose_pnp_ransac
from vision.track_klt import track_klt
from vision.car_prior import filter_car_prior_1pt_ransac


MIN_PNP_INLIERS = 4


def update_landmarks(previous_image, current_image, state):
    previous_points = np.ascontiguousarray(state["P"].T)
    previous_landmarks = state["X"].T

    _, current_points, tracked_mask = track_klt(
        previous_image,
        current_image,
        previous_points,
    )
    tracked_landmarks = previous_landmarks[tracked_mask]
    tracked_previous_points = previous_points[tracked_mask]

    prior_mask = filter_car_prior_1pt_ransac(tracked_previous_points, current_points, state["K"])
    current_points = current_points[prior_mask]
    tracked_landmarks = tracked_landmarks[prior_mask]

    if current_points.shape[0] < 4:
        next_state = {
            "P": current_points.T.astype(np.float32),
            "X": tracked_landmarks.T.astype(np.float32),
            "C": state["C"],
            "F": state["F"],
            "T_first": state["T_first"],
            "T_wc": state["T_wc"],
            "K": state["K"],
        }
        return next_state

    T_wc, pnp_inlier_mask = estimate_pose_pnp_ransac(
        tracked_landmarks,
        current_points,
        state["K"],
    )

    inlier_count = int(np.count_nonzero(pnp_inlier_mask))
    if T_wc is None or inlier_count < MIN_PNP_INLIERS:
        # Keep tracked points when PnP is unreliable so one bad frame does not wipe the map.
        next_state = {
            "P": current_points.T.astype(np.float32),
            "X": tracked_landmarks.T.astype(np.float32),
            "C": state["C"],
            "F": state["F"],
            "T_first": state["T_first"],
            "T_wc": state["T_wc"],
            "K": state["K"],
        }
        return next_state

    inlier_points = current_points[pnp_inlier_mask]
    inlier_landmarks = tracked_landmarks[pnp_inlier_mask]

    next_state = {
        "P": inlier_points.T.astype(np.float32),
        "X": inlier_landmarks.T.astype(np.float32),
        "C": state["C"],
        "F": state["F"],
        "T_first": state["T_first"],
        "T_wc": state["T_wc"] if T_wc is None else T_wc,
        "K": state["K"],
    }

    return next_state

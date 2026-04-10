import cv2
import numpy as np
from numba import njit

from vision.track_klt import track_klt
from vision.triangulation import triangulate_points
from vision.car_prior import filter_car_prior_1pt_ransac


MIN_TRIANGULATION_ANGLE_DEG = 1.5
ADAPTIVE_RETRI_LANDMARK_THRESHOLD = 100
ADAPTIVE_RETRI_MIN_ANGLE_DEG = 0.4
ADAPTIVE_RETRI_MAX_ANGLE_DEG = 8.0
MIN_LANDMARKS_FOR_REPLENISH = 160
MIN_CANDIDATES_BUFFER = 200
MAX_CANDIDATES = 700
MASK_RADIUS_PX = 12
REPLENISH_QUALITY_LEVEL = 0.005
REPLENISH_MIN_DISTANCE = 8
REPLENISH_BLOCK_SIZE = 7
REPLENISH_USE_HARRIS_DETECTOR = False


def _copy_state_with_updates(state, updates):
    next_state = {
        "P": state["P"],
        "X": state["X"],
        "C": state["C"],
        "F": state["F"],
        "T_first": state["T_first"],
        "T_wc": state["T_wc"],
        "K": state["K"],
    }
    next_state.update(updates)
    return next_state


@njit(fastmath=True)
def _pixel_to_world_ray(pixel, K_inv, R_wc):
    x = np.array([pixel[0], pixel[1], 1.0], dtype=np.float64)
    ray_cam = K_inv @ x
    ray_world = R_wc @ ray_cam
    norm = np.linalg.norm(ray_world)
    if norm == 0.0:
        return ray_world
    return ray_world / norm


@njit(fastmath=True)
def _filter_candidates_by_angle(first_points, current_points, first_poses, current_pose, K_inv, min_angle, max_angle):
    candidate_count = first_points.shape[0]
    passed_mask = np.zeros(candidate_count, dtype=np.bool_)
    current_R = np.ascontiguousarray(current_pose[:3, :3])
    
    for idx in range(candidate_count):
        first_R = np.ascontiguousarray(first_poses[idx, :3, :3])
        ray_first = _pixel_to_world_ray(
            first_points[idx],
            K_inv,
            first_R
        )
        ray_current = _pixel_to_world_ray(
            current_points[idx],
            K_inv,
            current_R
        )
        
        cosine = np.dot(ray_first, ray_current)
        if cosine > 1.0:
            cosine = 1.0
        elif cosine < -1.0:
            cosine = -1.0
            
        angle_deg = np.degrees(np.arccos(cosine))
        if angle_deg >= min_angle and angle_deg <= max_angle:
            passed_mask[idx] = True
            
    return passed_mask


def update_candidates(previous_image, current_image, state):
    previous_candidates = np.ascontiguousarray(state["C"].T)

    if previous_candidates.shape[0] == 0:
        return _copy_state_with_updates(state, {})

    _, current_candidates, tracked_mask = track_klt(
        previous_image,
        current_image,
        previous_candidates,
    )

    tracked_previous_candidates = previous_candidates[tracked_mask]
    prior_mask = filter_car_prior_1pt_ransac(tracked_previous_candidates, current_candidates, state["K"])
    current_candidates = current_candidates[prior_mask]

    first_observations = state["F"].T[tracked_mask][prior_mask]
    first_poses = state["T_first"][tracked_mask][prior_mask]

    updates = {
        "C": current_candidates.T.astype(np.float32),
        "F": first_observations.T.astype(np.float32),
        "T_first": first_poses.astype(np.float64),
    }

    return _copy_state_with_updates(state, updates)


def triangulate_candidates(state):
    candidate_count = state["C"].shape[1]
    if candidate_count == 0:
        return _copy_state_with_updates(state, {})

    landmark_count = state["X"].shape[1]
    if landmark_count < ADAPTIVE_RETRI_LANDMARK_THRESHOLD:
        angle_min = ADAPTIVE_RETRI_MIN_ANGLE_DEG
        angle_max = ADAPTIVE_RETRI_MAX_ANGLE_DEG
    else:
        angle_min = MIN_TRIANGULATION_ANGLE_DEG
        angle_max = 1e9

    K = state["K"]
    K_inv = np.ascontiguousarray(np.linalg.inv(K))
    current_pose = state["T_wc"]
    current_pose_inv = np.linalg.inv(current_pose)
    current_points = state["C"].T.astype(np.float64)
    first_points = state["F"].T.astype(np.float64)
    first_poses = state["T_first"]

    keep_mask = np.ones((candidate_count,), dtype=bool)
    accepted_points = []
    accepted_landmarks = []

    passed_angle_mask = _filter_candidates_by_angle(
        first_points,
        current_points,
        first_poses,
        current_pose,
        K_inv,
        angle_min,
        angle_max,
    )

    for idx in range(candidate_count):
        if not passed_angle_mask[idx]:
            continue

        first_pose = first_poses[idx]
        first_pose_inv = np.linalg.inv(first_pose)

        P_first = K @ first_pose_inv[:3, :]
        P_current = K @ current_pose_inv[:3, :]
        point_world = triangulate_points(
            P_first,
            P_current,
            first_points[idx : idx + 1],
            current_points[idx : idx + 1],
        )[0]

        if not np.isfinite(point_world).all():
            continue

        point_h = np.array(
            [point_world[0], point_world[1], point_world[2], 1.0],
            dtype=np.float64,
        )
        depth_first = (first_pose_inv @ point_h)[2]
        depth_current = (current_pose_inv @ point_h)[2]
        if depth_first <= 0.0 or depth_current <= 0.0:
            continue

        keep_mask[idx] = False
        accepted_points.append(current_points[idx])
        accepted_landmarks.append(point_world)

    remaining_candidates = current_points[keep_mask]
    remaining_first = first_points[keep_mask]
    remaining_poses = first_poses[keep_mask]

    landmarks_2d = state["P"]
    landmarks_3d = state["X"]
    if accepted_points:
        appended_points = np.vstack(accepted_points).astype(np.float32).T
        appended_landmarks = np.vstack(accepted_landmarks).astype(np.float32).T
        landmarks_2d = np.hstack((landmarks_2d, appended_points))
        landmarks_3d = np.hstack((landmarks_3d, appended_landmarks))

    updates = {
        "P": landmarks_2d,
        "X": landmarks_3d,
        "C": remaining_candidates.T.astype(np.float32),
        "F": remaining_first.T.astype(np.float32),
        "T_first": remaining_poses.astype(np.float64),
    }
    return _copy_state_with_updates(state, updates)


def replenish_candidates(image, state):
    landmark_count = state["P"].shape[1]
    candidate_count = state["C"].shape[1]
    if (
        landmark_count >= MIN_LANDMARKS_FOR_REPLENISH
        and candidate_count >= MIN_CANDIDATES_BUFFER
    ):
        return _copy_state_with_updates(state, {})

    remaining_slots = MAX_CANDIDATES - candidate_count
    if remaining_slots <= 0:
        return _copy_state_with_updates(state, {})

    mask = np.full(image.shape, 255, dtype=np.uint8)
    for points in (state["P"], state["C"]):
        if points.shape[1] == 0:
            continue
        for point in points.T:
            cv2.circle(
                mask,
                (int(round(point[0])), int(round(point[1]))),
                MASK_RADIUS_PX,
                0,
                -1,
            )

    new_points = cv2.goodFeaturesToTrack(
        image,
        maxCorners=remaining_slots,
        qualityLevel=REPLENISH_QUALITY_LEVEL,
        minDistance=REPLENISH_MIN_DISTANCE,
        blockSize=REPLENISH_BLOCK_SIZE,
        useHarrisDetector=REPLENISH_USE_HARRIS_DETECTOR,
        mask=mask,
    )
    if new_points is None:
        return _copy_state_with_updates(state, {})

    new_points = new_points.reshape(-1, 2).astype(np.float32)
    new_points_t = new_points.T
    new_first_poses = np.repeat(
        state["T_wc"][np.newaxis, :, :],
        new_points.shape[0],
        axis=0,
    )

    updates = {
        "C": np.hstack((state["C"], new_points_t)),
        "F": np.hstack((state["F"], new_points_t)),
        "T_first": np.concatenate((state["T_first"], new_first_poses), axis=0),
    }
    return _copy_state_with_updates(state, updates)

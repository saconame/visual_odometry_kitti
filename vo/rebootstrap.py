import numpy as np

from vision.detect_features import detect_features
from vo.update_candidates import MAX_CANDIDATES


CKP_THRESHOLD = 150
MIN_NEIGHBOUR_DISTANCE = 20.0


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


def _filter_by_neighbour_distance(query_points, reference_points, min_distance):
    if query_points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)

    if reference_points.shape[0] == 0:
        return query_points.astype(np.float32)

    min_dist_sq = float(min_distance * min_distance)
    accepted = []
    running_reference = reference_points.astype(np.float32)

    for point in query_points:
        diffs = running_reference - point
        dist_sq = np.sum(diffs * diffs, axis=1)
        if float(np.min(dist_sq)) > min_dist_sq:
            accepted.append(point)
            running_reference = np.vstack((running_reference, point))

    if len(accepted) == 0:
        return np.empty((0, 2), dtype=np.float32)

    return np.vstack(accepted).astype(np.float32)


def rebootstrap_candidates(image, state):
    candidate_count = state["C"].shape[1]
    if candidate_count >= CKP_THRESHOLD:
        return _copy_state_with_updates(state, {})

    remaining_slots = MAX_CANDIDATES - candidate_count
    if remaining_slots <= 0:
        return _copy_state_with_updates(state, {})

    detected_points = detect_features(image)
    if detected_points.shape[0] == 0:
        return _copy_state_with_updates(state, {})

    references = []
    if state["P"].shape[1] > 0:
        references.append(state["P"].T.astype(np.float32))
    if state["C"].shape[1] > 0:
        references.append(state["C"].T.astype(np.float32))

    if len(references) == 0:
        reference_points = np.empty((0, 2), dtype=np.float32)
    else:
        reference_points = np.vstack(references)

    filtered_points = _filter_by_neighbour_distance(
        detected_points,
        reference_points,
        MIN_NEIGHBOUR_DISTANCE,
    )
    if filtered_points.shape[0] == 0:
        return _copy_state_with_updates(state, {})

    if filtered_points.shape[0] > remaining_slots:
        filtered_points = filtered_points[:remaining_slots]

    filtered_points_t = filtered_points.T.astype(np.float32)
    first_poses = np.repeat(
        state["T_wc"][np.newaxis, :, :],
        filtered_points.shape[0],
        axis=0,
    )

    updates = {
        "C": np.hstack((state["C"], filtered_points_t)),
        "F": np.hstack((state["F"], filtered_points_t)),
        "T_first": np.concatenate((state["T_first"], first_poses), axis=0),
    }
    return _copy_state_with_updates(state, updates)

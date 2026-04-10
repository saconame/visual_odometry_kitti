import cv2
import numpy as np


LK_CRITERIA_COUNT = 30
LK_CRITERIA_EPS = 0.02
LK_WIN_SIZE = (27, 27)
LK_MAX_LEVEL = 3


def track_klt(previous_image, current_image, previous_points):
    if previous_points.size == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        empty_mask = np.zeros((0,), dtype=bool)
        return empty, empty, empty_mask

    lk_criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        LK_CRITERIA_COUNT,
        LK_CRITERIA_EPS,
    )

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_image,
        current_image,
        previous_points.reshape(-1, 1, 2),
        None,
        winSize=LK_WIN_SIZE,
        maxLevel=LK_MAX_LEVEL,
        criteria=lk_criteria,
    )

    valid = status.reshape(-1).astype(bool)
    tracked_previous = previous_points[valid]
    tracked_current = next_points.reshape(-1, 2)[valid]

    return tracked_previous.astype(np.float32), tracked_current.astype(np.float32), valid

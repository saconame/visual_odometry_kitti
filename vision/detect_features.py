import cv2
import numpy as np


MAX_CORNERS = 1800
QUALITY_LEVEL = 0.005
MIN_DISTANCE = 8
BLOCK_SIZE = 7
USE_HARRIS_DETECTOR = False


def detect_features(image):
    corners = cv2.goodFeaturesToTrack(
        image,
        maxCorners=MAX_CORNERS,
        qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE,
        blockSize=BLOCK_SIZE,
        useHarrisDetector=USE_HARRIS_DETECTOR,
    )

    if corners is None:
        return np.empty((0, 2), dtype=np.float32)

    return corners.reshape(-1, 2).astype(np.float32)

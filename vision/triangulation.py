import cv2


def triangulate_points(P_a, P_b, points_a, points_b):
    points_4d = cv2.triangulatePoints(P_a, P_b, points_a.T, points_b.T)
    points_3d = (points_4d[:3] / points_4d[3]).T
    return points_3d

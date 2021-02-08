import io

import numpy as np
from PIL import Image

def pil_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, 'jpeg')
    return buf.getvalue()

def _determine_quadrant(vector):
    if vector[0] > 0 and vector[1] > 0:
        return 1
    elif vector[0] < 0 and vector[1] > 0:
        return 2
    elif vector[0] < 0 and vector[1] < 0:
        return 3
    elif vector[0] > 0 and vector[1] < 0:
        return 4

def get_angle_2d(keypoints):
    # (2, 1, 5) -> (right shoulder, neck, left shoulder)
    # (9, 8, 12)-> (right hip, middle hip, left hip
    # (1, 0, 15, 16, 17, 18)
    # ==================================================================
    x = np.array([1, 0])
    y = np.array([0, -1])

    # Compute face orientation
    # =================================================================
    face_keypoints = keypoints[[1, 0, 15, 16, 17, 18]]
    neck, points = face_keypoints[0], face_keypoints[1:]
    face_xoffset = np.sum(points[points[:, -1] > 0][:, 0] - neck[0])
    face_nvalids = np.sum(points[points[:, -1] > 0])

    # Compute unit normal vector of shoudler part
    # =================================================================
    planB = keypoints[1]
    rs = keypoints[2] if keypoints[2][-1] > 0 else planB
    ls = keypoints[5] if keypoints[5][-1] > 0 else planB
    # When face 90 degree or 270 degree
    vector = (ls-rs)[:2] + 1e-6
    if (
        abs(vector[0]) < 50
        and abs(face_xoffset) > 50
        and face_nvalids >=3
    ):
        vector[1] *= 100000
    # Compute vector and vector norm
    unit = vector / np.sqrt(np.sum(vector**2))
    unit_norm = (unit[1], -unit[0])
    # Post processing
    s_quadrant = _determine_quadrant(unit_norm)
    s_degree = np.arccos(np.dot(unit_norm, y)) / np.pi * 180
    s_degree = s_degree if s_quadrant == 1 or s_quadrant == 4 else 360 - s_degree
    if face_xoffset > 50 and np.sum(points[points[:, -1] > 0]) >= 3:
        s_degree = s_degree if s_degree <= 180 else 360 - s_degree
    elif face_xoffset < -50 and np.sum(points[points[:, -1] > 0]) >= 3:
        s_degree = s_degree if s_degree > 180 else 360 - s_degree
    s_conf = np.sqrt(rs[-1]*ls[-1])

    # Compute unit normal vector of hip part
    # =================================================================
    planB = keypoints[8]
    rs = keypoints[9] if keypoints[9][-1] > 0 else planB
    ls = keypoints[12] if keypoints[12][-1] > 0 else planB
    # When face 90 degree or 270 degree
    vector = (ls-rs)[:2] + 1e-6
    if (
        abs(vector[0]) < 50
        and abs(face_xoffset) > 50
        and face_nvalids >=3
    ):
        vector[1] *= 100000
    # Compute vector and vector norm
    unit = vector / np.sqrt(np.sum(vector**2))
    unit_norm = (unit[1], -unit[0])
    # Post processing
    h_quadrant = _determine_quadrant(unit_norm)
    h_degree = np.arccos(np.dot(unit_norm, y)) / np.pi * 180
    h_degree = h_degree if h_quadrant == 1 or h_quadrant == 4 else 360 - h_degree
    if face_xoffset > 50 and np.sum(points[points[:, -1] > 0]) >= 3:
        h_degree = h_degree if h_degree <= 180 else 360 - h_degree
    elif face_xoffset < -50 and np.sum(points[points[:, -1] > 0]) >= 3:
        h_degree = h_degree if h_degree > 180 else 360 - h_degree
    h_conf = np.sqrt(rs[-1]*ls[-1])

    # Select candidate
    angle = s_degree if s_conf >= h_conf else h_degree
    conf = s_conf if s_conf >= h_conf else h_conf

    return angle, conf

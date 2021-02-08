import numpy as np

def compute_iou_dist(bboxes1, bboxes2):
    """Return iou distance between bboxes1 and bboxes2

    Args:
        bboxes1 (np.ndarray): array of shape (N, 4)
        bboxes2 (np.ndarray): array of shape (M, 4)

    Return:
        A N by M dimensional distance vector

    Note:
        A bbox is (xmin, ymin, xmax, ymax)
    """
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    interArea = np.maximum((xB-xA+1), 0)*np.maximum((yB-yA+1), 0)
    bbox1Area = (x12-x11+1)*(y12-y11+1)
    bbox2Area = (x22-x21+1)*(y22-y21+1)

    iou = interArea / (bbox1Area+np.transpose(bbox2Area)-interArea)
    return 1 - iou

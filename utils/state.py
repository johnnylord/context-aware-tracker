import numpy as np


def is_occluded(dbox, dboxes):
    for target in dboxes:
        if (
            np.sum(dbox) == np.sum(target)
            or dbox[4] <= target[4]
        ):
            continue
        # Compute overlap
        xmin1, _, xmax1, _ = dbox[:4]
        xmin2, _, xmax2, _ = target[:4]
        overlap = max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
        ratio = overlap/(max(xmax1, xmax2) - min(xmin1, xmin2))
        if overlap >= 0.3:
            return True

    return False

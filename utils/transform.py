import cv2
import numpy as np

def scale_bboxes_coord(bboxes, old_resolution, new_resolution):
    ratio = np.array(new_resolution) / np.array(old_resolution)
    x_scale = ratio[1]
    y_scale = ratio[0]
    bboxes[:, 0] *= x_scale
    bboxes[:, 1] *= y_scale
    bboxes[:, 2] *= x_scale
    bboxes[:, 3] *= y_scale
    return bboxes

def scale_masks_coord(masks, old_resolution, new_resolution):
    ratio = np.array(new_resolution) / np.array(old_resolution)
    x_scale = ratio[1]
    y_scale = ratio[0]
    new_masks = []
    for mask in masks:
        height, width = mask.shape[:2]
        width *= x_scale
        height *= y_scale
        new_mask = cv2.resize(mask, (int(width), int(height)))
        new_masks.append(new_mask)
    return new_masks

def scale_keypoints_coord(keypoints, old_resolution, new_resolution):
    ratio = np.array(new_resolution) / np.array(old_resolution)
    x_scale = ratio[1]
    y_scale = ratio[0]
    keypoints[:, 0] *= x_scale
    keypoints[:, 1] *= y_scale
    return keypoints

def softmax(x):
    """Transform a 1D array to probability distribution"""
    return np.exp(x-np.max(x)) / np.sum(np.exp(x-np.max(x)))

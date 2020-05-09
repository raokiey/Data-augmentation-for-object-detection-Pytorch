import cv2
import math
import numpy as np
import random

from affine_transform.utils import affine_transformation


class RandomScale(object):
    """Scale the given using a randomly chosen amount in the given range.
    
        Args:
            min_ratio (float): Minimum amount to scale the image.
            max_ratio (float): Maximum amount to scale the image.
        
        Returns:
            dest_img (ndarray): Scaled an image.
            target (dict): Ground truth includes bounding boxes compatible with the scale.
    """
    def __init__(self, min_ratio, max_ratio):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
    
    def __call__(self, image, target):
        image = image.transpose(1,2,0)
        bboxes = target['boxes']
        img_h, img_w = image.shape[:2]
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        scaled_center_h = (img_h / 2) * ratio
        scaled_center_w = (img_w / 2) * ratio
        diff = np.array((scaled_center_w - img_w / 2, scaled_center_h - img_h / 2))
        src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
        dest = src * ratio
        dest -= diff.reshape(1,-1).astype(np.float32)
        affine = cv2.getAffineTransform(src, dest)
        image, bboxes = affine_transformation(image, bboxes, affine, img_w, img_h)
        target['boxes'] = bboxes
        return image, target

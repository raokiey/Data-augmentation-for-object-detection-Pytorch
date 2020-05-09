import cv2
import math
import numpy as np
import random

from affine_transform.utils import affine_transformation


class RandomRotate(object):
    """Rotate the given image using a randomly chosen amount in the given range.
    
        Args:
            min_angle (int): Minimum amount to rotate the image. The unit is degrees.
            max_angle (int): Maximum amount to rotate the image. The unit is degrees.
        
        Returns:
            dest_img (ndarray): Rotated an image.
            target (dict): Ground truth includes bounding boxes compatible with the rotation.
    """
    def __init__(self, min_angle, max_angle):
        self.min_angle = min_angle
        self.max_angle = max_angle
    
    def __call__(self, image, target):
        image = image.transpose(1,2,0)
        bboxes = target['boxes']
        img_h, img_w = image.shape[:2]
        angle = random.randint(self.min_angle, self.max_angle)
        affine = cv2.getRotationMatrix2D((img_w / 2.0, img_h / 2.0), angle, 1.0)
        image, bboxes = affine_transformation(image, bboxes, affine, img_w, img_h)
        target['boxes'] = bboxes
        return image, target

import cv2
import math
import numpy as np
import random

from affine_transform.utils import affine_transformation


class RandomXShear(object):
    """Shear an image along the x-axis with a randomly chosen amount in the given range.
    
        Args:
            min_angle (int): Minimum amount to shear the image. The unit is degrees.
            max_angle (int): Maximum amount to shear the image. The unit is degrees.
        
        Returns:
            dest_img (ndarray): Sheared an image along the x-axis.
            target (dict): Ground truth includes bounding boxes compatible with the shear.
    """
    def __init__(self, min_angle, max_angle):
        self.min_angle = min_angle
        self.max_angle = max_angle
    
    def __call__(self, image, target):
        image = image.transpose(1,2,0)
        bboxes = target['boxes']
        img_h, img_w = image.shape[:2]
        angle = random.randint(self.min_angle, self.max_angle)
        tan = math.tan(math.radians(abs(angle)))
        shear_x = tan * img_w
        src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
        dest = src.copy()
        if angle >= 0:
            dest[:, 0] += (shear_x / img_h * (img_h - src[:,1])).astype(np.float32)
        else:
            dest[:, 0] += (shear_x / img_h * src[:,1]).astype(np.float32)
        dest[:, 0] *= img_w / (img_w + img_h * tan)
        affine = cv2.getAffineTransform(src, dest)
        image, bboxes = affine_transformation(image, bboxes, affine, img_w, img_h)
        target['boxes'] = bboxes
        return image, target


class RandomYShear(object):
    """Shear an image along the y-axis with a randomly chosen amount in the given range.
    
        Args:
            min_angle (int): Minimum amount to shear the image. The unit is degrees.
            max_angle (int): Maximum amount to shear the image. The unit is degrees.
        
        Returns:
            dest_img (ndarray): Sheared an image along the y-axis.
            bboxes (ndarray): Bounding boxes compatible with the shear.
    """
    def __init__(self, min_angle, max_angle):
        self.min_angle = min_angle
        self.max_angle = max_angle
    
    def __call__(self, image, target):
        image = image.transpose(1,2,0)
        bboxes = target['boxes']
        img_h, img_w = image.shape[:2]
        angle = random.randint(self.min_angle, self.max_angle)
        tan = math.tan(math.radians(abs(angle)))
        shear_y = tan * img_h
        src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
        dest = src.copy()
        if angle >=0:
            dest[:, 1] += (shear_y / img_w * (img_w - src[:,0])).astype(np.float32)
        else:
            dest[:, 1] += (shear_y / img_w * src[:,0]).astype(np.float32)
        dest[:, 1] *= img_h / (img_h + img_w * tan)
        affine = cv2.getAffineTransform(src, dest)
        image, bboxes = affine_transformation(image, bboxes, affine, img_w, img_h)
        target['boxes'] = bboxes
        return image, target

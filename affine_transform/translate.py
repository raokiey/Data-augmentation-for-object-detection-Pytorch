import cv2
import numpy as np
import random

from affine_transform.utils import affine_transformation


class RandomTranslate(object):
    """Translate the given image using a randomly chosen amount in the given range.
    
        Args:
            shifts (tuple): The amount to translate the image in the x-axis and y-axis directions.
        
        Returns:
            dest_img (ndarray): Translated an image.
            target (dict): Ground truth includes bounding boxes compatible with the translation.
    """
    def __init__(self, shifts):
        self.shifts = shifts
    
    def __call__(self, image, target):
        image = image.transpose(1,2,0)
        bboxes = target['boxes']
        img_h, img_w = image.shape[:2]
        src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
        dest = src.copy()
        random_shift_x = random.randint(-self.shifts[0], self.shifts[0])
        random_shift_y = random.randint(-self.shifts[1], self.shifts[1])
        shifts_array = np.array((random_shift_x, random_shift_y))
        
        dest = src + shifts_array.reshape(1,-1).astype(np.float32)
        affine = cv2.getAffineTransform(src, dest)
        image, bboxes = affine_transformation(image, bboxes, affine, img_w, img_h)
        target['boxes'] = bboxes
        return image, target


class RandomXTranslate(RandomTranslate):
    """Shear the given image along the x-axis with a randomly chosen amount in the given range.
    
        Args:
            shift (int): The amount to translate the image in the x-axis directions.
        
        Returns:
            dest_img (ndarray): Translated an image.
            bboxes (ndarray): Bounding boxes compatible with the translation.
    """
    def __init__(self, shift):
        
        super().__init__(shift)
        self.shifts = (shift, 0)


class RondomYTranslate(RandomTranslate):
    """Shear the given image along the y-axis with a randomly chosen amount in the given range.
    
        Args:
            shift (int): The amount to translate the image in the y-axis directions.
        
        Returns:
            dest_img (ndarray): Translated an image.
            bboxes (ndarray): Bounding boxes compatible with the translation.
    """
    def __init__(self, shift):
        super().__init__(shift)
        self.shifts = (0, shift)

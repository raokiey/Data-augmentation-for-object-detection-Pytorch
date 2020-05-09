import cv2
import numpy as np
import random

from affine_transform.utils import affine_transformation


class RandomFlip(object):
    """Flip the given image randomly with a given probability and mode.
    
        Args:
            prob (float): Probability of the image being flipped. Default value is 0.5.
            mode (str): Direction to flip.　Specify ``h`` or ``v``.
        
        Returns:
            dest_img (ndarray): Flipped an image.
            target (dict): Ground truth includes bounding boxes compatible with the flip.
    """
    def __init__(self, prob=0.5, mode=None):
        self.prob = prob
        self.mode = mode
    
    def __call__(self, image, target):
        if np.random.rand() < self.prob:
            image = image.transpose(1,2,0)
            img_h, img_w = image.shape[:2]
            bboxes = target['boxes']
            src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
            dest = src.copy()
            if self.mode == 'h':
                dest[:,0] = img_w - src[:,0] 
            elif self.mode == 'v':
                dest[:,1] = img_h - src[:,1]
            affine = cv2.getAffineTransform(src, dest)
            image, transform_bboxes = affine_transformation(image, bboxes, affine, img_w, img_h)
            target['boxes'] = transform_bboxes
            return image, target
        else:
            return image, target
    

class RandomVerticalFlip(RandomFlip):
    """Vertically flip the given image randomly with a given probability.
    
        Args:
            prob (float): Probability of the image being flipped. Default value is 0.5.
        
        Returns:
            dest_img (ndarray): Horizontally flipped an image.
            bboxes (ndarray): Bounding boxes compatible with the vertically flip.
    """
    def __init__(self, prob):
        super().__init__(prob=0.5)
        self.prob = prob
        self.mode = 'v'


class RandomHorizontalFlip(RandomFlip):
    """Horizontally flip the given image randomly with a given probability.
    
        Args:
            prob (float): Probability of the image being flipped. Default value is 0.5.
        
        Returns:
            dest_img (ndarray): Horizontally flipped an image.
            bboxes (ndarray): Bounding boxes compatible with the horizontally flip.
    """
    def __init__(self, prob):
        super().__init__(prob=0.5)
        self.prob = prob
        self.mode = 'h'

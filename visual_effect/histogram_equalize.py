import cv2
import numpy as np


def clahe(image, bboxes):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    hsv_img[:,:,2] = clahe.apply(hsv_img[:,:,2])
    dest_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return dest_img, bboxes

class HistogramEqualize(object):
    """Simple histogram equalization.
        
        Returns:
            dest_img (ndarray): Histogram equalized image.
            target (ndarray): Given target dictionary.
    """
    def __init__(self):
        pass
    
    def __call__(self, image, target):
        image = image.transpose(1,2,0)
        hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) 
        hsv_img[:,:,2] = cv2.equalizeHist(hsv_img[:,:,2])
        dest_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        return dest_img, target


class CLAHE(object):
    """Contrast limited adaptive histogram equalization.

        Args:
            clip_limit (float): Threshold for contrast limiting.
            tile_grid_size (tuple): Size of grid for histogram equalization. 

        Returns:
            dest_img (ndarray): Limited adaptive histogram equalized image.
            target (dict): Given target dictionary.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, image, target):
        image = image.transpose(1,2,0)
        hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) 
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        hsv_img[:,:,2] = clahe.apply(hsv_img[:,:,2])
        dest_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        return dest_img.transpose(2,0,1), target

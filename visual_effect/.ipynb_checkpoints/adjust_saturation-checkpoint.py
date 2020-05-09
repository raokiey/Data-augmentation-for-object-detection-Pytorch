import cv2
import random
import numpy as np


class RandomAdjustSaturation(object):
    """Adjust the saturation of the given image using a randomly chosen amount in the given range.

        Args:
            min_delta (float): Minimum of factor multiplying the saturation values of each pixel.
            max_delta (float): Maximum of factor multiplying the saturation values of each pixel.

        Returns:
            dest_img (ndarray): Saturation adjusted image.
            target (ndarray): Given target dictionary.
    """
    def __init__(self, min_delta=0.95, max_delta=1.05):
        self.min_delta = min_delta
        self.max_delta = max_delta

    def __call__(self, image, target):
        image = image.transpose(1,2,0)
        delta = random.uniform(self.min_delta, self.max_delta)
        hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_img = hsv_img.astype(np.uint16)
        hsv_img[:, :, 1] = hsv_img[:, :, 1] * delta
        hsv_img = hsv_img.astype(np.uint8)
        dest_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        return dest_img.transpose(2,0,1), target

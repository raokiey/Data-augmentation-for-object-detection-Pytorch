import cv2
import random
import numpy as np

class RandomAdjustHue(object):
    """Adjust the hue of the given image using a randomly chosen amount in the given range.
    
        Args:
            min_angle (int): Minimum of the amount added to the hue channel.
            max_angle (int): Maximum of the amount added to the hue channel.
        
        Returns:
            dest_img (ndarray): Hue adjusted image.
            target (dict): Given target dictionary.
    """
    def __init__(self, min_angle=-9, max_angle=9):
        self.min_angle = min_angle
        self.max_angle = max_angle
    
    def __call__(self, image, target):
        image = image.transpose(1,2,0)
        angle = random.randint(self.min_angle, self.max_angle)
        hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_img = hsv_img.astype(np.uint16)
        hsv_img[:, :, 0] = (hsv_img[:, :, 0] + angle) % 180
        hsv_img = hsv_img.astype(np.uint8)
        dest_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        return dest_img.transpose(2,0,1), target

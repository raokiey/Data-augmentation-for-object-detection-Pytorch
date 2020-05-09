# Data augmentation for object detection

Data augmentation for object detection for Pytorch.   
Please see [`implemented_augmentation.ipynb`](implemented_augmentation.ipynb) for each processing.  
Also, please see [`example.ipynb`](example.ipynb) for usage.

## Implemented augmentation processings 
The implemented augmentation procesings are as follows.  

- ### Using affine transformation

| Class name | Effect |  
|---|---|
| RandomRotate | Rotate the given image using a randomly chosen amount in the given range. |
| RandomTranslate | Translate the given image using a randomly chosen amount in the given range. |
| RandomXShear | Shear an image along the x-axis with a randomly chosen amount in the given range. |
| RandomYShear | Shear an image along the y-axis with a randomly chosen amount in the given range. |
| RandomScale | Scale the given using a randomly chosen amount in the given range. |
| RandomHorizontalFlip | Horizontally flip the given image randomly with a given probability. |
| RandomVerticalFlip | Vertically flip the given image randomly with a given probability. |

- ### Changes in appearance  

| Class name | Effect |  
|---|---|
| HistogramEqualize | Simple histogram equalization. |
| CLAHE | Contrast limited adaptive histogram equalization. |
| RandomAdjustBrightness | Adjust the brightness of the given image using a randomly chosen amount in the given range. |
| RandomAdjustHue | Adjust the hue of the given image using a randomly chosen amount in the given range. |
| RandomAdjustSaturation  | Adjust the saturation of the given image using a randomly chosen amount in the given range. |  

## TODO  
- [ ] Perform the processing that uses affine transformation in a batch.　　 
- [ ] Performs the process of changing colors in a batch.  

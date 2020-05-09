import cv2
import numpy as np

def _to_homogeneous_matrix(bboxes):
    """Convert bounding box to homogeneous coordinates.
    
        Args:
            bboxes (list or ndarray): Bounding box as training or test data.
        
        Returns:
            bbox_matrix (ndarray): Bounding box converted to homogeneous coordinates.
    """
    _bboxes = []
    for bbox in bboxes:
        elements = []
        elements.append(list((bbox[0], bbox[1])))
        elements.append(list((bbox[2], bbox[1])))
        elements.append(list((bbox[0], bbox[3])))
        elements.append(list((bbox[2], bbox[3])))
        _bboxes.append(elements)
    _bboxes = np.array(_bboxes)
    bbox_matrix = np.concatenate([_bboxes, np.ones((_bboxes.shape[0],_bboxes.shape[1],1), np.float32)], axis=-1)
    return bbox_matrix

def _convert_format(bboxes):
    """Convert the bounding box to a format compatible with the model.
    
        Args:
            bboxes (ndarray): Affine transformed bounding box.
        
        Returns:
            converted_bboxes (ndarray): Bounding box converted to a format compatible with the model.
    """
    converted_bboxes = []
    for bbox in bboxes:
        elements = []
        elements.append(np.min(bbox[:,0]))
        elements.append(np.min(bbox[:,1]))
        elements.append(np.max(bbox[:,0]))
        elements.append(np.max(bbox[:,1]))
        elements = [int(elem) for elem in elements]
        converted_bboxes.append(elements)
    return np.array(converted_bboxes)

def affine_transformation(image, bboxes, affine, w, h):
    """Affine transform to the given image and given bounding boxes.

        Args:
            image (ndarray): Image for affine transformation.
            bboxes (ndarray): Bounding box corresponding to the image.
            affine (ndarray): Transformation matrix.
            w (int): The width of the image.
            h (int): The height of the image.

        Returns:
            transform_image (ndarray): Affine transformed bounding box.
            within_bboxes (ndarray): Bounding boxes compatible with the affine transformation.
    """
    transform_image = cv2.warpAffine(image, affine, (w, h), cv2.INTER_LANCZOS4)
    bbox_matrix = _to_homogeneous_matrix(bboxes)
    tmp_bboxes = np.tensordot(affine, bbox_matrix.T, 1).T
    transform_bboxes = _convert_format(tmp_bboxes).clip(min=0, max=w)
    flag = (transform_bboxes[:, 0] != transform_bboxes[:, 2])\
                 & (transform_bboxes[:, 1] != transform_bboxes[:, 3])
    within_bboxes = transform_bboxes[flag]
    if within_bboxes.shape[0] == 0:
        return image.transpose(2,0,1), bboxes
    else:
        return transform_image.transpose(2,0,1), within_bboxes

import numpy as np
import os

def toTensor(x):
    x = x.astype(np.float32)
    x /= 127.5
    x -= 1.
    return x

def get_abs_path(path):
    if os.path.isabs(path):
        dataset_dir = path
    else:
        dataset_dir = os.path.join(os.getcwd(), path)
    return dataset_dir

def iou(box1, box2):
    '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
      Args:
        box1: (np.ndarray) bounding boxes, sized [N,4].
        box2: (np.ndarray) bounding boxes, sized [M,4].
      Return:
        (np.ndarray) iou, sized [N,M].
    '''

    N = box1.shape[0]
    M = box2.shape[0]

    # max of left and top, [x1, y1, _, _]
    b1 = np.broadcast_to(np.expand_dims(box1[:,:2], axis=1), (N,M,2))  # [N,2] -> [N,1,2] -> [N,M,2]
    b2 = np.broadcast_to(np.expand_dims(box2[:,:2], axis=0), (N,M,2))  # [M,2] -> [1,M,2] -> [N,M,2]

    lt = np.maximum(b1,b2)

    # min of right and bottom, [_, _, x2, y2]
    b1 = np.broadcast_to(np.expand_dims(box1[:,2:], axis=1), (N,M,2))  # [N,2] -> [N,1,2] -> [N,M,2]
    b2 = np.broadcast_to(np.expand_dims(box2[:,2:], axis=0), (N,M,2))  # [M,2] -> [1,M,2] -> [N,M,2]
    rb = np.minimum(b1,b2)


    # width and height = right - left, bottom - top
    wh = rb - lt  # [N,M,2]
    wh[wh<0] = 0  # clip at 0
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
    area1 = np.broadcast_to(np.expand_dims(area1, axis=1), (N, M)) # [N,] -> [N,1] -> [N,M]
    area2 = np.broadcast_to(np.expand_dims(area2, axis=0), (N, M))  # [M,] -> [1,M] -> [N,M]

    return inter / (area1 + area2 - inter)
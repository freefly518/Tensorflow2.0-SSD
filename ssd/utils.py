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

def nms(bboxes, scores, threshold=0.3):
    '''Non maximum suppression.
    Args:
        bboxes: (ndarray) bounding boxes, sized [N,4].
        scores: (ndarray) bbox scores, sized [N,].
        threshold: (float) overlap threshold.
        mode: (str) 'union' or 'min'.
    Returns:
        keep: (ndarray) selected indices.
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    areas = (x2-x1) * (y2-y1)
    order = np.argsort(scores, axis=0)
    order = order[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break
        xx1 = np.minimum(x1[order[1:]], x1[i])
        yy1 = np.minimum(y1[order[1:]], y1[i])
        xx2 = np.maximum(x2[order[1:]], x2[i])
        yy2 = np.maximum(y2[order[1:]], y2[i])

        w = np.maximum((xx2-xx1), 0)
        h = np.maximum((yy2-yy1), 0)
        inter = w*h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
     

        ids = (ovr<=threshold)
        if len(ids) == 0:
            break
        order = order[1:] 
        order = order[ids]
    return keep
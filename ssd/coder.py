import math
import numpy as np
import itertools
from .anchor import get_anchor_box
from .utils import iou

def encode(boxes, classes, default_boxes, threshold=0.5):
    '''Transform target bounding boxes and class labels to SSD boxes and classes.
      Match each object box to all the default boxes, pick the ones with the
      Jaccard-Index > 0.5:
          Jaccard(A,B) = AB / (A+B-AB)
      Args:
        #obj: image has #obj number of object
        boxes: (np.ndarray) object bounding boxes (xmin,ymin,xmax,ymax) of a image, sized [#obj, 4].
        classes: (np.ndarray) object class labels of a image, sized [#obj,].
        threshold: (float) Jaccard index threshold
      Returns:
        boxes: (np.ndarray) bounding boxes, sized [anchor_num, 4].
        classes: (np.ndarray) class labels, sized [anchor_num,]
    '''
    num_default_boxes = len(default_boxes)
    num_objs = len(boxes)

    # aspect raito to x1, y1, x2, y2
    x1 = np.array(default_boxes[:,:2] - default_boxes[:,2:]/2)
    x2 = np.array(default_boxes[:,:2] + default_boxes[:,2:]/2)
    d_boxse = np.concatenate((x1, x2), axis=1)
    # [#obj,anchor_num]
    iou_box = iou(boxes, d_boxse)

    max_idx = np.argmax(iou_box, axis=0)  # [1,anchor_num]
    max_iou = np.max(iou_box, axis=0) # [1,anchor_num]

    boxes = boxes[max_idx]     # [anchor_num,4]
    variances = [0.1, 0.2]
    cxcy = (boxes[:,:2] + boxes[:,2:])/2 - default_boxes[:,:2]  # [anchor_num,2]
    cxcy /= variances[0] * default_boxes[:,2:]
    wh = (boxes[:,2:] - boxes[:,:2]) / default_boxes[:,2:]      # [anchor_num,2]
    wh = np.log(wh) / variances[1]

    loc =np.concatenate((cxcy, wh), axis=1)  # [anchor_num,4]
    conf = 1 + classes[max_idx]   # [anchor_num,], background class = 0
    conf[max_iou<threshold] = 0       # background
    return loc, conf


def decode(loc, conf, default_boxes, num_classes=21, threshold=0.5):
    '''Transform predicted loc/conf back to real bbox locations and class labels.
      Args:
        loc: (ndarray) predicted loc, sized [anchor_num,4].
        conf: (ndarray) predicted conf, sized [anchor_num,21].
        threshold: (float) threshold for object score
      Returns:
        boxes: (ndarray) bbox locations, sized [#obj, 4].
        labels: (ndarray) class labels, sized [#obj,1].
    '''
    variances = (0.1, 0.2)
    wh = np.exp(loc[:,2:]*variances[1]) * default_boxes[:,2:]
    cxcy = loc[:,:2] * variances[0] * default_boxes[:,2:] + default_boxes[:,:2]
    box_preds = np.concatenate([cxcy-wh/2, cxcy+wh/2], 1)  # [anchor_num,4]

    boxes = []
    labels = []
    scores = []
    num_classes = num_classes
    for i in range(num_classes-1):
        if i == 0:
            # background
            continue
        score = conf[:,i]  # class i corresponds to (i+1) column
        mask = score > threshold

        if not mask.any():
            continue

        # get score
        box = box_preds[mask]
        score = score[mask]

        boxes.append(box)
        labels.append(i)
        scores.append(score)

    return boxes, labels, scores





if __name__ == "__main__":
    box1 = [[0.4900, 0.3024, 0.6000, 0.8054]]
    box2 = [[0.4900, 0.3024, 0.6000, 0.8054], [0.4900, 0.3024, 0.6000, 0.8054]]

    default_box = get_anchor_box()
    default_box = np.array(default_box)
    output = iou(np.array(box1), np.array(box2)).flatten()
    print(output)

    classes = np.array([4, 23, 2])


    loc, conf = encode(np.array(box2), classes, default_box)
    print(loc, conf)
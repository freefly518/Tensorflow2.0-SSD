import tensorflow as tf
import numpy as np
import cv2
from ssd.model import SSD
from ssd.coder import get_anchor_box, decode, nms
from ssd.utils import toTensor
from setting import *
from ssd.voc import VOC_LABELS

if __name__ == "__main__":
    ori_img = cv2.imread("demo/img1.jpg")
    img = ori_img.copy()
    w,h = img.shape[1], img.shape[0]

    img = cv2.resize(img, (300,300))
    img = img.astype(np.float32)

    net = tf.keras.models.load_model(model_path)

    img = toTensor(img)
    img = np.expand_dims(img, axis=0)
    confs, locs  = net(img)

    confs = tf.math.softmax(confs, axis=-1)
    classes = tf.math.argmax(confs, axis=-1)
    scores = tf.math.reduce_max(confs, axis=-1)
    default_box = np.array(get_anchor_box())
   

    boxes, labels, scores = decode(locs.numpy()[0], confs.numpy()[0], default_box, ori_img.shape[1], ori_img.shape[0], threshold=0.9)
    print(scores)
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.9:
            b = list(box)
            print(b, score, label)
            cv2.rectangle(ori_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 2)
            cv2.putText(ori_img, "{}".format(VOC_LABELS[label-1]), (int(b[0]), int(b[1])), cv2.FONT_ITALIC, 0.6, (0, 255, 0), 2)
    cv2.imshow("test.jpg", ori_img)
    cv2.waitKey(0)



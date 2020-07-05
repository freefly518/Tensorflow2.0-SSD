import tensorflow as tf
import os
import numpy as np
import cv2
from .coder import encode
from .anchor import get_anchor_box
class Datagen(object):
    def __init__(self, root, list_file, img_scale_size, train=True, transform=None):
        '''
        Args:
            root: (str) ditectory to images.
            list_file: (str) path to index file.
            train: (boolean) train or test.
            img_scale_size: (int) image scale size
        '''

        self.root = root
        self.train = train
        self.img_scale_size = img_scale_size
        self.dataset = tf.data.TextLineDataset(filenames=list_file)
        self.batch_dataset = self.dataset.batch(batch_size=1)
        self.dataset_list = list(self.batch_dataset.as_numpy_iterator())
        self.transform = transform
        self.default_box = np.array(get_anchor_box())

    def set_batch(self, batch_size):
        self.batch_dataset = self.dataset.batch(batch_size=batch_size)
        self.dataset_list = list(self.batch_dataset.as_numpy_iterator())

    def get_count(self):
        return len(self.dataset_list)

    def get_train_data(self, idx):
        result_img = list()
        result_loc = list()
        result_conf = list()
        for data in self.dataset_list[idx]:
            img_str = bytes.decode(data, encoding="utf-8")

            splited = img_str.strip().split()
            img_path = splited[0]
            num_objs = splited[1]
            box = []
            label = []
            for i in range(int(num_objs)):
                c = splited[2+5*i]
                xmin = splited[3+5*i]
                ymin = splited[4+5*i]
                xmax = splited[5+5*i]
                ymax = splited[6+5*i]
                box.append([float(xmin),float(ymin),float(xmax),float(ymax)])
                label.append(int(c))

            boxes = np.array(box)
            label = np.array(label)
            img = cv2.imread(os.path.join(self.root, img_path))

            # Scale bbox locaitons to [0,1].
            w,h = img.shape[1], img.shape[0]
            boxes /= np.array([w, h, w, h])

            img = cv2.resize(img, (self.img_scale_size, self.img_scale_size))
            img = img.astype(np.float32)

            if self.transform:
                img = self.transform(img)

            loc, conf = encode(boxes, label, self.default_box)
            result_img.append(img)
            result_loc.append(loc)
            result_conf.append(conf)

        return np.array(result_img), np.array(result_loc), np.array(result_conf)

if __name__ == "__main__":
    txt_file = "a.txt"
    dataset_folder = "data/VOC2012/JPEGImages/"

    a = Datagen(dataset_folder, txt_file, 300)
    a.set_batch(4)
    img, b, c = a.get_train_data(0)
    print(img.shape, b.shape, c.shape)

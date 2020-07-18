from ssd.datagen import Datagen
from ssd.model import SSD
from ssd.loss import BoxLoss
from ssd.utils import get_abs_path, toTensor
from setting import *
import tensorflow as tf
import numpy as np
import itertools
import math

from setting import *

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def train_one_step(batch_img, gt_loc, gt_conf):
    with tf.GradientTape() as tape:
        confs, locs = net(batch_img)
        conf_loss, loc_loss = criterion(locs, gt_loc, confs, gt_conf)

        loss = conf_loss + loc_loss
        l2_loss = [tf.nn.l2_loss(t) for t in net.trainable_variables]
        l2_loss = 5e-4 * tf.math.reduce_sum(l2_loss)
        loss += l2_loss

    gradients = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, net.trainable_variables))
    return loss, conf_loss, loc_loss

if __name__ == "__main__":
    txt_file = get_abs_path(output_dataset_txt)
    dataset_folder = get_abs_path(dataset_jpeg_folder)

    dataset = Datagen(dataset_folder, txt_file, image_shape[0], transform=toTensor)
    dataset.set_batch(batch_size)

    dataset_count = dataset.get_count()

    if use_pretrain:
        pretrain_net = tf.keras.models.load_model(model_path)
        net = SSD(num_classes=classes_number,
                    anchor_num=anchor_number,
                    input_shape=image_shape)
        net.set_weights(pretrain_net.get_weights())
    else:
        net = SSD(num_classes=classes_number,
                    anchor_num=anchor_number,
                    input_shape=image_shape)

    net.set_batch_size(batch_size)

    criterion = BoxLoss()
    # optimizer

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=0.9)

    epoc_count = 0
    for epoch in range(num_epochs):
        count = 0
        train_loss = 0
        avg_conf_loss = 0
        avg_loc_loss = 0
        avg_loss = 0
        for i in range(dataset_count):
            batch_img, gt_loc, gt_conf = dataset.get_train_data(i)
            loss, conf_loss, loc_loss = train_one_step(batch_img, gt_loc, gt_conf)

            train_loss += loss.numpy()
            avg_conf_loss += conf_loss.numpy()
            avg_loc_loss += loc_loss.numpy()
            avg_loss += loss.numpy()
            print('Epoch: {}({}/{}) Batch {} | Loss: {:.4f} Conf: {:.4f} Loc: {:.4f} | Train Loss {:.4f}'.format(
                epoch + 1, count, dataset_count, batch_size, avg_loss/(count + 1), avg_conf_loss/(count + 1), avg_loc_loss/(count + 1), conf_loss.numpy()))

            count += 1
            if count % 200 == 0:
                net.save(model_path, save_format='tf')
        dataset.shuffle()
        net.save(model_path, save_format='tf')
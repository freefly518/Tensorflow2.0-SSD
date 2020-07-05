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


def train_one_step(batch_img, gt_conf, gt_loc):
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
                    input_shape=image_shape,
                    batch_size=batch_size)
        net.set_weights(pretrain_net.get_weights())
    else:
        net = SSD(num_classes=classes_number,
                    anchor_num=anchor_number,
                    input_shape=image_shape,
                    batch_size=batch_size)

    net.set_batch_size(batch_size)

    criterion = BoxLoss()
    # optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                                 decay_steps=decay_steps,
                                                                 decay_rate=decay_rate)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule)

    count = 0
    for epoch in range(num_epochs):
        for i in range(dataset_count):
            batch_img, gt_conf, gt_loc = dataset.get_train_data(i)
            loss, conf_loss, loc_loss = train_one_step(batch_img, gt_loc, gt_conf)

            if count % 10 == 0:
                print('Epoch: {}({}/{}) Batch {} | Loss: {:.4f} Conf: {:.4f} Loc: {:.4f}'.format(
                    epoch + 1, count, dataset_count, batch_size, loss.numpy(), conf_loss.numpy(), loc_loss.numpy()))

            count += 1
            if count % 200 == 0:
                net.save(model_path, save_format='tf')

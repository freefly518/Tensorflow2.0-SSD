import tensorflow as tf
from ssd.model import SSD
from setting import *
import numpy as np
pretrain_net = tf.keras.applications.VGG16()
pretrain_net.summary()
pretrain_data = dict()

for layer in pretrain_net.layers:
    if isinstance(layer, tf.python.keras.engine.input_layer.InputLayer):
        continue
    pretrain_layers = layer.get_weights()
    pretrain_data[layer.name] = layer.get_weights()

net = SSD(num_classes=21,
                anchor_num=anchor_number,
                input_shape=image_shape)


fc1_weights, fc1_biases = pretrain_data['fc1']
fc2_weights, fc2_biases = pretrain_data['fc2']


for layer in net.layers:
    if layer.__class__.__name__ == "Conv2D":
        print(layer.name)
        if layer.name in pretrain_data:
            layer.set_weights(pretrain_data[layer.name])
        if layer.name in ['block_conv5_1', 'block_conv5_2', 'block_conv5_3']:
            weights = np.random.choice(
                np.reshape(fc1_weights, (-1,)), (3, 3, 512, 512))
            biases = np.random.choice(
                fc1_biases, (512,))
            layer.set_weights([weights, biases])

        if layer.name == 'block_conv6':
            weights = np.random.choice(
                np.reshape(fc1_weights, (-1,)), (3, 3, 512, 1024))
            biases = np.random.choice(
                fc1_biases, (1024,))
            layer.set_weights([weights, biases])

        if layer.name == 'block_conv7':
            weights = np.random.choice(
                np.reshape(fc1_weights, (-1,)), (1, 1, 1024, 1024))
            biases = np.random.choice(
                fc1_biases, (1024,))
            layer.set_weights([weights, biases])

        if layer.name == 'block_conv8_1':
            weights = np.random.choice(
                np.reshape(fc1_weights, (-1,)), (1, 1, 1024, 1024))
            biases = np.random.choice(
                fc1_biases, (1024,))
            layer.set_weights([weights, biases])

        if layer.name == 'block_conv8_2':
            weights = np.random.choice(
                np.reshape(fc1_weights, (-1,)), (3, 3, 1024, 1024))
            biases = np.random.choice(
                fc1_biases, (1024,))
            layer.set_weights([weights, biases])

        if layer.name == 'block_conv9_1':
            weights = np.random.choice(
                np.reshape(fc2_weights, (-1,)), (1, 1, 1024, 128))
            biases = np.random.choice(
                fc2_biases, (128,))
            layer.set_weights([weights, biases])

        if layer.name == 'block_conv9_2':
            weights = np.random.choice(
                np.reshape(fc2_weights, (-1,)), (3, 3, 128, 256))
            biases = np.random.choice(
                fc2_biases, (256,))
            layer.set_weights([weights, biases])

        if layer.name == 'block_conv10_1':
            weights = np.random.choice(
                np.reshape(fc2_weights, (-1,)), (1, 1, 256, 128))
            biases = np.random.choice(
                fc2_biases, (128,))
            layer.set_weights([weights, biases])

        if layer.name == 'block_conv10_2':
            weights = np.random.choice(
                np.reshape(fc2_weights, (-1,)), (3, 3, 128, 256))
            biases = np.random.choice(
                fc2_biases, (256,))
            layer.set_weights([weights, biases])

        if layer.name == 'block_conv11_1':
            weights = np.random.choice(
                np.reshape(fc2_weights, (-1,)), (1, 1, 256, 128))
            biases = np.random.choice(
                fc2_biases, (128,))
            layer.set_weights([weights, biases])

        if layer.name == 'block_conv11_2':
            weights = np.random.choice(
                np.reshape(fc2_weights, (-1,)), (3, 3, 128, 256))
            biases = np.random.choice(
                fc2_biases, (256,))
            layer.set_weights([weights, biases])

        if layer.name == 'block_conv12_1':
            weights = np.random.choice(
                np.reshape(fc2_weights, (-1,)), (1, 1, 256, 128))
            biases = np.random.choice(
                fc2_biases, (128,))
            layer.set_weights([weights, biases])

        if layer.name == 'block_conv12_2':
            weights = np.random.choice(
                np.reshape(fc2_weights, (-1,)), (3, 3, 128, 256))
            biases = np.random.choice(
                fc2_biases, (256,))
            layer.set_weights([weights, biases])




net.save(model_path, save_format='tf')
print("Convert vgg16 finish, model_path: {}".format(model_path))

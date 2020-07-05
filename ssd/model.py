
import tensorflow as tf
from tensorflow.keras import layers

class SSD(tf.keras.Model):
    def __init__(self, num_classes=21,
                        anchor_num=[4,6,6,6,4,4],
                        input_shape=(300,300,3),
                        batch_size=1,
                        **kwargs):
        super(SSD, self).__init__()
        self.num_classes = num_classes # classes + background
        self.anchor_num = anchor_num
        self.batch_size = batch_size
        self.base = self.vgg_16()
        # layer norm
        self.layer_norm = layers.LayerNormalization(axis=1, epsilon=0.001, scale=True) # 38
        self.conv5_1 = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None))
        self.conv5_2 = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None))
        self.conv5_3 = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None))

        self.conv6 = layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=1, padding="same", dilation_rate=(6, 6), activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None))

        self.conv7 = layers.Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None))

        self.conv8_1 = layers.Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None))
        self.conv8_2 = layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=2, padding="same", activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None))

        self.conv9_1 = layers.Conv2D(filters=128, kernel_size=(1, 1), activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None))
        self.conv9_2 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", strides=2, activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None))

        self.conv10_1 = layers.Conv2D(filters=128, kernel_size=(1, 1), activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None))
        self.conv10_2 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding="valid", activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None))

        self.conv11_1 = layers.Conv2D(filters=128, kernel_size=(1, 1), activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None))
        self.conv11_2 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding="valid", activation='relu', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None))

        self.loc_layers = list()
        self.conf_layers = list()

        # mutli box
        for i in range(len(self.anchor_num)):
            self._predict_layer(k=i)

        self.input_layer = tf.keras.Input(shape=input_shape, batch_size=batch_size)

        self.out = self.call(self.input_layer)

        super(SSD, self).__init__(
            inputs=self.input_layer,
            outputs=self.call(self.input_layer),
            **kwargs)
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def vgg_16(self, training=True):
        '''VGG16 layers.'''
        vgg = tf.keras.Sequential()
        vgg.add(layers.Conv2D(
            64, (3, 3), activation='relu', padding='same', name='block1_conv1'))
        vgg.add(layers.Conv2D(
            64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
        vgg.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool'))

        # Block 2
        vgg.add(layers.Conv2D(
            128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
        vgg.add(layers.Conv2D(
            128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
        vgg.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool'))

        # Block 3
        vgg.add(layers.Conv2D(
            256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
        vgg.add(layers.Conv2D(
            256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
        vgg.add(layers.Conv2D(
            256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
        vgg.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool'))

        # Block 4
        vgg.add(layers.Conv2D(
            512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
        vgg.add(layers.Conv2D(
            512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
        vgg.add(layers.Conv2D(
            512, (3, 3), activation='relu', padding='same', name='block4_conv3'))

        return vgg

    def _predict_layer(self, k):
        self.conf_layers.append(tf.keras.layers.Conv2D(filters=self.anchor_num[k] * self.num_classes, kernel_size=(3, 3), strides=1, padding="same"))
        self.loc_layers.append(tf.keras.layers.Conv2D(filters=self.anchor_num[k] * 4, kernel_size=(3, 3), strides=1, padding="same"))

    def call(self, inputs, training=True):

        confs = list()
        locs = list()
        x = self.base(inputs)

        # conv4_3
        conv4_3_conf, conv4_3_loc = self.conf_layers[0](self.layer_norm(x)), self.loc_layers[0](self.layer_norm(x))
        confs.append(conv4_3_conf)
        locs.append(conv4_3_loc)
        f_max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")

        x = f_max_pool(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)

        f_max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding="same")
        x = f_max_pool(x)

        x = self.conv6(x)
        x = self.conv7(x)

        conv_7_conf, conv_7_loc = self.conf_layers[1](x), self.loc_layers[1](x)

        confs.append(conv_7_conf)
        locs.append(conv_7_loc)

        x = self.conv8_1(x)
        x = self.conv8_2(x)

        conv_8_2_conf, conv_8_2_loc = self.conf_layers[2](x), self.loc_layers[2](x)

        confs.append(conv_8_2_conf)
        locs.append(conv_8_2_loc)

        x = self.conv9_1(x)
        x = self.conv9_2(x)

        conv_9_2_conf, conv_9_2_loc = self.conf_layers[3](x), self.loc_layers[3](x)

        confs.append(conv_9_2_conf)
        locs.append(conv_9_2_loc)

        x = self.conv10_1(x)
        x = self.conv10_2(x)

        conv_10_2_conf, conv_10_2_loc = self.conf_layers[4](x), self.loc_layers[4](x)

        confs.append(conv_10_2_conf)
        locs.append(conv_10_2_loc)


        x = self.conv11_1(x)
        x = self.conv11_2(x)

        conv_11_2_conf, conv_11_2_loc = self.conf_layers[5](x), self.loc_layers[5](x)

        confs.append(conv_11_2_conf)
        locs.append(conv_11_2_loc)

        confs = [tf.reshape(i, [self.batch_size, -1, self.num_classes]) for i in confs]
        locs = [tf.reshape(i, [self.batch_size, -1, 4]) for i in locs]

        confs = tf.concat(confs, axis=1)
        locs = tf.concat(locs, axis=1)
        return confs, locs


if __name__ == "__main__":
    net = SSD(num_classes=21,
                anchor_num=[4,6,6,6,4,4],
                input_shape=(300,300,3),
                batch_size=1)
    import numpy as np
    inputs = np.ones((1,300,300,3), dtype=np.float32)
    net.summary()
    conf, loc = net(inputs)
    print(conf.shape, loc.shape)
    print("PASS")
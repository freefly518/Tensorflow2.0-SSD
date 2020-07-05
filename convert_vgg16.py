import tensorflow as tf
from ssd.model import SSD
from setting import *

pretrain_net = tf.keras.applications.VGG16()
pretrain_net.summary()
pretrain_data = dict()

for layer in pretrain_net.layers:
    if isinstance(layer, tf.python.keras.engine.input_layer.InputLayer):
        continue
    pretrain_layers = layer.get_weights()
    if layer.__class__.__name__ == "Conv2D":
        print(layer.name)
        pretrain_data[layer.name] = layer.get_weights()

net = SSD(num_classes=21,
                anchor_num=anchor_number,
                input_shape=image_shape,
                batch_size=1)

for layer in net.layers:
    if layer.__class__.__name__ == "Conv2D":
        print(layer.name)
        if layer.name in pretrain_data:
            layer.set_weights(pretrain_data[layer.name])

net.summary()
net.save(model_path, save_format='tf')
print("Convert vgg17 finish, model_path: {}".format(model_path))

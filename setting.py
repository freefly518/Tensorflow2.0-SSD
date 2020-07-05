image_shape = (300, 300, 3)

# train
num_epochs = 40000
batch_size = 16
learning_rate =0.0001
decay_steps= 20000
decay_rate = 0.9


# model save path(use savedmodel)
model_path = "models/ssd"
use_pretrain = True

# voc data
dataset_anno_dir= "data/VOC2012/Annotations/"
dataset_jpeg_folder = "data/VOC2012/JPEGImages/"
output_dataset_txt="voc_list.txt"


VOC_LABELS = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]
classes_number = len(VOC_LABELS) + 1 # background

# anchor box
anchor_number = [4,6,6,6,4,4]
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
feature_map_size = [38, 19, 10, 5, 3, 1]
steps = [8, 16, 32, 64, 100, 300]
sizes = [30, 60, 111, 162, 213, 264, 315]




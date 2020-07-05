import math
import numpy as np
import itertools

def get_anchor_box(image_size=300.0, steps=[8, 16, 32, 64, 100, 300],
                  sizes=[30, 60, 111, 162, 213, 264, 315], aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                  feature_map_sizes=[38, 19, 10, 5, 3, 1]):
    '''Transform predicted loc/conf back to real bbox locations and class labels.
      Args:
        image_size: (float) image size.
        steps: (list[int])
        sizes: (list[int])
        aspect_ratios: (list[list[int]]) anchor box aspect ratio
        feature_map_sizes: (list[int]) each feature map size
      Returns:
        boxes: (ndarray) bbox locations, sized [#obj, 4].
    '''
    scale = image_size
    steps = [s / scale for s in steps]
    sizes = [s / scale for s in sizes]
    aspect_ratios = aspect_ratios
    feature_map_sizes = feature_map_sizes

    num_layers = len(feature_map_sizes)

    boxes = []
    for i in range(num_layers):
        fmsize = feature_map_sizes[i] # feature map size
        for h,w in itertools.product(range(fmsize), repeat=2):
            cx = (w + 0.5)*steps[i]
            cy = (h + 0.5)*steps[i]

            s = sizes[i]
            boxes.append([cx, cy, s, s])

            s = math.sqrt(sizes[i] * sizes[i+1])
            boxes.append([cx, cy, s, s])

            s = sizes[i]
            for ar in aspect_ratios[i]:
                boxes.append([cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)])
                boxes.append([cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)])

    return boxes
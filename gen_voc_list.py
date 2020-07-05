from setting import *
from ssd.voc import VOC_Dataset
from ssd.utils import get_abs_path
import os


if __name__ == "__main__":

    dataset_dir = get_abs_path(dataset_anno_dir)

    print(dataset_dir)
    voc = VOC_Dataset(dataset_dir)
    voc.read_all_xml()
    voc.write_to_txt(output_dataset_txt)
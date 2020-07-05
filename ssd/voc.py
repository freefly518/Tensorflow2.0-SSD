import glob
import os
import xml.etree.ElementTree as ET


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

class ImageData(object):
    def __init__(self, image_name):
        self.img_name = image_name

        # object [class, xmin, ymin, xmax, ymax]
        self.objects = list()

    def add_object(self, classes, xmin, ymin, xmax, ymax):
        self.objects.append([classes, xmin, ymin, xmax, ymax])


class VOC_Dataset():
    def __init__(self, xml_dir):
        if not os.path.isdir(xml_dir):
            raise RuntimeError("{} is not dir".format(xml_dir))

        self.xml_dir = xml_dir
        self.imgs = list()


    def read_all_xml(self):
        xmlfiles = []
        for xmlfile in glob.glob("{}/*.xml".format(self.xml_dir)):
            tree = ET.parse(xmlfile)
            root = tree.getroot()
            filename = tree.find("filename").text
            print(filename)

            # Create Image Data to save object
            img_data = ImageData(filename)

            folder = tree.find("folder").text
            objects = tree.findall("object")
            for i in objects:
                classes = i.find("name").text
                box = i.find("bndbox")
                xmax = box.find("xmax").text
                xmin = box.find("xmin").text
                ymax = box.find("ymax").text
                ymin = box.find("ymin").text

                # index 0 is background, here we add 1
                img_data.add_object(VOC_LABELS.index(classes), xmin, ymin, xmax, ymax)
            self.imgs.append(img_data)

    def write_to_txt(self, txt):
        with open(txt, "w") as f:
            for i in self.imgs:
                jpg_name = i.img_name
                number = len(i.objects)
                line = "{} {}".format(jpg_name, number)
                for o in i.objects:
                    line = "{} {} {} {} {} {}".format(line, o[0], o[1], o[2], o[3], o[4])
                f.write("{}\n".format(line))
        print("write voc data to {} finish!".format(txt))

if __name__ == "__main__":
    xml_dir = "data/VOC2012/Annotations/"
    a = VOC_Dataset(xml_dir)
    a.read_all_xml()
    a.write_to_txt("a.txt")
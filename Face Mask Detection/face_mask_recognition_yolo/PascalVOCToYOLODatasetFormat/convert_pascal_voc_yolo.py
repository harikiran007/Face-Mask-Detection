import glob
import os
import xml.etree.ElementTree as ET
from os import getcwd

dirs = ['JPEGImages', 'JPEGImages']
classes = ['face', 'face_mask']
count = 0


# This Method is used to get all image directory path
def getImagesInDir(dir_path):
    list_of_images = []
    for filename in glob.glob(dir_path + '/*.jpg'):
        list_of_images.append(filename)

    return list_of_images


# This method is used to convert the dimensions into yolo format
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


# This method will read the Pascal XML format and convert into yolo format
def convert_annotation(dir_path, output_path, image_path, count):
    imageFileName = os.path.basename(image_path)
    image_file_name_noExt = os.path.splitext(imageFileName)[0]

    input_file = open(dir_path + '/' + image_file_name_noExt + '.xml')
    output_file = open(output_path + image_file_name_noExt + '.txt', 'w')
    tree = ET.parse(input_file)
    root = tree.getroot()
    size = root.find('size')
    try:
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            output_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        input_file.close()
        os.remove(dir_path + '/' + image_file_name_noExt + '.xml')
    except:
        count = count + 1
        print("Unable to convert for: '" + imageFileName + "' file no, '" + count + "'")
        input_file.close()
        output_file.close()
        os.remove(dir_path + '/' + image_file_name_noExt + '.xml')
        os.remove(dir_path + '/' + image_file_name_noExt + '.jpg')
        os.remove(output_path + image_file_name_noExt + '.txt')


cwd = getcwd()

for dataset_directory_path in dirs:
    dataset_path = cwd + '/' + dataset_directory_path
    output_path = dataset_path + '/yolo/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image_paths = getImagesInDir(dataset_path)
    list_file = open(dataset_path + '.txt', 'w')

    for image_path in image_paths:
        list_file.write(image_path + '\n')
        convert_annotation(dataset_path, output_path, image_path, count)
    list_file.close()

    print("Conversion completed for input data set images: " + dataset_directory_path)

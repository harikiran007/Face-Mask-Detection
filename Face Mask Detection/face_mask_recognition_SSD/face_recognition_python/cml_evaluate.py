# -*- coding:utf-8 -*-
import cv2
import time
import argparse
from xml.dom.minidom import parse
import os

import numpy as np
from PIL import Image
from keras.models import model_from_json
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.keras_loader import load_keras_model, keras_inference

model = load_keras_model('models/face_mask_detection.json', 'models/face_mask_detection.hdf5')

feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)


anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}

def readFile(pathTemp):
    pathArray=pathTemp.split(".")
    pathTemp=pathArray[0]+".xml"
    domTree = parse(pathTemp)
    rootNodes = domTree.documentElement
    # print(rootNodes.nodeName)
    faceMaskobject = rootNodes.getElementsByTagName("object")[0]
    return faceMaskobject.getElementsByTagName("name")[0].childNodes[0].data
def inference(image,imgPath,file,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(160, 160),
              draw_result=True,
              show_result=True
              ):
  
   
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)

    y_bboxes_output, y_cls_output = keras_inference(model, image_exp)
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )
    srcClassName=readFile(imgPath)
    pathArray=imgPath.split("/")

    if(len(keep_idxs)>1):
        return
    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        class_name="face_mask"
        if(class_id==1):
            class_name="face"
        print("predictClass:%s,confidence：%f,srcClassName:%s,imageName:%s"% (class_name,conf,srcClassName,pathArray[len(pathArray)-1]) )
        file.write("{} {} {} {}\n".format(class_name,conf,srcClassName,pathArray[len(pathArray)-1]))
        return


if __name__ == "__main__":
    paths=[]
    dir='/Users/lskmac/Downloads/FaceMaskDataset/val/'
    for root, dirs, files in os.walk(dir):
        for f in files:
            if(f.find("jpg")!=-1):
                jpgPath=os.path.join(root, f)
                paths.append(jpgPath)
    f=open("/Users/lskmac/Downloads/numbers.txt","a")
    # paths=['/Users/lskmac/Downloads/FaceMaskDataset/val/test_00000102.jpg','/Users/lskmac/Downloads/FaceMaskDataset/val/test_00000112.jpg']
    for index in range(len(paths)):
        imgPath = paths[index]
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inference(img, imgPath,f,show_result=True, target_shape=(260, 260),)
    f.close()

   
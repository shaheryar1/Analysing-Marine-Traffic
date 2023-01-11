from flask import Flask, render_template, json, request
from WebView.ObjectsApp.DAL.VideoDAL import VideoDAL
from config import Config, MySQL_DB
import pandas as pd

app = Flask(__name__)
app.config.from_object(Config)
MySQL_DB.mysql.init_app(app)
from utils.ImageUtitls import non_max_suppression_fast
import config
from WebView.ObjectsApp.DAL.VideoDAL import VideoDAL
from utils.ImageUtitls import drawBoxes
# import miscellaneous modules
import matplotlib.pyplot as plt
from utils.ImageUtitls import nms
from Color.Color_Extraction import extract_colors, hex_to_rgb
import time
import dlib
import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import time
# import darknet as dn
from darknet.python import darknet as dn


def load_yolo(weights_path, cfg_path, names_path):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    classes = []
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]

			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    print(len(boxes))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    return img
def start_video(video_path):
    model, classes, colors, output_layers =load_yolo(model_weights, model_cfg, classes_file)
    cap = cv2.VideoCapture(video_path)
    while True:
        _, frame = cap.read()
        temp=frame.copy()
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        img=draw_labels(boxes, confs, colors, class_ids, classes, temp)
        cv2.imshow('a', img)

        key = cv2.waitKey(1)

        if key == 27:
            break
    cap.release()

test_videos_base_path = '/home/shaheryar/Desktop/Projects/oceans11/TestVideos'
model_cfg = 'darknet/Marrine_Vessel_cfg/yolov3.cfg'
model_weights = 'darknet/backup/yolov3_final.weights'
meta_data = 'darknet/Marrine_Vessel_cfg/marrine-vessel.data'
classes_file = 'darknet/data/vessel.names'

if __name__ == '__main__':

    start_video('TestVideos/sail1.MOV')





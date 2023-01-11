from flask import Flask, render_template, json, request
from WebView.ObjectsApp.DAL.VideoDAL import VideoDAL
from config import Config, MySQL_DB

app = Flask(__name__)
app.config.from_object(Config)
MySQL_DB.mysql.init_app(app)



import keras
import config
from WebView.ObjectsApp.DAL.VideoDAL import VideoDAL
from OCR.ocr import extractShipName
# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.colors import label_color
from utils.ImageUtitls import drawBoxes, non_max_suppression
# import miscellaneous modules
import matplotlib.pyplot as plt
from utils.ImageUtitls import nms,non_max_suppression_fast
from Color.Color_Extraction import extract_colors,hex_to_rgb
import cv2
import os
import numpy as np
import time
import dlib
import argparse
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


# Change these paths according your system
output_dir = '/home/shaheryar/Desktop/Projects/oceans11/Test Images Results/retina_net'
MODEL_PATH='/home/shaheryar/Desktop/Projects/oceans11/snapshots/resnet50_csv_07.h5'
labels_to_names=config.labels_to_name

def parseArgument():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True,
                    help="path to input image")

    ap.add_argument("--imagedir", required=False,
                    help="path to input image directory")
    ap.add_argument( "--model", required=True,
                    help="path to trained keras model")
    ap.add_argument( "--video", required=False,
                    help="path to input video")

    ap.add_argument("--output",default='inference_results',required=True,
                    help="path to output image")
    args = ap.parse_args()
    MODEL_PATH=args['model']
    output_dir=args['output']
    return args

def get_session():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def loadModel(path):
    keras.backend.tensorflow_backend.set_session(get_session())
    model = models.load_model(path, backbone_name='resnet50')
    model = models.convert_model(model)
    print(model.summary())
    return model

def detection_image(filepath, model,threshold=0.8):
    image = read_image_bgr(filepath)
    #     print(image.shape)
    #     print(image)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)
    # correct for image scale
    boxes /= scale

    box_list=[]
    scores_list=[]
    labels_list=[]

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < threshold:
            break
        rect = box.astype(int)
        x1, y1, x2, y2 = rect
        box_list.append((x1,y1,x2,y2))
        scores_list.append(score)
        labels_list.append(label)

        # scores are sorted so we can break

    print("before nms",len(box_list))
    print(labels_list)
    print(scores_list)
    idx_list=non_max_suppression(np.array(box_list),0.5)
    print(idx_list)
    print("after nms",len(idx_list))
    for idx in idx_list:

        box   = box_list[idx]
        score = scores_list[idx]
        label = labels_list[idx]

        box_color = (255, 190, 99)


        x1, y1, x2, y2 = box
        try:
            # ship_name = str(extractShipName(draw[y1:y2,x1:x2]))
            # caption = labels_to_names[label] + " - " + str(round(score * 100, 2)) + " - Name : " + ship_name


            bottom_roi=draw[int(y1+(y2-y1)/2):y2,x1:x2]
            dominant_colors=list(extract_colors(bottom_roi,2))
            draw=cv2.rectangle(draw,(x2 - 100,y1 - 30),(x2 - 50,y1),hex_to_rgb(dominant_colors[0]),cv2.FILLED)
            draw=cv2.rectangle(draw, (x2 - 50, y1 - 30), (x2 , y1), hex_to_rgb(dominant_colors[1]), cv2.FILLED)

            caption = labels_to_names[label]+" - "+str(round(score*100,2))

            draw = drawBoxes(draw, (x1, y1, x2, y2), box_color, caption)

            # draw = cv2.rectangle(draw, (x1, y1 - 30), (x1 + 200, y1), box_color, cv2.FILLED)

            # draw = draw_caption(draw, (x1, y1, x2, y2), caption)
            file, ext = os.path.splitext(filepath)
            image_name = file.split('/')[-1] + ext
            output_path = os.path.join(output_dir, image_name)
            draw_conv = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_path, draw_conv)

        except:
            pass



def take_snapshot(video_id,class_name,snapshot,confidence,dominent_colors=None,ship_name=""):
    dal = VideoDAL()
    print(type(snapshot))
    try:
        is_success, im_buf_arr = cv2.imencode(".jpg",snapshot)
        byte_im = im_buf_arr.tobytes()
    except:
        return
    video = {}
    video['video_id'] = video_id
    video['category'] = class_name
    video['confidence'] = confidence
    video['snapshot'] = byte_im
    if(dominent_colors is not None):
        video['color1'] = dominent_colors[0]
        video['color2'] = dominent_colors[1]
    else:
        video['color1'] = "#ffffff"
        video['color2'] = "#ffffff"
    video['ship_name'] = ""
    video['start_time'] = "00:12:00"
    video['span'] = "00:14:20"
    result=dal.insertInference(video)
    if(result==True):
        print("Snapshot taken")
    else:
        print("Cant save snapshot")

def detection_video(video_path, model,output_path="video_results/test_video1.avi",threshold=0.4):
    # create video with id in DB
    dal=VideoDAL()
    video_id = len(dal.getAllVideos())+1;
    name =  str.split(video_path,'/')[-1]
    dal.insertVideo(video_id,name)


    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")


    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

    i = 0;
    start = time.time()
    captions = [];
    trackers=[];
    t = dlib.correlation_tracker()
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()


    # Read until video is completed
    print("Video fps :",fps)

    while (cap.isOpened()):
        ret, frame = cap.read()
        if(frame is None):
            return
        t1 = cv2.getTickCount()
        temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret == True:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Display the resulting frame
            image = preprocess_image(image)
            image, scale = resize_image(image)


            if(i%int((fps*5))==0):
                captions = [];
                trackers = [];
                boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
                boxes /= scale

                for box, score, label in zip(boxes[0], scores[0], labels[0]):
                    box_color = label_color(label)
                    # scores are sorted so we can break
                    if score < threshold:
                        break

                    rect = box.astype(int)
                    x1, y1, x2, y2 = rect
                    # box_color=(255, 190, 99)
                    caption = labels_to_names[label] + " - " + str(round(score * 100)) + "%"


                    rect = dlib.rectangle(x1, y1, x2, y2)
                    t = dlib.correlation_tracker()
                    t.start_track(temp_frame, rect)
                    # update our set of trackers and corresponding class
                    # labels
                    captions.append(caption)
                    trackers.append(t)
		    

                    print("Taking snapshot at", i / int(fps))
                    bottom_roi = frame[int(y1 + (y2 - y1) / 2):y2, x1:x2]
                    bottom_roi = cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2RGB)
     		    # Extract colors
                    dominent_colors = list(extract_colors(bottom_roi, 2))
                    class_name = caption.split("-")[0]
                    score = caption.split("-")[1]
                    score = float(score[0:-1])
                    roi = frame[y1:y2, x1:x2]

                    take_snapshot(video_id, class_name, roi,
                                  confidence=score, dominent_colors=dominent_colors)
                    frame = drawBoxes(frame, (x1, y1, x2, y2), box_color, caption)
               

            else:
                for (t, caption) in zip(trackers, captions):
                    # update the tracker and grab the position of the tracked
                    # object
                    t.update(temp_frame)
                    pos = t.get_position()
                    # unpack the position object
                    x1 = int(pos.left())
                    y1 = int(pos.top())
                    x2 = int(pos.right())
                    y2 = int(pos.bottom())
                    frame = drawBoxes(frame, (x1, y1, x2, y2), box_color, caption)
            # if (i % int(fps*10) == 0):
            #
            #     print("Taking snapshot at",i/int(fps))
            #     bottom_roi = frame[int(y1 + (y2 - y1) / 2):y2, x1:x2]
            #     bottom_roi=cv2.cvtColor(bottom_roi,cv2.COLOR_BGR2RGB)
            #     dominent_colors = list(extract_colors(bottom_roi, 2))
            #     class_name=caption.split("-")[0]
            #     score=caption.split("-")[1]
            #     score=float(score[0:-1])
            #     roi= frame[y1 - 30:y2 + 30, x1 - 20:x2 + 20]
            #
            #     take_snapshot(video_id, class_name,roi,
            #                   confidence=score, dominent_colors=dominent_colors)

        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0),
                    2, cv2.LINE_AA)

        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1
        out.write(frame)
        i=i+1;
        # frame=cv2.resize(frame,(0,0),fx=0.75,fy=0.75)
        # cv2.imshow('Frame', frame)
        if(i%int(fps)==0):
            print("Processed ",str(int(i/fps)),"seconds")

        # Press Q on keyboard to  exit
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break
        # else:
        #     pass
    # When everything done, release the video capture object
    cap.release()
    out.release()
    # out.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':



    model=loadModel(MODEL_PATH)
    input_path ='/home/shaheryar/Desktop/Projects/Marrine-Vessel-Detection/Dataset/Annotated_Dataset/Tanker/Test'
    test_videos_base_path = '/home/shaheryar/Desktop/Projects/oceans11/TestVideos'

    # detection_video(input_path,model,'video_results/IMG_3890.MOV')
    # output_path = '/home/shaheryar/Desktop/Projects/oceans11/Video Results/retina_net'
    # for v in os.listdir(test_videos_base_path):
    #     print(v)
    #     if (len(str.split(v, '.')) == 2):
    #         detection_video(os.path.join(test_videos_base_path,v),model,os.path.join(output_path,v),threshold=0.5)

    test_imgs = os.listdir(input_path)
    for img_name in test_imgs:
        if(img_name[-3:]=='jpg'):
            detection_image(os.path.join(input_path, img_name), model,threshold=0.7)




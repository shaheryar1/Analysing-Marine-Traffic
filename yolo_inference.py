import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import time
from utils.ImageUtitls import non_max_suppression_fast
from utils.ImageUtitls import drawBoxes
# import darknet as dn
from darknet.python import darknet as dn

from pdb import set_trace

def detection_image(image_path,net,meta,output_path,threshold=0.3):
    frame = cv2.imread(image_path)
    now = time.time()
    r = dn.detect(net, meta, image_path.encode('utf-8'), thresh=threshold)
    print(time.time()-now)

    for i, object in enumerate(r):
        # print(i,p)
        class_name = (str(object[0])[2:-1])
        # name = class_name + "-" + str(round(object[1] * 100)) + "%"
        score = str(round(object[1] * 100))
        rect = object[2]
        centerX, centerY, w, h = rect
        w = int(w)
        h = int(h)
        x1 = int(centerX - w / 2)
        y1 = int(centerY - h / 2)
        x2 = x1 + w
        y2 = y1 + h

        # rect = box.astype(int)
        # x1, y1, x2, y2 = rect
        box_color = (255, 190, 99)
        caption = class_name + " - " + score + "%"
        frame = drawBoxes(frame, (x1, y1, x2, y2), box_color, caption)
    img_name = str.split(image_path,'/')[-1]
    # print(img_name)
    cv2.imwrite(os.path.join(output_path,img_name),frame)

def decode_yolo_detections(detections):
    boxes = []
    scores = []
    classes = []
    for j, object in enumerate(detections):
        # print(i,p)
        class_name = (str(object[0])[2:-1])
        # name = class_name + "-" + str(round(object[1] * 100)) + "%"
        score = str(round(object[1] * 100))
        rect = object[2]
        centerX, centerY, w, h = rect
        w = int(w)
        h = int(h)
        x1 = int(centerX - w / 2)
        y1 = int(centerY - h / 2)
        x2 = x1 + w
        y2 = y1 + h
        boxes.append((x1, y1, x2, y2))
        classes.append(class_name)
        scores.append(score)
    return boxes,scores,classes


def detection_video(video_path, net, meta,output_path, threshold=0.3):
    #
    # dal = VideoDAL()
    #
    # video_id = len(dal.getAllVideos()) + 1;
    # name = str.split(video_path, '/')[-1]
    # dal.insertVideo(video_id, name)
    print(output_path)
    cap = cv2.VideoCapture(video_path)
    start_frame_number = 55000
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    (grabbed, frame) = cap.read()
    fshape = frame.shape
    fheight = fshape[0]
    fwidth = fshape[1]

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path,fourcc, 24.0, (fwidth,fheight))
    i = 0;


    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Read until video is completed

    print("Video fps :", fps)
    try:
        while (cap.isOpened()):
            ret, frame = cap.read()

            t1 = cv2.getTickCount()
            if(frame is None):
                return
            temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if ret == True:
                cv2.imwrite('a.jpg', frame)
                now = time.time()
                detections = dn.detect(net, meta, 'a.jpg'.encode('utf-8'), thresh=threshold)
                print(detections)
                print(time.time()-now)
                boxes,scores,classes=decode_yolo_detections(detections)

                nms_idx=non_max_suppression_fast(np.array(boxes),overlapThresh=0.3)
                for idx in nms_idx:
                    box_color=(255, 190, 99)
                    class_name=classes[idx]
                    x1,y1,x2,y2=boxes[idx]
                    score= scores[idx]
                    caption = class_name + " - " + score + "%"
                    frame = drawBoxes(frame, (x1, y1, x2, y2), box_color, caption)

                t2 = cv2.getTickCount()
                time1 = (t2 - t1) / freq
                frame_rate_calc = 1 / time1
                # out.write(frame)
                i = i + 1;
                out.write(frame)
                frame=cv2.resize(frame,(0,0),fx=0.75,fy=0.75)
                cv2.imshow('Frame', frame)
                if (i % int(fps) == 0):
                    print("Processed ", str(int(i / fps)), "seconds")

                # Press Q on keyboard to  exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                else:
                    pass
            # When everything done, release the video capture object
    finally:
        cap.release()
        out.release()
        # out.release()
        # Closes all the frames
        cv2.destroyAllWindows()



def run_on_validation(net,meta,validation_file_path,ouput_path):
    f= open(validation_file_path,'r')

    count = 100
    img_path_list=f.readlines()

    for i in range(0,count):
        idx = int(np.random.randint(0,len(img_path_list)))

        img_path=str.strip(img_path_list[idx])
        i=i+1;
        detection_image(img_path,net,meta,output_path=ouput_path)
        print("Detected",i,"images")
    f.close()



#######################################
# Mo
#######################################
test_videos_base_path='/home/shaheryar/Desktop/Projects/oceans11/TestVideos'
model_cfg='darknet/Marrine_Vessel_cfg/yolov3.cfg'
meta_data='darknet/Marrine_Vessel_cfg/marrine-vessel.data'
model_weights='darknet/yolov3_final.weights'
classes_file = 'darknet/data/vessel.names'

#######################################
# Author
#######################################
model_cfg='darknet/cfg/yolov3.cfg'
model_weights='darknet/yolov3.weights'
meta_data='darknet/Marrine_Vessel_cfg/marrine-vessel.data'
classes_file = 'darknet/data/voc.names'

########################################
# Simen
#########################################
# test_videos_base_path='/home/shaheryar/Desktop/Projects/oceans11/TestVideos'
# model_cfg='darknet/cfg/simen.cfg'
# meta_data='darknet/cfg/simen.data'
# model_weights='darknet/yolov3_simen.weights'
# classes_file = 'darknet/data/simen.names'


if __name__ == '__main__':


    net = dn.load_net(model_cfg.encode('utf-8'),
                      model_weights.encode('utf-8'), 0)
    meta = dn.load_meta(meta_data.encode('utf-8'))

    # dataset_base_path = "/home/shaheryar/Desktop/Projects/Marrine-Vessel-Detection/Dataset/Annotated_Dataset"
    # dataset_dir = os.listdir(dataset_base_path)
    # classes=config.labels_to_name.values()

    # for folder in dataset_dir:
    #     if(folder in classes):
    #         test_folder_path=os.path.join(dataset_dir,folder,'Test')
    #         for test_imgs in os.listdir(test_folder_path):

    detection_video('raw_video.avi', net, meta, output_path='test2.MOV', threshold=0.3)

    #detection_video('rtsp://admin:striker62@192.168.0.50/doc/page/preview.asp! decodebin ! videoconvert ! appsink max-buffers=1 drop=true', net, meta, output_path=os.path.join(output_path, 'test.MOV'), threshold=0.3)

    # for v in os.listdir(test_videos_base_path):
    #     print(v)
    #     if(len(str.split(v,'.'))==2):
    #         detection_video(test_videos_base_path+'/'+v,net,meta,output_path=os.path.join(output_path,v),threshold=0.3)



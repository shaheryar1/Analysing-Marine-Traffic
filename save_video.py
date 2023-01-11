#!/usr/bin/env python
import numpy as np
import cv2
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--filepath", help="filepath",default="raw_video.avi")
parser.add_argument("--videopath", help="videopath",default="rtsp://admin:striker62@192.168.0.50/doc/page/preview.asp! rtph264depay ! h264parse ! omxh264dec ! appsink max-buffers=1 drop=true")


args = parser.parse_args()

if os.path.exists(args.videopath):
    raise ValueError("file already exist")

cap = cv2.VideoCapture(args.videopath)

if cap.isOpened() == False:
    raise ValueError("couldn't open")

(grabbed, frame) = cap.read()
fshape = frame.shape
fheight = fshape[0]
fwidth = fshape[1]

print("-----------------------------------")
print("saving...", args.filepath)
print("resulution is:",fwidth,"x",fheight)
# Define the codec and create VideoWriter object


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(args.filepath,fourcc, 24.0, (fwidth,fheight))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:

        # write the flipped frame
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

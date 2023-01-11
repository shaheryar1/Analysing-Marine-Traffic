

## Getting Started with Yolo Version


#### Install dependencies using 

Open terminal and browse into oceans11 repo


```pip3 install -r requirements.txt```

####  Install darknet/Yolo framework

```cd <path-to-repo>/darknet```

```make```

To use CUDA follow original instructions here https://pjreddie.com/darknet/install/

#### Download Yolov3 and Yolov3-tiny weights for Marrine-Vessel-Detection from here
[Yolov3-Tiny weights](https://drive.google.com/open?id=1_vOYj5tezlTz5pPaEUtReg2DIfQ634S0)

[Yolov3](https://drive.google.com/open?id=1MEvAfEapA47HPDhanm8hF038z7RAne1R)

In yolo_inference.py change the path of weights and input_video to run inference on a video

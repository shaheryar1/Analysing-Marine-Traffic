

## Overview 
Building an autonomous vessel in marrine envrionment have so many challenges. One of the challenges is providing the vessel with the vision capabilities so it can analyze the marrine traffic and make different decisions i:e navigation, collision detection etc. This project is initiated to handle this problem. It focuses on following 
* ### Detect and classify maritime vessels using a vision based system
* ### Determine the speed of detected vessel 
* ### Detect name of ship 

## Dataset
The dataset was collected by scraping images from internet. It contains 11 classes.

| Classes        |       
|------------------|
| **Tanker**         |  
| **Tug**        |                                                                                            
| **Fishing Vessel** |                                                                                       
| **Container**  | 
| **Passenger Ship** |  
| **Sailing Vessel**|                                                                                            
| **Military Vessel** |                                                                                       
| **Supply Vessel**| 
| **Power Boat** |
| **Jet Ski**

## Approach
Trained Yolov3, Yolov3-Tiny and RetinaNet


## Detection Results 



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



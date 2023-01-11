import os
import glob
from xml.dom import minidom
import numpy as np

import shutil
import pandas as pd
import xml.etree.ElementTree as ET
import config




def parseXml(file_path):
    xmldoc = minidom.parse(file_path)
    itemlist = xmldoc.getElementsByTagName('object')
    size = xmldoc.getElementsByTagName('size')[0]
    width = int((size.getElementsByTagName('width')[0]).firstChild.data)
    height = int((size.getElementsByTagName('height')[0]).firstChild.data)
    itemlist = xmldoc.getElementsByTagName('object')
    size = xmldoc.getElementsByTagName('size')[0]
    width = int((size.getElementsByTagName('width')[0]).firstChild.data)
    height = int((size.getElementsByTagName('height')[0]).firstChild.data)

    data = {"width":width,
                "height":height}
    objects = []
    for item in itemlist:
        # get class label
        classid = (item.getElementsByTagName('name')[0]).firstChild.data

        # get bbox coordinates
        xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
        ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
        xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
        ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
        b = (float(xmin), float(xmax), float(ymin), float(ymax))
        info = (classid,b)
        objects.append(info)
    data['objects']=objects
    return data;


def extarctClass(dataset_path,class_name,destination_dir):
    try:
        os.mkdir(destination_dir)
    except:
        print("Error creating directory")
        return

    for annotation in os.listdir(dataset_path):
        if(annotation[-3:]=='xml'):
            # print(os.path.join(dataset_path,annotation))
            objects_dict=parseXml(os.path.join(dataset_path,annotation))
            image_name = annotation[:-4]+'.jpg'

            if(objects_dict['objects'][0][0]==class_name):

                shutil.copy(os.path.join(dataset_path,image_name),destination_dir)
                shutil.copy(os.path.join(dataset_path, annotation), destination_dir)

# extarctClass(config.SEA_SHIP_DATASET_PATH,'fishing boat','/home/ai/Desktop/Python Projects/Marrine-Vessel-Detection/Dataset/SeaShips(7000)/Extracted_Classes/fishing boat')
# print(parseXml('/home/ai/Desktop/Python Projects/Marrine-Vessel-Detection/Dataset/SeaShips(7000)/Annotations/003060.xml')['objects'][0][1])
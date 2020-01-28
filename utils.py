import os
import pandas as pd
DATASET_PATh="/home/ai/Desktop/Python Projects/Marrine_Vessel_Detection/Dataset/Backup";


def loadDataset():
    labels=[];
    X=[];
    dir = os.listdir(DATASET_PATh)

    for class_name in dir:

        for img_name in os.listdir(os.path.join(DATASET_PATh,class_name)):
            if(img_name[-3:]=='jpg'):
                X.append(os.path.join(DATASET_PATh,class_name)+'/'+img_name)
                labels.append(class_name)

    df = pd.DataFrame({'filename':X,
                       'label':labels})
    return df
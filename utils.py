import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)



DATASET_PATH="/home/shaheryar/Desktop/Projects/Marrine-Vessel-Detection/Dataset/Classification_Data";



def loadDataset():
    labels=[];
    X=[];
    dir = os.listdir(DATASET_PATH)

    for class_name in dir:

        for img_name in os.listdir(os.path.join(DATASET_PATH,class_name)):
            if(img_name[-3:]=='jpg'):
                X.append(os.path.join(DATASET_PATH,class_name)+'/'+img_name)
                labels.append(class_name)

    df = pd.DataFrame({'filename':X,
                       'label':labels})
    return df

def loadAugmentedDataset():
    labels=[];
    X=[];
    dir = os.listdir(DATASET_PATH)

    for class_name in dir:
        try:
            for img_name in os.listdir(os.path.join(DATASET_PATH,class_name,'augmented_data')):
                if(img_name[-3:]=='jpg'):
                    X.append(os.path.join(DATASET_PATH,class_name,'augmented_data')+'/'+img_name)
                    labels.append(class_name)
        except:
            pass
    df = pd.DataFrame({'filename':X,
                       'label':labels})
    return df
def loadTestDataset():
    labels = [];
    X = [];
    dir = os.listdir(DATASET_PATH)

    for class_name in dir:
        i=0;
        for img_name in os.listdir(os.path.join(DATASET_PATH, class_name,"Test")):
            i=i+1;
            if (img_name[-3:] == 'jpg'):
                X.append(os.path.join(DATASET_PATH, class_name,"Test") + '/' + img_name)
                labels.append(class_name)
            if(i==300):
                break;

    df = pd.DataFrame({'filename': X,
                       'label': labels})
    return df


def strong_aug(p=.5):
    return Compose([
        HorizontalFlip(p=1),
        OneOf([
            MotionBlur(p=.4),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=25, p=0.5),
    ], p=p)


def augment_and_save(aug, image, path):
    image = aug(image=image)['image']
    cv2.imwrite(path, image)
    print("saved in", path)


def dataAugmentation(df, class_name,total):
    try:
        path_to_save = os.path.join(DATASET_PATH, class_name, "augmented_data")
        os.mkdir(path_to_save)
    except:
        pass

    class_data = (df.loc[df['label'] == class_name, ["filename", 'label']])

    print(class_data.shape)
    k=0;
    for i in range(0,total):
        image_path = class_data.iloc[k]['filename']
        k=(k+1)%class_data.shape[0]

        img = cv2.imread(image_path)

        aug = strong_aug(p=1)
        augment_and_save(aug, img, os.path.join(path_to_save, str(i) + ".jpg"))

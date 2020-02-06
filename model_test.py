from utils import loadDataset,loadTestDataset
from sklearn.model_selection import train_test_split
from keras.applications.mobilenet import preprocess_input
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

from keras.preprocessing import image
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from app import buildModel


CLASSES=['Container ship','Fishing vessels','Military ship','Passenger ship','Sailing vessel','Tanker','Tug']
mapping={
    'Container ship':0,
    'Fishing vessels':1,
    'Military ship':2,
    'Passenger ship':3,
    'Sailing vessel':4,
    'Tanker':5,
    'Tug':6
}

def plotConfusionMatrix(mat):

    # Normalise
    normalized_mat = mat.astype('float') /mat.sum(axis=1)[:, np.newaxis]


    df_cm = pd.DataFrame(mat,index=CLASSES,columns=CLASSES)
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=0.8)  # for label size
    sn.heatmap(df_cm, annot=True,fmt='d', cbar=False)  # font size

    plt.show()


def load_image(img_path,expand_dim=False ):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    if(expand_dim):
        img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


def evaluateOnDataFrame(model,df):
    predictions = [];
    img_paths = list(df.loc[:, 'filename'])
    truth=list(df.loc[:,'label'])
    i=0;
    print(df['label'].value_counts())
    for path in img_paths:
        x = load_image(path,expand_dim=True)
        truth[i]=mapping[truth[i]]
        p=model.predict(x)
        p=np.argmax(p)
        predictions.append(p)
        i=i+1;

    accuracy = accuracy_score(truth,predictions)
    confusion_mat=confusion_matrix(truth,predictions)
    precsion, recall, f1_ccore, _ = precision_recall_fscore_support(truth, predictions)
    print("Accuracy :",accuracy)
    print("Precision :", precsion)
    print("Recall :", recall)
    print("f1 :", f1_ccore)
    print(confusion_mat)
    plotConfusionMatrix(confusion_mat)


if __name__ == '__main__':
    model = buildModel(7)
    model.load_weights('/home/shaheryar/Desktop/Projects/Vessel_Classification/snapshots2/cp-0010_best_loss.hdf5')
    df=loadDataset()
    train, val = train_test_split(df, test_size=0.20, random_state=42)
    test=loadTestDataset()
    evaluateOnDataFrame(model,test)



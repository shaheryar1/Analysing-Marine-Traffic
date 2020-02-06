from utils import loadDataset,loadTestDataset
from sklearn.model_selection import train_test_split
from keras.applications.mobilenet import preprocess_input
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import cv2
from math import sqrt
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import time
import matplotlib.pyplot as plt
from Model import buildModel
from config import mapping,CLASSES

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


def inferenceImage(model,path):
    img = load_image(img_path=path,expand_dim=True)
    p=model.predict(img)
    predicted_class=CLASSES[np.argmax(p)]
    print(predicted_class)
    return predicted_class



def inferenceVideo(model,video_path,save=False):
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        print("Error opening video  file")
        return
    i=1;
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if(i%30==0):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(frame,(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        p=model.predict(preprocess_input(img_array))
        print(CLASSES[np.argmax(p)],p[0][np.argmax(p)])

        i=i+1;

        frame = cv2.resize(frame,(0,0),fx=0.75,fy=0.75)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



def plot_images(images, cls_true, cls_pred=None):

    # Create figure with 3x3 sub-plots.
    count = len(images)
    fig, axes = plt.subplots(int(sqrt(count)), int(sqrt(count)))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i])

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def testRandom(model,df,count =16):
    idx=np.random.randint(0,df.shape[0],count)
    file_path_list=list(df.loc[idx,'filename'])
    true_labels =list(df.loc[idx,'label'])

    img_list =[];
    predicted_labels=[]
    for file_path in file_path_list:
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_list.append(img)
        prediction=inferenceImage(model,file_path)
        predicted_labels.append(prediction)


    plot_images(img_list,true_labels,predicted_labels)


if __name__ == '__main__':
    model = buildModel(len(CLASSES))
    weights_path='/home/ai/Desktop/Python Projects/Vessel_Classification/snapshots2/cp-0010_best_loss.hdf5'
    model.load_weights(weights_path)
    # path = "/home/ai/Desktop/Python Projects/Marrine_Vessel_Detection/Dataset/videos/videos 2/videos 2/Richardson_Bay/IMG_4301.MOV"
    # inferenceVideo(model,video_path=path)

    df=loadTestDataset()
    testRandom(model,df)
    # train, val = train_test_split(df, test_size=0.20, random_state=42)
    # test=loadTestDataset()
    # evaluateOnDataFrame(model,test)



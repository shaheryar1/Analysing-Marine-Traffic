from Model import buildModel,startTraining
from utils import loadDataset, loadAugmentedDataset
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from tensorflow import ConfigProto
from tensorflow import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def train():
    df = loadDataset()
    train, val = train_test_split(df, test_size=0.20, random_state=42)
    print(train.shape)
    print(val.shape)
    aug_df = loadAugmentedDataset()
    print(aug_df.head())
    train_aug_df = train.append(aug_df)
    print(train_aug_df.shape)

    model = buildModel(7);

    history = startTraining(model, train_aug_df, val, epochs=40)

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test', 'train loss', 'test loss'], loc='upper left')
    plt.show()


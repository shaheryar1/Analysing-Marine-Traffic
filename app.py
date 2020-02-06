from keras.applications import MobileNet
from keras import Model

from keras.layers import Dense, GlobalAveragePooling2D
import os
import keras
from utils import loadDataset, loadAugmentedDataset
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.mobilenet import preprocess_input
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from tensorflow import ConfigProto
from tensorflow import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


CLASSES=['Container ship','Fishing vessels','Military ship','Passenger ship','Sailing vessel','Tanker','Tug']
def buildModel(NUM_CLASSES):
    base_model = MobileNet(weights='imagenet', include_top=True)

    x = base_model.layers[-7].output
    x=GlobalAveragePooling2D()(x)
    preds = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=preds)

    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-1:]:
        layer.trainable = True

    return model


def startTraining(model, train_aug_df, val_df, BATCH_SIZE=64, epochs=50, checkpoint_path='snapshots2'):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_aug_df,
        x_col="filename",
        y_col="label",
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical')
    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filename",
        y_col="label",
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #
    checkpoint_path = checkpoint_path + "/cp-{epoch:04d}.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=False,
        period=1)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=int(train_aug_df.shape[0] / (BATCH_SIZE)),
        epochs=epochs,
        callbacks=[cp_callback],
        validation_data=validation_generator,
        validation_steps=int(val_df.shape[0] / (BATCH_SIZE))
    )

    print(history.history.keys())
    return history

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


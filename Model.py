from keras.applications import MobileNet
from keras import Model
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.mobilenet import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
import os
import keras
from config import CLASSES

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

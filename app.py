from keras.applications import MobileNet,VGG16,InceptionV3
from keras import Model
from keras.datasets import cifar10
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.utils import np_utils

from utils import loadDataset
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input

from tensorflow import ConfigProto
from tensorflow import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
NUM_CLASSES=4




base_model =InceptionV3(weights='imagenet',include_top=True)
base_model.layers.pop()
BATCH_SIZE = 64

x=base_model.layers[-1].output
# x=GlobalAveragePooling2D()(x)
# x=BatchNormalization()(x)
# x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
# x=Dropout(0.5)(x)
# x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(NUM_CLASSES,activation='softmax')(x)
model=Model(inputs=base_model.input,outputs=preds)

for layer in model.layers:
    layer.trainable=False
# or if we want to set the first 20 layers of the network to be non-trainable
for layer in model.layers[-1:]:
    layer.trainable=True

model.summary()

df=loadDataset()
print(df['label'].value_counts())
X_train, X_val = train_test_split(df, test_size=0.20, random_state=42)

print(X_train.shape)
print(X_val.shape)

train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        zoom_range=0.2,
        horizontal_flip=True)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = train_datagen.flow_from_dataframe(
        dataframe=X_train,
        x_col="filename",
        y_col="label",
        target_size=(299, 299),
        batch_size=BATCH_SIZE,
        class_mode='categorical')
validation_generator = val_datagen.flow_from_dataframe(
        dataframe=X_val,
        x_col="filename",
        y_col="label",
        target_size=(299, 299),
        batch_size=BATCH_SIZE,
        class_mode='categorical')


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(
    train_generator,
    steps_per_epoch=int(X_train.shape[0]/(BATCH_SIZE)),
    epochs=100,
    validation_data=validation_generator,
    validation_steps=int(X_val.shape[0]/(BATCH_SIZE))
)
# model.fit(x_train,y_train,16,epochs=100)

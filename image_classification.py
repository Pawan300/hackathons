import numpy as np 
import pandas as pd 
import cv2, pickle
from PIL import Image 
import os
import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications import InceptionResNetV2
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils import np_utils

data = pd.read_csv("/content/drive/MyDrive/demo/dataset/train.csv")

test_data_path = "/content/drive/MyDrive/demo/dataset/array/test.npy"
train_data_path = "/content/drive/MyDrive/demo/dataset/array/train.npy"

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 5
test_filename = []

with open("/content/drive/MyDrive/demo/temp.txt", 'r') as f:
  for i in f.readlines():
    test_filename.append(i.split("\n")[0])

if not os.path.isfile(train_data_path):
    train_data = []
    for i in data["Image"]:    
        path = "/content/drive/MyDrive/demo/dataset/train"+"/"+i
        img_data = cv2.imread(path)
        img_data = cv2.resize(img_data, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_NEAREST)
        train_data.append(np.array(img_data))
    train_data = np.array(train_data)
    np.save(train_data_path, train_data)

path = "/content/drive/MyDrive/demo/dataset/test"

if not os.path.isfile(test_data_path): 
    test_data = []
    for filename in os.listdir(path):
        p = os.path.join(path,filename)
        img = cv2.imread(p)
        if img is not None:
            img_data = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_NEAREST)
            test_data.append(np.array(img_data))
    test_data = np.array(test_data)
    np.save(test_data_path, test_data)

train = np.load(train_data_path) 
test = np.load(test_data_path)

encoder = LabelEncoder()
encoder.fit(data["Class"])
encoded_Y = encoder.transform(data["Class"])
y_labels = np_utils.to_categorical(encoded_Y)

from tensorflow.keras.models import Model
import tensorflow.keras as keras

resnet = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3),pooling='avg')

output = resnet.layers[-1].output
output = tf.keras.layers.Flatten()(output)
resnet = Model(resnet.input, output)

res_name = []
for layer in resnet.layers:
    res_name.append(layer.name)

set_trainable = False
for layer in resnet.layers:
    if layer.name in res_name[-447:]:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout

model = Sequential()
model.add(resnet)
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(y_labels.shape[1], activation='softmax'))


adam = tf.keras.optimizers.Adam(learning_rate=0.0001)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='f1_score', patience=8,
                                              restore_best_weights=False
                                              )

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='f1_score',
                                   factor=0.2,
                                   patience=4,
                                   verbose=1,
                                   min_delta=5*1e-3,min_lr = 5*1e-7,
                                   )

model.compile(optimizer = adam, 
              loss = 'categorical_crossentropy', 
              metrics=['accuracy',tfa.metrics.F1Score(num_classes=y_labels.shape[1])])

model.fit(train, y_labels,steps_per_epoch=np.ceil(float(train.shape[0]) / float(BATCH_SIZE)),
                        epochs = 50,callbacks=[early_stop,reduce_lr])

pred = model.predict_classes(t)
pred = encoder.inverse_transform(pred)
result = pd.DataFrame(pred, test_filename, columns=["Class"])
result.to_csv("/content/drive/MyDrive/demo/sample.csv")


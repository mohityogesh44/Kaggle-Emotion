#!/usr/bin/env python
# coding: utf-8

#Uncomment below lines if you are using a Google Colab notebook. These lines wil help you fetch data directly using Kaggle api and enable in Google Drive use.

# from google.colab import drive
# drive.mount('/content/gdrive')

# import os
# os.environ['KAGGLE_CONFIG_DIR'] = '/content/gdrive/My Drive/Kaggle'

# %cd '/content/gdrive/My Drive/Kaggle'

# !kaggle datasets download -d msambare/fer2013

#Assuming that you have dataset zip file download from Kaggle.

#Extracting the data from zip to "/tmp" folder.
import zipfile
zip = zipfile.ZipFile("fer2013.zip",'r')
zip.extractall('/tmp')
zip.close()

#Importing the necessary libraries and packages.

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,Dropout,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#Path to train and test directories
train_dir = "/tmp/train"
test_dir = "/tmp/test"

#Using Image Data Generator to load images(Training and Testing)...Feel free to play around with "Data Augmentation".
train_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_dir,target_size = (48,48),batch_size = 64,color_mode='grayscale',class_mode = 'categorical')
test_generator = val_datagen.flow_from_directory(test_dir,target_size=(48,48),batch_size=64,color_mode='grayscale',class_mode = 'categorical')

#Defining the model architecture...Using a Dropout layers to reduce overfitting.
model = Sequential([ Conv2D(32, (3,3),activation='relu',input_shape=(48,48,1)),
                     Conv2D(64, (3,3),activation='relu'),
                     BatchNormalization(),
                     MaxPooling2D(2,2),
                     Dropout(0.25),
                     
                     Conv2D(128, (3,3),activation='relu'),                     
                     Conv2D(128,(3,3),activation='relu'),
                     BatchNormalization(),
                     MaxPooling2D(2,2),
                     Dropout(0.25),
                     
                     Conv2D(256,(3,3),activation='relu'),
                     Flatten(),
                     Dense(1024,activation='relu'),
                     Dropout(0.5),
                     Dense(7,activation='softmax')
                   ])

#Compiling the model with Adam Optimizer
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

#Print the model summary
model.summary()

#Training the model
model_hist = model.fit(train_generator,steps_per_epoch=28709//64,epochs=100,validation_data=test_generator,validation_steps=7178//64)

#Save the model weights and architecture.
model.save("/tmp/emotion_model.h5")

#Prediction on your own image-->Uncomment below lines to get your image to be predicted.

# path = 'path to image'
# img = image.load_img(path, target_size=(300, 300))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)

# images = np.vstack([x])
# classes = model.predict(images, batch_size=10)

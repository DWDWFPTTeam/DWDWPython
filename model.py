import os
from keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from keras.utils.np_utils import to_categorical
import random,shutil
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from imutils import paths
from keras.models import load_model
import tensorflow as tf

#
# def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):
#
    # return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)
#
# BS= 32
# TS=(24,24)
# train_batch= generator('data/train',shuffle=True, batch_size=BS,target_size=TS)
# valid_batch= generator('data/valid',shuffle=True, batch_size=BS,target_size=TS)
# SPE= len(train_batch.classes)//BS
# VS = len(valid_batch.classes)//BS
# print(SPE,VS)


image_path = list(paths.list_images('Dataset2/'))

random.shuffle(image_path)
#load labels from folder
labels = [int(p.split(os.path.sep)[1]) for p in image_path]
labels = np.asarray(labels)
listImage = []

#load full-trainset from folder
for (i, imagePath) in enumerate (image_path):
    image = load_img(imagePath, target_size=(24,24), color_mode="grayscale")
    image = img_to_array(image)
    listImage.append(image)

#prepare trainset and validset from full-trainset
size = len(listImage)
size_train = int(size * 0.8)
X_train = np.asarray(listImage)
print("Full XTrain shape: " + str(X_train.shape))
x_train, y_train = X_train[:size_train,:], labels[:size_train]
x_val, y_val = X_train[size_train:size,:], labels[size_train:size]
print("XTrain shape: " + str(x_train.shape))
print("XValid shape: " + str(x_val.shape))
print("YTrain shape: " + str(y_train.shape))
print("YValid shape: " + str(y_val.shape))
print(y_train[100])

#Reshape X_Train and X_Validate
# x_train = np.reshape(x_train.shape[0], x_train.shape[3], x_train.shape[1], x_train.shape[2])
# x_val = np.reshape(x_val.shape[0], x_val.shape[3], x_val.shape[1], x_val.shape[2])
# x_train = np.reshape(x_train.shape[0], 1, 24, 24)
# x_val = np.reshape(x_val.shape[0], 1, 24, 24)

print("XTrain shape: " + str(x_train.shape))
print("XValid shape: " + str(x_val.shape))

#Encoding labels
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
print(y_train[100])
print("YTrain shape: " + str(y_train.shape))
print("YValid shape: " + str(y_val.shape))

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_batch = datagen.flow(x_train, y_train, batch_size=32, shuffle=True, )
valid_batch = datagen.flow(x_val, y_val, batch_size=32, shuffle=True)
SPE = len(x_train)/32
VS = len(x_val)/32
print("Train batch size: " + str(len(train_batch)))
print("SPE : " + str(SPE))
print("VS : " + str(VS))
# img,labels= next(train_batch)
# print(img.shape)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
#32 convolution filters used each of size 3x3
#again
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),

#64 convolution filters used each of size 3x3
#choose the best features via pooling
    
#randomly turn neurons on and off to improve convergence
    Dropout(0.25),
#flatten since too many dimensions, we only want a classification output
    Flatten(),
#fully connected to get all relevant data
    Dense(128, activation='relu'),
#one more dropout for convergence' sake :) 
    Dropout(0.5),
#output a softmax to squash the matrix into output probabilities
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(train_batch, validation_data=valid_batch,epochs=15,steps_per_epoch=SPE,validation_steps=VS)

model.save('models/cnnCat.h5', overwrite=True)
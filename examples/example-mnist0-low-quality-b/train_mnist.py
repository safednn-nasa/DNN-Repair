import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten, Activation
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import math
import cv2
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint


alpha = 1e-4
batch_size = 128
epochs = 10
num_filters = 32 # increase this to 32
lam_bda = 0.05 # regularization constant


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)


y_train=to_categorical(y_train, num_classes=10)
y_test=to_categorical(y_test, num_classes=10)

print (x_train.shape)


model=Sequential()
model.add(Conv2D(2, kernel_size=(3, 3), strides=(1, 1),padding="valid",
                 kernel_initializer='random_uniform',
                bias_initializer='random_uniform',
                 activation='linear',input_shape=[28,28,1]))
model.add(Activation('relu'))
model.add(Conv2D(4,kernel_size=(3,3),strides=(1,1),padding="valid",
                kernel_initializer='random_uniform',
                bias_initializer='random_uniform',
                activation="linear"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="valid"))
model.add(Flatten())
model.add(Dense(128,activation="linear",kernel_initializer='random_uniform',
                bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(Dense(10,activation="linear",kernel_initializer='random_uniform',
                bias_initializer='zeros'))
model.add(Activation('softmax'))


model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
filepath = "saved-model-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]
#model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=1, validation_data=(x_test, y_test))
#model.fit(x_train[0:10000], y_train[0:10000], epochs=8, batch_size=batch_size, verbose=1, validation_data=(x_test, y_test))
model.fit(x_train[0:10000], y_train[0:10000], epochs=20, batch_size=batch_size, verbose=1, validation_data=(x_test, y_test), callbacks=callbacks_list)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('mnist0_normal_low_accuracy.h5')


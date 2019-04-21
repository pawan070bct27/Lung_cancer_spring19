# -*- coding: utf-8 -*-
"""

Created on Sat Apr 20 17:04:34 2019

@author: Avishkar
"""

from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D, MaxPooling2D, Dropout
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras import initializers
from keras import backend as K
import numpy as np



xy_train_data = np.load('C:\\Users\\Avishkar\\Desktop\\lung_caner\\Lung_Data_for_Pawan\\train_xy.npy')
xy_testing_data = np.load('C:\\Users\\Avishkar\\Desktop\\lung_caner\\Lung_Data_for_Pawan\\test_xy.npy')




print("data ndim: ", xy_train_data.ndim)

print("data shape:", xy_train_data.shape)

print("data size: ", xy_train_data.size)





#Set the initial parameters
batch_size = 128
nb_classes = 6
nb_epoch = 6

img_rows, img_cols = 56, 56         # input image dimensions
pool_size = (2, 2)                  # size of pooling area for max pooling
prob_drop_conv = 0.2                # drop probability for dropout @ conv layer
prob_drop_hidden = 0.5              # drop probability for dropout @ fc layer


def init_weights(shape, name=None):
    return initializers.normal(shape, scale=0.01, name=name)







if K.image_dim_ordering() == 'th':
    # For Theano backend
    xy_train_data = xy_train_data.reshape(xy_train_data.shape[0], 1, img_rows, img_cols)
    xy_testing_data = xy_testing_data.reshape(xy_testing_data.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    # For TensorFlow backend
    xy_train_data = xy_train_data.reshape(xy_train_data.shape[0], img_rows, img_cols,1)
    xy_testing_data = xy_testing_data.reshape(xy_testing_data, img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

xy_train_data = xy_train_data.astype('float32') / 255.
xy_testing_data = xy_testing_data.astype('float32') / 255.
xy_train_data = np_utils.to_categorical(xy_train_data, nb_classes)
xy_testing_data = np_utils.to_categorical(xy_testing_data, nb_classes)


# Convolutional model
model = Sequential()

# conv1 layer
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=pool_size, strides=(2,2), border_mode='same'))
model.add(Dropout(prob_drop_conv))

# conv2 layer
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size, strides=(2,2), border_mode='same'))
model.add(Dropout(prob_drop_conv))

# conv3 layer
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size, strides=(2,2), border_mode='same'))
model.add(Flatten())
model.add(Dropout(prob_drop_conv))

# fc1 layer
model.add(Dense(625, activation='relu'))
model.add(Dropout(prob_drop_hidden))

# fc2 layer
model.add(Dense(10, activation='softmax'))

opt = RMSprop(lr=0.001, rho=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# Train
history = model.fit(xy_train_data, xy_testing_data, nb_epoch=nb_epoch, batch_size=batch_size, shuffle=True, verbose=1)



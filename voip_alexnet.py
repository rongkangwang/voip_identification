#coding:utf-8
from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
#from six.moves import range
from data_voip import load_data
import random,cPickle
from keras.callbacks import EarlyStopping
import numpy as np

np.random.seed(1024)  # for reproducibility
input_shape = (100,256,1)

nb_class = 7

def create_alexnet_model():
	model = Sequential()
	# Conv layer 1 output shape (55, 55, 48)
	model.add(Conv2D(
	    kernel_size=(11, 11), 
	    data_format="channels_last", 
	    activation="relu",
	    filters=48, 
	    strides=(4, 4), 
	    input_shape=input_shape
	))
	model.add(Dropout(0.25))
	# Conv layer 2 output shape (27, 27, 128)
	model.add(Conv2D(
	    strides=(2, 2), 
	    kernel_size=(5, 5), 
	    activation="relu", 
	    filters=128
	))
	model.add(Dropout(0.25))
	# Conv layer 3 output shape (13, 13, 192)
	model.add(Conv2D(
	    kernel_size=(3, 3),
	    activation="relu", 
	    filters=192,
	    padding="same",
	    strides=(2, 2)
	))
	model.add(Dropout(0.25))
	# Conv layer 4 output shape (13, 13, 192)
	model.add(Conv2D(
	    padding="same", 
	    activation="relu",
	    kernel_size=(3, 3),
	    filters=192
	))
	model.add(Dropout(0.25))
	# Conv layer 5 output shape (128, 13, 13)
	model.add(Conv2D(
	    padding="same",
	    activation="relu", 
	    kernel_size=(3, 3),
	    filters=128
	))
	model.add(Dropout(0.25))
	# fully connected layer 1
	model.add(Flatten())
	model.add(Dense(2048, activation='relu'))
	model.add(Dropout(0.25))

	# fully connected layer 2
	model.add(Dense(2048, activation='relu'))
	model.add(Dropout(0.25))

	# output
	model.add(Dense(nb_class, activation='softmax'))
	return model

def train():
	data, label = load_data()
	label = np_utils.to_categorical(label, nb_class)

	model = create_alexnet_model()
	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

	index = [i for i in range(len(data))]
	random.shuffle(index)
	data = data[index]
	label = label[index]
	(X_train,X_val) = (data[0:12000],data[12000:])
	(Y_train,Y_val) = (label[0:12000],label[12000:])

	#使用early stopping返回最佳epoch对应的model
	early_stopping = EarlyStopping(monitor='val_loss', patience=1)
	model.fit(X_train, Y_train, batch_size=100,validation_data=(X_val, Y_val),epochs=10,callbacks=[early_stopping])
	json_string = model.to_json()
	open('alexnet_model_architecture_100.json','w').write(json_string)
	model.save_weights('alexnet_model_weights_100.h5')

def checkprint():
	model = create_alexnet_model()
	model.summary()

if __name__=="__main__":
	train()
	# checkprint()
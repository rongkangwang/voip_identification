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

#rows = 100
np.random.seed(1024)  # for reproducibility
#input_shape = (rows,256,1)

nb_class = 7

def create_alexnet_model(input_shape=(100,256,1)):
	model = Sequential()
	# Conv layer 1 output shape (55, 55, 48)
	# model.add(Conv2D(
	#     kernel_size=(11, 11), 
	#     data_format="channels_last", 
	#     activation="relu",
	#     filters=48, 
	#     strides=(4, 4), 
	#     input_shape=input_shape
	# ))
	model.add(Conv2D(
	    kernel_size=(5, 5),
	    data_format="channels_last", 
	    activation="relu",
	    filters=48, 
	    strides=(2, 2), 
	    input_shape=input_shape
	))
	model.add(Dropout(0.25))
	# Conv layer 2 output shape (27, 27, 128)
	# model.add(Conv2D(
	#     strides=(2, 2), 
	#     kernel_size=(5, 5), 
	#     activation="relu", 
	#     filters=128
	# ))
	model.add(Conv2D(
	    strides=(2, 2), 
	    kernel_size=(3, 3),
	    activation="relu", 
	    filters=128,
	    padding="same"
	))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	# Conv layer 3 output shape (13, 13, 192)
	model.add(Conv2D(
	    kernel_size=(3, 3),
	    activation="relu", 
	    filters=192,
	    padding="same",
	    strides=(1, 1)
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

def create_alexnet_model_original(input_shape=(100,256,1)):
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
	    strides=(1, 1)
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

def train(rows=100):
	data, label = load_data(rows)
	label = np_utils.to_categorical(label, nb_class)

	model = create_alexnet_model(input_shape=(rows,256,1))

	import math
	from keras.callbacks import LearningRateScheduler
	def step_decay(epoch):
	    initial_lrate = 0.01
	    drop = 0.5
	    epochs_drop = 2.0
	    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
	    print(lrate)
	    return lrate
	lrate = LearningRateScheduler(step_decay)

	sgd = SGD(lr=0.0, decay=0.0, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

	index = [i for i in range(len(data))]
	random.shuffle(index)
	data = data[index]
	label = label[index]
	(X_train,X_val) = (data[0:12000],data[12000:])
	(Y_train,Y_val) = (label[0:12000],label[12000:])

	#使用early stopping返回最佳epoch对应的model
	early_stopping = EarlyStopping(monitor='val_loss', patience=1)
	model.fit(X_train, Y_train, batch_size=100,validation_data=(X_val, Y_val),epochs=10,callbacks=[early_stopping,lrate])
	json_string = model.to_json()
	open('../data/model_json/alexnet_model_architecture_'+str(rows)+'.json','w').write(json_string)
	model.save_weights('../data/model_json/alexnet_model_weights_'+str(rows)+'.h5')

	(x_test,y_test) = (data[0:],label[0:])
	loss,accuracy = model.evaluate(x_test,y_test)
	open('../data/model_json/result.txt', 'a+').write("pkt_num:%d, loss:%f, accuracy:%f\r\n"%(rows,loss,accuracy))

def checkprint(rows=100):
	model = create_alexnet_model(input_shape=(rows,256,1))
	model.summary()

if __name__=="__main__":
	rs = [10]
	for rows in rs:
		train(rows=rows)
	# checkprint(rows=10)

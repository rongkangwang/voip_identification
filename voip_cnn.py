#coding:utf-8

#导入各种用到的模块组件
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
rows_default = 100
cols_default = 256
image_shape = (rows_default,cols_default,1)


#加载数据
data, label = load_data()


#label为0~9共10个类别，keras要求形式为binary class matrices,转化一下，直接调用keras提供的这个函数
nb_class = 7
label = np_utils.to_categorical(label, nb_class)


def create_model():
	model = Sequential()
	model.add(Conv2D(4, (5, 5), padding='valid',input_shape=image_shape))
	model.add(Activation('relu'))

	model.add(Conv2D(8,(3, 3), padding='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(16,(3, 3), padding='valid')) 
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(128, kernel_initializer='normal'))
	model.add(Activation('relu'))

	model.add(Dense(nb_class, kernel_initializer='normal'))
	model.add(Activation('softmax'))
	return model


#############
#开始训练模型
##############
model = create_model()
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
open('cnn_model_architecture_100.json','w').write(json_string)
model.save_weights('cnn_model_weights_100.h5')
#cPickle.dump(model,open("./model.pkl","wb"))
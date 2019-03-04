#coding:utf-8
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers import LSTM, Reshape, concatenate
from keras.models import Model
from keras.utils.vis_utils import plot_model
from data_voip import load_data
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
import math
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils, generic_utils
import numpy as np
import random,cPickle

nb_class = 10

def create_clnn_model(input_shape=(100,256,1)):
	img_input=Input(input_shape)
	conv = create_cnn_layers(img_input)
	lstm = create_lstm_layers(img_input, input_shape[0])
	model = concatenate([conv, lstm],axis=1)
	model = Dense(2048, activation='relu')(model)
	model = Dropout(0.5)(model)
	model = Dense(nb_class, activation='softmax')(model)
	return model, img_input


def create_cnn_layers(img_input):
	# 1
	conv = Conv2D(
	    kernel_size=(5, 5),
	    data_format="channels_last", 
	    activation="relu",
	    filters=48, 
	    strides=(2, 2), 
	    padding="same"
	)(img_input)
	# 1 maxpooling
	#conv = MaxPooling2D(pool_size=(2, 2))(conv)
	conv = Dropout(0.5)(conv)
	# 2
	conv = Conv2D(
	    strides=(2, 2), 
	    kernel_size=(3, 3),
	    activation="relu", 
	    filters=128,
	    padding="same"
	)(conv)
	# 2 maxpooling
	#conv = MaxPooling2D(pool_size=(2, 2))(conv)
	conv = Dropout(0.5)(conv)
	# 3
	conv = Conv2D(
	    kernel_size=(3, 3),
	    activation="relu", 
	    filters=192,
	    padding="same",
	    strides=(1, 1)
	)(conv)
	conv = Dropout(0.5)(conv)
	# 4 
	conv = Conv2D(
	    padding="same", 
	    activation="relu",
	    kernel_size=(3, 3),
	    filters=192
	)(conv)
	conv = Dropout(0.5)(conv)
	conv = Conv2D(
	    padding="same",
	    activation="relu", 
	    kernel_size=(3, 3),
	    filters=128
	)(conv)
	# 3 maxpooling
	#conv = MaxPooling2D(pool_size=(2, 2))(conv)
	conv = Dropout(0.5)(conv)

	conv = Flatten()(conv)
	conv = Dense(2048, activation='relu')(conv)
	conv = Dropout(0.5)(conv)

	return conv

def create_lstm_layers(img_input, rows=100):
	lstm = Reshape((rows,256))(img_input)
	lstm = LSTM(
        units=512,
        activation='tanh',
        return_sequences=True)(lstm)
	lstm = LSTM(
        units=512,
        activation='tanh',
        return_sequences=False)(lstm)
	lstm = Dropout(0.5)(lstm)
	return lstm

def check_print(rows=100):
    model, img_input=create_clnn_model(input_shape=(rows,256,1))

    # Create a Keras Model
    model=Model(input=img_input,output=[model])
    model.summary()
    #model.summary(print_fn=printfunc)
    #print(model.get_config())
    #plot_model(model, to_file='clnn.svg',show_shapes=False)
    print(model)

def printfunc(string):
	print(string[0:30]+string[67:90])

def train(rows=100):
	data, label = load_data(rows)
	label = np_utils.to_categorical(label, nb_class)

	model, img_input=create_clnn_model(input_shape=(rows,256,1))
	model=Model(input=img_input,output=[model])

	import math
	from keras.callbacks import LearningRateScheduler
	def step_decay(epoch):
	    initial_lrate = 0.01
	    drop = 0.5
	    epochs_drop = 5.0
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
	(X_train,X_val) = (data[0:80000],data[80000:])
	(Y_train,Y_val) = (label[0:80000],label[80000:])

	#使用early stopping返回最佳epoch对应的model
	early_stopping = EarlyStopping(monitor='loss', patience=1)
	#全部载入内存
	model.fit(X_train, Y_train, batch_size=100,validation_data=(X_val, Y_val),epochs=20,callbacks=[early_stopping,lrate])
	
	json_string = model.to_json()
	open('../data/model_json/alexnet_model_architecture_'+str(rows)+'.json','w').write(json_string)
	model.save_weights('../data/model_json/alexnet_model_weights_'+str(rows)+'.h5')

	(x_test,y_test) = (data[0:],label[0:])
	loss,accuracy = model.evaluate(x_test,y_test)
	open('../data/model_json/result.txt', 'a+').write("pkt_num:%d, loss:%f, accuracy:%f\r\n"%(rows,loss,accuracy))

if __name__=="__main__":
	#rs = [6,8,10,20,40,100]
	#for rows in rs:
	#	train(rows=rows)
	train(rows=10)
	#check_print(rows=100)
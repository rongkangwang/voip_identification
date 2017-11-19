import numpy as np
import struct
# from PIL import Image
# from scipy import signal
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
#
# img = Image.open("./mnist/0.0.jpg")
# arr = np.asarray(img,dtype="float32")
#
# ker = np.random.random((4,5,5))
#
# a = [[1,2],[3,4]]
# b = [[-1,1],[-2,2]]
#
# #c = np.convolve([1,2,3,4],[1,1,3],mode="full")
# #c = signal.convolve2d(arr,ker,mode="valid")
#
# c = [[17,24,1,8,15],[23,5,7,14,16],[4,6,13,20,22],[10,12,19,21,3],[11,18,25,2,8]]
# d = [[8,1,6],[3,5,7],[4,9,2]]
#
# e = signal.convolve2d(c,d,mode="same")
# arr = arr.reshape( 28, 28, 1)
# arr = arr.reshape( 28, 28)
#
# print(np.shape(arr))
# plt.imshow(arr,cmap = cm.Greys_r)
# plt.show()

# import struct
#
# a = "\x04\x00"
# packet_len = struct.unpack('h', a)[0]
# print packet_len

# a = struct.pack("I",1409286213)
# print a

# a = np.zeros((1,2,4),dtype="float32")
# b = np.zeros((1,2,4),dtype="float32")
# c = np.array([a,b])
# print c
#
# a = 111
# b = 111/4*3
# print b

# a = "baidu__bre.pcap"
# print a.split("_")
# a = 0x2123
# flag = struct.unpack('H', "!#")[0]
# print flag
# print int('2321', 16)
# print hex(8993)

# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation, Flatten
# #from keras.layers.convolutional import Convolution2D, MaxPooling2D
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.optimizers import SGD
# from keras.utils import np_utils, generic_utils
# from six.moves import range
# from data import load_data
# import random,cPickle
# from keras.callbacks import EarlyStopping
# import numpy as np
#
# nb_class = 10
#
# def create_model():
# 	model = Sequential()
# 	model.add(Conv2D(4, (5, 5), padding='valid',input_shape=(28,28,1)))
# 	model.add(Activation('relu'))
#
# 	model.add(Conv2D(8,(3, 3), padding='valid'))
# 	model.add(Activation('relu'))
# 	model.add(MaxPooling2D(pool_size=(2, 2)))
#
# 	model.add(Conv2D(16,(3, 3), padding='valid'))
# 	model.add(Activation('relu'))
# 	model.add(MaxPooling2D(pool_size=(2, 2)))
#
# 	model.add(Flatten())
# 	model.add(Dense(128, kernel_initializer='normal'))
# 	model.add(Activation('relu'))
#
# 	model.add(Dense(nb_class, kernel_initializer='normal'))
# 	model.add(Activation('softmax'))
# 	return model
#
# model = create_model()
# model.summary()

# import random
# data = [1 ,2, 3]
# index = [i for i in range(len(data))]
# random.shuffle(index)
# print(index)
# data = data[index]
# print(data[index])
# import numpy as np
# l = [[1,2,3],[5,6,7]]
# print l.size()

# import numpy as np
# a = []
# a.append([1,1,1])
# a.append([2,2,2])
# an = np.asarray([3,3,3])
# a.append(an)
# anp = np.asarray(a,dtype="float32")
#
# print(anp)

import data_voip
import numpy as np
from PIL import Image

(data,label) = data_voip.load_data()

print(np.shape(data))
print(np.shape(label))
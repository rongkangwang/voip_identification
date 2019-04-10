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

# import data_voip
# import numpy as np
# from PIL import Image

# (data,label) = data_voip.load_data(rows=2)

# print(np.shape(data))
# print(np.shape(label))
# for i in range(10):
# 	open("test.txt","a+").write("%d\r\n"%(233))
import numpy
import matplotlib.pyplot as plt

# a = [1,2,3,4]
# a = tuple(a)
# acc10 =  (1,2,3,4,1,3,2)
# acc20 =  (1,2,3,4,1,3,2)
# acc40 =  (1,2,3,4,1,3,2)
# acc100 =  (1,2,3,4,1,3,2)
# n_groups = 7
# index = np.arange(n_groups)  
# bar_width = 0.15
# opacity=0.4
# rects10 = plt.bar(index, acc10, bar_width,alpha=opacity, color='b',label='10')
# rects20 = plt.bar(index + bar_width, acc20, bar_width,alpha=opacity,color='r',label='20')
# rects40 = plt.bar(index + 2*bar_width, acc40, bar_width,alpha=opacity, color='y',label='40')
# rects200 = plt.bar(index + 3*bar_width, acc100, bar_width,alpha=opacity,color='m',label='100')

# plt.xlabel('Category')  
# plt.ylabel('Scores')  
# plt.title('Scores by group and Category')  

# plt.xticks(index+2*bar_width-0.075,('balde','bunny','dragon','happy','pillow','1','2'),fontsize=18)
# plt.yticks(fontsize =18)  #change the num axis size

# plt.ylim(0,4)  #The ceil
# plt.legend()  

# plt.tight_layout(); 
# plt.show()
# print(type(acc10))
# print(type(a))

# file = open("../result/pred/test.txt")
# for line in file:
# 	label = line.strip().split(" ")
# 	print label
# 	print(type(label[0]))
# 	print(type(label))
# 	str_label = np.array(label)
# 	print(type(str_label[0]))
# 	print(type(str_label))
# 	float_label = str_label.astype(np.float)
# 	print(type(float_label[0]))
# 	print(type(float_label))
#
# 	float2_label = np.asarray(label,dtype="float")
# 	print(type(float2_label[0]))
# 	print(type(float2_label))
# m = [ [] for i in range(7) ]
# print m
# import random
# print (random.randint(10, 100))
# import seaborn as sns
# x=[1,2,3,4,5,6,7]
# y=[0.7,0.8,0.77,0.9,0.65,0.55,0.79]
# sns.set()
# plt.plot(x,y,color="r")
#
# plt.show()
<<<<<<< HEAD
#result1 = struct.pack('B',0xca)
#result2 = struct.pack('B',0x4e)

# port = struct.unpack('H', result2+result1)[0]
# print port

# from capturer import isbigendian
# isbigendian()
a = [2,3]
a = np.array(a)
print(a.shape)
=======
result1 = struct.pack('B',0xca)
result2 = struct.pack('B',0x4e)

port = struct.unpack('H', result2+result1)[0]
print port

from capturer import isbigendian
isbigendian()
>>>>>>> 162bf8fb56f49c5dda2580a420014a1b1b2be0a3

#coding=utf-8

"""
Author:wepon
Code:https://github.com/wepe
File: get_feature_map.py
	1.  visualize feature map of Convolution Layer, Fully Connected layer
	2.  rewrite the code so you can treat CNN as feature extractor, see file: cnn-svm.py
--
2016.06.06更新：
keras的API已经发生变化，现在可视化特征图可以直接调用接口，具体请参考：http://keras.io/visualization/
"""
from __future__ import print_function
# import cPickle,theano
from data_voip import load_data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from keras.models import model_from_json
import numpy as np

#load the saved model
#model = cPickle.load(open("model.pkl","rb"))
model = model_from_json(open("").read())
model.load_weights("")

#define theano funtion to get output of  FC layer
#get_feature = theano.function([model.layers[0].input],model.layers[11].output,allow_input_downcast=False) 

#define theano funtion to get output of  first Conv layer 
#get_featuremap = theano.function([model.layers[0].input],model.layers[2].output,allow_input_downcast=False) 
from keras import backend as K
get_feature = K.function([model.layers[0].input],[model.layers[1].output])
get_featuremap = K.function([model.layers[0].input],[model.layers[1].output])

data, label = load_data()
# visualize feature  of  Fully Connected layer
#data[0:10] contains 10 images

feature = get_feature([data[0:10]])[0]  #visualize these images's FC-layer feature
#feature = np.squeeze(np.array(feature))
#print(feature)
plt.imshow(feature,cmap = cm.Greys_r)
plt.show()

#visualize feature map of Convolution Layer
num_fmap = 4	#number of feature map
for i in range(num_fmap):
	featuremap = get_featuremap([data[0:10]])[0]
	#plt.imshow(featuremap[0][i],cmap = cm.Greys_r) #visualize the first image's 4 feature map
	plt.imshow(featuremap[0][:,:,i],cmap = cm.Greys_r)
	plt.show()

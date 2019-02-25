#coding=utf-8

from __future__ import print_function
# import cPickle,theano
from data_voip import load_data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from keras.models import model_from_json
import numpy as np



def getvoipfeature(rows=100):
	from keras import backend as K
	K.set_learning_phase(False)
	#load the saved model
	#model = cPickle.load(open("model.pkl","rb"))
	model = model_from_json(
	        open("../result/model_json_final_10/alexnet_model_architecture_" + str(rows) + ".json").read())
	model.load_weights("../result/model_json_final_10/alexnet_model_weights_" + str(rows) + ".h5")

	#define theano funtion to get output of  FC layer
	#get_feature = theano.function([model.layers[0].input],model.layers[11].output,allow_input_downcast=False) 

	#define theano funtion to get output of  first Conv layer 
	#get_featuremap = theano.function([model.layers[0].input],model.layers[2].output,allow_input_downcast=False) 

	get_feature = K.function([model.layers[0].input],[model.layers[-1].output])
	get_featuremap = K.function([model.layers[0].input],[model.layers[1].output])

	data, label = load_data(rows=rows)
	# visualize feature  of  Fully Connected layer
	#data[0:10] contains 10 images
	print(len(get_feature([data[0:10]])[0]))
	feature = get_feature([data[0:10]])[0]  #visualize these images's FC-layer feature
	#feature = np.squeeze(np.array(feature))
	#print(feature)
	plt.imshow(feature,cmap = cm.Greys_r)
	plt.show()

	#visualize feature map of Convolution Layer
	#featuremap = get_featuremap([data[0:10]])[0]

	num_fmap = 4	#number of feature map
	for i in range(num_fmap):
		featuremap = get_featuremap([data[0:10]])[0]
		#plt.imshow(featuremap[0][i],cmap = cm.Greys_r) #visualize the first image's 4 feature map
		plt.imshow(featuremap[1][:,:,i],cmap = cm.Greys_r)
		plt.savefig("../result/skype"+str(i)+".eps",format="eps")
		plt.show()

if __name__=="__main__":
	getvoipfeature(rows=40)

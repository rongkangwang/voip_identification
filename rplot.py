from data_voip import load_data
import random
from keras.models import model_from_json
import numpy as np

def acc4voip(rows=100):
	data, label = load_data(rows=rows)
	origin_model = model_from_json(open("../data/model_json/alexnet_model_architecture_"+str(rows)+".json").read())
	origin_model.load_weights("../data/model_json/alexnet_model_weights_"+str(rows)+".h5")
	pred_label = origin_model.predict(data,batch_size=1, verbose=1)
	num = len(label)
	voip_len = [0 for i in range(7)]
	for k in range(1,8):
		for i in range(num):
			if label[i]==np.argmax(pred_label[i]):
				voip_len[label[i]-1] += 1
	voip_acc = [voip_len[i]/(num/7.0) for i in range(7)]
	print(voip_acc)

if __name__=="__main__":
	acc4voip(rows=10)
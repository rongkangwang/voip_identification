#coding:utf-8

import os
from PIL import Image
import numpy as np
import platform


def load_data():
	data = np.empty((42000,1,28,28),dtype="float32")
	label = np.empty((42000,),dtype="uint8")
	imgs = os.listdir("../data/mnist")
	num = len(imgs)
	for i in range(num):
		img = Image.open("../data/mnist/"+imgs[i])
		arr = np.asarray(img,dtype="float32")
		data[i,:,:,:] = arr
		label[i] = int(imgs[i].split('.')[0])
	#归一化和零均值化
	data /= np.max(data)
	data -= np.mean(data)
	return data,label

def load_data():
    if (platform.uname()[0] == "Linux"):
        filepath = "/home/kang/Documents/data/224/"
    elif (platform.uname()[0] == "Darwin"):
        filepath = "/Users/kang/Documents/workspace/data/224/"

    d = []
    l = []
    for di in os.listdir(filepath):
        if(di=="skype" or di=="jumblo"):
            voipdir = os.path.join(filepath,di)
            for i in os.listdir(voipdir):
                if(i.__contains__(".png")):
                    img = Image.open(os.path.join(voipdir,i))
                    arr = np.asarray(img,dtype="float32")
                    arr = arr.reshape(224, 224, 1)
                    d.append(arr)
                    if(di=="skype"):
                        l.append(0)
                    elif(di=="jumblo"):
                        l.append(1)
                    elif (di == "kc"):
                        l.append(2)
                    elif (di == "uu"):
                        l.append(3)
                    elif (di == "alt"):
                        l.append(4)
    data = np.asarray(d,dtype="float32")
    label = np.asarray(l,dtype="float32")
    data /= np.max(data)
    data -= np.mean(data)
    return data,label
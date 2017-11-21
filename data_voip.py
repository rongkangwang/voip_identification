#coding:utf-8

import os
from PIL import Image
import numpy as np
import platform


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
    data = np.asarray(d,dtype="float32")
    label = np.asarray(l,dtype="float32")
    data /= np.max(data)
    data -= np.mean(data)
    return data,label
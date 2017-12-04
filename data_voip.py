#coding:utf-8

import os
from PIL import Image
import numpy as np
import platform

rows = 100
cols = 256
def load_data():
    if (platform.uname()[0] == "Linux"):
        filepath = "/home/kang/Documents/data/"+str(rows)+"/"
    elif (platform.uname()[0] == "Darwin"):
        filepath = "/Users/kang/Documents/workspace/data/"+str(rows)+"/"

    pnum = 0
    d = []
    l = []
    for di in os.listdir(filepath):
        if(di=="skype" or di=="jumblo" or di=="uu" or di=="xlite" or di=="zoiper"):
            voipdir = os.path.join(filepath,di)
            for i in os.listdir(voipdir):
                if(i.__contains__(".png")):
                    if(pnum>=1000):
                        pnum = 0
                        break
                    img = Image.open(os.path.join(voipdir,i))
                    arr = np.asarray(img,dtype="float32")
                    arr = arr.reshape(rows, cols, 1)
                    d.append(arr)
                    pnum = pnum+1
                    if(di=="skype"):
                        l.append(0)
                    elif(di=="jumblo"):
                        l.append(1)
                    elif (di == "xlite"):
                        l.append(2)
                    elif (di == "zoiper"):
                        l.append(3)
                    elif (di == "uu"):
                        l.append(4)
    data = np.asarray(d,dtype="float32")
    label = np.asarray(l,dtype="float32")
    data /= np.max(data)
    data -= np.mean(data)
    return data,label
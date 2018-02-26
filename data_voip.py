#coding:utf-8

import os
from PIL import Image
import numpy as np
import platform

rows = 100
cols = 256
def load_data(rows=100):
    if (platform.uname()[0] == "Linux"):
        filepath = "/home/kang/Documents/data/"+str(rows)+"/"
    elif (platform.uname()[0] == "Darwin"):
        filepath = "/Users/kang/Documents/workspace/data/"+str(rows)+"/"

    pnum = 0
    d = []
    l = []
    for di in os.listdir(filepath):
        if(di=="skype" or di=="jumblo" or di=="uu" or di=="xlite" or di=="zoiper" or di=="kc" or di=="alt" or di=="eyebeam" or di=="expresstalk" or di=="bria"):
            # print di
            pnum = 0
            voipdir = os.path.join(filepath,di)
            for i in os.listdir(voipdir):
                if(i.__contains__(".png")):
                    img = Image.open(os.path.join(voipdir,i))
                    arr = np.asarray(img,dtype="float32")
                    arr = arr.reshape(rows, cols, 1)
                    d.append(arr)
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
                    elif (di == "alt"):
                        l.append(5)
                    elif (di == "kc"):
                        l.append(6)
                    elif (di == "eyebeam"):
                        l.append(7)
                    elif (di == "expresstalk"):
                        l.append(8)
                    elif (di == "bria"):
                        l.append(9)
                    pnum = pnum+1
                    if(pnum>=10000):
                        print pnum
                        pnum = 0
                        break
    data = np.asarray(d,dtype="float32")
    label = np.asarray(l,dtype="float32")
    #data += 1
    data /= np.max(data)
    data -= np.mean(data)
    return data,label

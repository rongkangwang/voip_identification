#!/usr/bin/python
#coding=utf-8

# a='\xD3\x92'
# for x in a:
#     print ("%#x"%ord(x))

import struct
import numpy as np
import os

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
import random,cPickle
from keras.callbacks import EarlyStopping

num = 0
app_class = 3
dict = {"baidu":0,"dangdang":1,"jingdong":2}

def transferpacket2matrix(packet):
    p_len = len(packet)
    #print p_len
    m = np.zeros((1,256,54),dtype="float32")
    for i in range(54):
        c = packet[i]
        a_num = struct.unpack('B', c)[0]
        m[0][a_num][i] = 1
    return m

def pcap2packets(filename):
    fpcap = open(filename, 'rb')
    string_data = fpcap.read()
    # pcap header
    pcap_header = {}
    pcap_header['magic_number'] = string_data[0:4]
    pcap_header['version_major'] = string_data[4:6]
    pcap_header['version_minor'] = string_data[6:8]
    pcap_header['thiszone'] = string_data[8:12]
    pcap_header['sigfigs'] = string_data[12:16]
    pcap_header['snaplen'] = string_data[16:20]
    pcap_header['linktype'] = string_data[20:24]

    num = 0  # packet no.
    packets = []    # all of the packets
    pcap_packet_header = {}  # packet header
    i = 24    # pcap header takes 24 bytes
    while (i < len(string_data)):
        # paser packet header
        pcap_packet_header['GMTtime'] = string_data[i:i + 4]
        pcap_packet_header['MicroTime'] = string_data[i + 4:i + 8]
        pcap_packet_header['caplen'] = string_data[i + 8:i + 12]
        pcap_packet_header['len'] = string_data[i + 12:i + 16]
        # len is real, so use it
        packet_len = struct.unpack('I', pcap_packet_header['len'])[0]
        # add packet to packets
        packets.append(string_data[i + 16:i + 16 + packet_len])
        i = i + packet_len + 16
        num += 1
    fpcap.close()
    return packets

def preprocess(path):
    global num  # packet no.
    ftxt = open('packets.txt', 'w')
    for file in os.listdir(path):
        if(file.__contains__("pcap")):
            filename = os.path.join(path,file)
            fpcap = open(filename, 'rb')
            string_data = fpcap.read()

            # packets = []  # all of the packets
            pcap_packet_header = {}  # packet header
            i = 24  # pcap header takes 24 bytes
            while (i < len(string_data)):
                # paser packet header
                pcap_packet_header['GMTtime'] = string_data[i:i + 4]
                pcap_packet_header['MicroTime'] = string_data[i + 4:i + 8]
                pcap_packet_header['caplen'] = string_data[i + 8:i + 12]
                pcap_packet_header['len'] = string_data[i + 12:i + 16]
                # len is real, so use it
                packet_len = struct.unpack('I', pcap_packet_header['len'])[0]
                # add packet to packets
                ftxt.write(repr(string_data[i + 16:i + 16 + packet_len]) + '\n')
                # packets.append(string_data[i + 16:i + 16 + packet_len])
                i = i + packet_len + 16
                num += 1
            fpcap.close()
    return num

def preprocess(path):
    global num  # packet no.

    for p in os.listdir(path):
        if(os.path.isdir(os.path.join(path,p))):
            ftxt = open("result/"+p+'.txt', 'w')
            for file in os.listdir(os.path.join(path,p)):
                if(file.__contains__("pcap")):
                    # print file
                    filename = os.path.join(os.path.join(path,p),file)
                    fpcap = open(filename, 'rb')
                    string_data = fpcap.read()

                    # packets = []  # all of the packets
                    pcap_packet_header = {}  # packet header
                    i = 24  # pcap header takes 24 bytes
                    while (i < len(string_data)):
                        # paser packet header
                        pcap_packet_header['GMTtime'] = string_data[i:i + 4]
                        pcap_packet_header['MicroTime'] = string_data[i + 4:i + 8]
                        pcap_packet_header['caplen'] = string_data[i + 8:i + 12]
                        pcap_packet_header['len'] = string_data[i + 12:i + 16]
                        # len is real, so use it
                        packet_len = struct.unpack('I', pcap_packet_header['len'])[0]
                        # add packet to packets
                        ftxt.write(repr(string_data[i + 16:i + 16 + packet_len]) + '\n')
                        # packets.append(string_data[i + 16:i + 16 + packet_len])
                        i = i + packet_len + 16
                        num += 1
                    fpcap.close()
            ftxt.close()
    return num

def create_model():
	model = Sequential()
	model.add(Conv2D(4, (5, 5), padding='valid',input_shape=(256,54,1)))
	model.add(Activation('relu'))

	model.add(Conv2D(8,(3, 3), padding='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(16,(3, 3), padding='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(128, kernel_initializer='normal'))
	model.add(Activation('relu'))

	model.add(Dense(app_class, kernel_initializer='normal'))
	model.add(Activation('softmax'))
	return model

if __name__ == "__main__":
    resultpath = "result/"
    path = "/Users/kang/Documents/workspace/voip_identification/pcap"
    num = preprocess(path)
    print num
    data = np.zeros((num,1,256,54))
    labels = np.empty((num,),dtype="uint8")
    i = 0
    for file in os.listdir(resultpath):
        flag = file.split(".",1)[0]
        type = dict[flag]
        filename = os.path.join(resultpath, file)
        f_obj = open(filename,"rb")

        for line in f_obj:
            m = transferpacket2matrix(line)
            data[i,:,:,:] = m
            labels[i] = type  #应用类型有几种
            i += 1
    labels = np_utils.to_categorical(labels, app_class)
    # 创建模型 设置随即梯度下降
    model = create_model()
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    # shuffle
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]
    labels = labels[index]
    split_num = num/4*3
    (X_train, X_val) = (data[0:split_num], data[split_num:])
    (Y_train, Y_val) = (labels[0:split_num], labels[split_num:])
    X_train = X_train.reshape(X_train.shape[0], 256, 54, 1)
    X_val = X_val.reshape(X_val.shape[0], 256, 54, 1)

    early_stopping = EarlyStopping(monitor='val_loss', patience=1)
    model.fit(X_train, Y_train, batch_size=1000, validation_data=(X_val, Y_val), epochs=3, callbacks=[early_stopping])
    json_string = model.to_json()
    open('my_model_architecture.json', 'w').write(json_string)
    model.save_weights('my_model_weights.h5')


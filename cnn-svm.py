'''
Author:wepon
Code:https://github.com/wepe
File: cnn-svm.py
'''
from __future__ import print_function
import cPickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from data_voip import load_data
import random
from keras.models import model_from_json
import numpy as np


def svc(traindata,trainlabel,testdata,testlabel):
    print("Start training SVM...")
    svcClf = SVC(C=1.0,kernel="rbf",cache_size=3000)
    svcClf.fit(traindata,trainlabel)
    
    pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("cnn-svm Accuracy:",accuracy)

def rf(traindata,trainlabel,testdata,testlabel):
    print("Start training Random Forest...")
    rfClf = RandomForestClassifier(n_estimators=400,criterion='gini')
    rfClf.fit(traindata,trainlabel)
    
    pred_testlabel = rfClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("cnn-rf Accuracy:",accuracy)

if __name__ == "__main__":
    #load data
    data, label = load_data()
    #shuffle the data
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]
    label = label[index]
    print(type(index))
    # print(data)
    # data = data.reshape(data.shape[0], 28, 28, 1)

    (traindata,testdata) = (data[0:3000],data[3000:15000])
    (trainlabel,testlabel) = (label[0:3000],label[3000:15000])
    # traindata = traindata.reshape(traindata.shape[0], 28, 28, 1)
    # testdata = testdata.reshape(testdata.shape[0], 28, 28, 1)

    # use origin_model to predict testdata
    # origin_model = cPickle.load(open("model.pkl","rb"))
    origin_model = model_from_json(open("cnn_model_architecture_224.json").read())
    origin_model.load_weights("cnn_model_weights_224.h5")
    # origin_model = tf2th(origin_model)
    #print(origin_model.layers)
    pred_testlabel = origin_model.predict(testdata,batch_size=1, verbose=1)
    print(pred_testlabel)
    file = open('cnn_pred_testlabel_224', 'w')
    for pred in pred_testlabel:
        file.write(str(pred[0])+" "+str(pred[1])+" "+str(pred[2])+" "+str(pred[3])+" "+str(pred[4])+"\r\n")
    file.close()
    num = len(testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==np.argmax(pred_testlabel[i])])/float(num)
    # for squential model
    # accuracy = len([1 for i in range(num) if testlabel[i] == pred_testlabel[i]]) / float(num)
    print(" Origin_model Accuracy:",accuracy)
    # extract the feature using keras from first layer to last layer
    # from keras import backend as K
    # get_feature = K.function([origin_model.layers[0].input],[origin_model.layers[-1].output])
    # feature = get_feature([data])[0]
    #
    # #define theano funtion to get output of FC layer
    # # get_feature = theano.function([origin_model.layers[0].input],origin_model.layers[-1].output,allow_input_downcast=False)
    # # feature = get_feature(data)
    # #train svm using FC-layer feature
    # scaler = MinMaxScaler()
    # feature = scaler.fit_transform(feature)
    # svc(feature[0:5000],label[0:5000],feature[5000:],label[5000:])
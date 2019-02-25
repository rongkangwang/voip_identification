'''
Author:wepon
Code:https://github.com/wepe
File: cnn-svm.py
'''
from __future__ import print_function
# import cPickle
from data_voip import load_data
import random
from keras.models import model_from_json
import numpy as np


def svc(traindata,trainlabel,testdata,testlabel):
    print("Start training SVM...")
    svcClf = SVC(C=1.0,kernel="rbf",cache_size=20000,verbose=True)
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

def nabayes(traindata,trainlabel,testdata,testlabel):
    print("Start training Navie Bayes...")
    bayes = naive_bayes.GaussianNB()
    bayes.fit(traindata,trainlabel)
    
    pred_testlabel = bayes.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("cnn-naviebayes Accuracy:",accuracy)

def dtree(traindata,trainlabel,testdata,testlabel):
    print("Start training Decision Tree...")
    dtree = tree.DecisionTreeClassifier()
    dtree.fit(traindata,trainlabel)
    
    pred_testlabel = dtree.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("cnn-naviebayes Accuracy:",accuracy)


rows = 20

if __name__ == "__main__":
    #load data
    data, label = load_data(rows=rows)
    #shuffle the data
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]
    label = label[index]
    print(type(index)) 

    (traindata,testdata) = (data[0:800],data[800:])
    (trainlabel,testlabel) = (label[0:800],label[800:])

    # use origin_model to predict testdata
    # origin_model = model_from_json(open("../result/model_json_final_10/alexnet_model_architecture_"+str(rows)+".json").read())
    # origin_model.load_weights("../result/model_json_final_10/alexnet_model_weights_"+str(rows)+".h5")
    # pred_testlabel = origin_model.predict(testdata,batch_size=1, verbose=1)
    # print(pred_testlabel)
    # file = open('../data/model_json/alexnet_pred_testlabel_'+str(rows), 'w')
    # for pred in pred_testlabel:
    #     file.write(str(pred[0])+" "+str(pred[1])+" "+str(pred[2])+" "+str(pred[3])+" "+str(pred[4])+" "+str(pred[5])+" "+str(pred[6])+"\r\n")
    # file.close()
    # num = len(testlabel)
    # accuracy = len([1 for i in range(num) if testlabel[i]==np.argmax(pred_testlabel[i])])/float(num)
    # # for squential model
    # print(" Origin_model Accuracy:",accuracy)



    #extract the feature using keras from first layer to last layer
    from keras import backend as K
    K.set_learning_phase(False)
    model = model_from_json(
            open("../result/model_json_final_10/alexnet_model_architecture_" + str(rows) + ".json").read())
    model.load_weights("../result/model_json_final_10/alexnet_model_weights_" + str(rows) + ".h5")

    #define theano funtion to get output of  FC layer
    #get_feature = theano.function([model.layers[0].input],model.layers[11].output,allow_input_downcast=False) 

    #define theano funtion to get output of  first Conv layer 
    #get_featuremap = theano.function([model.layers[0].input],model.layers[2].output,allow_input_downcast=False) 

    get_feature = K.function([model.layers[0].input],[model.layers[-1].output])
    #get_feature = K.function([origin_model.layers[0].input],[origin_model.layers[-1].output])
    feature = get_feature([data])[0]
    
    #define theano funtion to get output of FC layer
    # get_feature = theano.function([origin_model.layers[0].input],origin_model.layers[-1].output,allow_input_downcast=False)
    # feature = get_feature(data)
    #train svm using FC-layer feature
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import MinMaxScaler
    from sklearn import naive_bayes
    from sklearn import tree
    scaler = MinMaxScaler()
    feature = scaler.fit_transform(feature)
    nabayes(feature[0:800],label[0:800],feature[800:],label[800:])

from data_voip import load_data
import random
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt

def getpretable(rows=100):
    model = model_from_json(
        open("../result/model_json_final/alexnet_model_architecture_" + str(rows) + ".json").read())
    model.load_weights("../result/model_json_final/alexnet_model_weights_" + str(rows) + ".h5")
    data ,label = load_data(rows=rows)
    pred_label = model.predict(data,batch_size=1, verbose=1)
    print(type(pred_label[0]))
    file = open('../result/pred/pred_label_'+str(rows), 'w')
    for pred in pred_label:
        file.write(str(pred[0])+" "+str(pred[1])+" "+str(pred[2])+" "+str(pred[3])+" "+str(pred[4])+" "+str(pred[5])+" "+str(pred[6])+"\r\n")
    file.close()
    file_label = open('../result/pred/label_'+str(rows), 'w')
    for l in label:
        file_label.write(str(l)+"\r\n")
    file_label.close()

class plotmodel:
    def __init__(self,rows=100):
        self.rows = rows
        pred = []
        file = open('../result/pred/pred_label_'+str(self.rows), 'r')
        for line in file:
            label = line.strip().split(" ")
            #folat_label = np.asarray(label,dtype="float")
            pred.append(label)
        self.pred_label = np.asarray(pred,dtype="float")
        tru = []
        file = open('../result/pred/label_'+str(self.rows), 'r')
        for line in file:
            label = line.strip()
            tru.append(label)
        floatlabel = np.asarray(tru,dtype="float")
        self.label = np.asarray(floatlabel,dtype="int")
        self.labelnum = len(self.pred_label)
        self.overallacc = self.overallacc()
        
        # self.precision = self.precision()
        # self.recall = self.recall()
        # self.fscore = self.fscore()

    def overallacc(self):
        accuracy = len([1 for i in range(self.labelnum) if self.label[i]==np.argmax(self.pred_label[i])])/float(self.labelnum)
        return accuracy

    def acc4voip(self):   # Class detection rate
        voip_len = [0 for i in range(7)]
        for i in range(self.labelnum):
            if self.label[i] == np.argmax(self.pred_label[i]):
                index = self.label[i].astype(np.int64)
                voip_len[index] += 1
        voip_acc = [voip_len[i] / (self.labelnum / 7.0) for i in range(7)]
        print(voip_acc)
        return voip_acc
        
    def fprandfnr(self):

        return

    def precision(self):  #ppv Positive Predictive Value
        tp = [0.0 for i in range(7)]
        fp = [0.0 for i in range(7)]
        for i in range(self.labelnum):
            predict = np.argmax(self.pred_label[i])
            if self.label[i] == predict:
                tp[predict] += 1.0
            else:
                fp[predict] += 1.0
        voip_prec = [tp[i]/(tp[i]+fp[i]) for i in range(7)]
        # print tp
        # print fp
        print voip_prec
        return voip_prec

    def recall(self):
        tp = [0.0 for i in range(7)]
        fn = [0.0 for i in range(7)]
        for i in range(self.labelnum):
            predict = np.argmax(self.pred_label[i])
            if self.label[i] == predict:
                tp[predict] += 1.0
            else:
                index = self.label[i].astype(np.int64)
                fn[index] += 1.0
        voip_recall = [tp[i]/(tp[i]+fn[i]) for i in range(7)]
        print voip_recall
        return voip_recall

    def fpr(self):
        fp = [0.0 for i in range(7)]
        tn = [0.0 for i in range(7)]
        for i in range(self.labelnum):
            predict = np.argmax(self.pred_label[i])
            if self.label[i] == predict:
                for j in range(7):
                    if j != predict:
                        tn[j] += 1
            else:
                fp[predict] += 1.0
        voip_fpr = [fp[i]/(fp[i]+tn[i]) for i in range(7)]
        return voip_fpr

    def far(self):   # Class FAR or class FP rate
        f = [0.0 for i in range(7)]
        for i in range(self.labelnum):
            predict = np.argmax(self.pred_label[i])
            if self.label[i] != predict:
                index = self.label[i].astype(np.int64)
                f[index] += 1.0
        far = [f[i]/(self.labelnum/7.0*6.0) for i in range(7)]
        return far

    def fscore(self):
        return 2*self.precision*self.recall/(self.precision+self.recall)

    def unknowpkt(self):
        return

def drawpicture(pmodel10,pmodel20,pmodel40,pmodel100):
    x = [10,20,40,100]
    y = [pmodel10.overallacc,pmodel20.overallacc,pmodel40.overallacc,pmodel100.overallacc]
    plt.plot(x,y,".")
    plt.savefig("../result/plot/overallacc.eps",format="eps")
    plt.show()

    acc10 =  tuple(pmodel10.acc4voip())
    acc20 =  tuple(pmodel20.acc4voip())
    acc40 =  tuple(pmodel40.acc4voip())
    acc100 =  tuple(pmodel100.acc4voip())
    n_groups = 7
    index = np.arange(n_groups)  
    bar_width = 0.15
    opacity=0.4
    rects10 = plt.bar(index, acc10, bar_width,alpha=opacity, color='b',label='10')
    rects20 = plt.bar(index + bar_width, acc20, bar_width,alpha=opacity,color='r',label='20')
    rects40 = plt.bar(index + 2*bar_width, acc40, bar_width,alpha=opacity, color='y',label='40')
    rects200 = plt.bar(index + 3*bar_width, acc100, bar_width,alpha=opacity,color='m',label='100')

    # plt.xlabel('Model')  
    plt.ylabel('Accuracy')  
    # plt.title('')  

    plt.xticks(index+2*bar_width-0.075,('skype','jumblo','xlite','zoiper','uucall','altcall','kccall'),fontsize=18)
    plt.yticks(fontsize =18)  #change the num axis size

    plt.ylim(0.0,1.0)  #The ceil
    plt.legend()  

    plt.tight_layout()
    plt.savefig("../result/plot/accbar.eps",format="eps")
    plt.show()



if __name__=="__main__":
    for row in [10,20,40,100]:
        getpretable(rows=row)
    pmodel10 = plotmodel(rows=10)
    pmodel20 = plotmodel(rows=20)
    pmodel40 = plotmodel(rows=40)
    pmodel100 = plotmodel(rows=100)
    drawpicture(pmodel10,pmodel20,pmodel40,pmodel100)
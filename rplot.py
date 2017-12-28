from data_voip import load_data
import random
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt

class plotmodel:
    def __init__(self,rows=100):
        self.rows = rows
        model = model_from_json(
            open("../result/model_json_final/alexnet_model_architecture_" + str(rows) + ".json").read())
        model.load_weights("../result/model_json_final/alexnet_model_weights_" + str(rows) + ".h5")
        self.model = model
        self.data ,self.label = load_data(rows=rows)
        self.pred_label = model.predict(self.data,batch_size=1, verbose=1)
        self.labelnum = len(self.label)
        self.overallacc = self.overallacc()
        
        # self.precision = self.precision()
        # self.recall = self.recall()
        # self.fscore = self.fscore()

    def overallacc(self):
        accuracy = len([1 for i in range(self.labelnum) if self.label[i]==self.pred_label[i]])/float(self.labelnum)
        return accuracy

    def acc4voip(self):   # Class detection rate
        voip_len = [0 for i in range(7)]
        for i in range(self.labelnum):
            if self.label[i] == np.argmax(self.pred_label[i]):
                index = self.label[i].astype(np.int64)
                voip_len[index] += 1
        voip_acc = [voip_len[i] / (self.labelnum / 7.0) for i in range(7)]
        print(voip_acc)

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

def drawpicture(pmodel10,pmodel20):
    x = [10,20]
    y = [pmodel10.overallacc,pmodel20.overallacc]
    plt.plot(x,y,".")
    plt.show()




if __name__=="__main__":
    pmodel10 = plotmodel(rows=10)
    pmodel20 = plotmodel(rows=20)
    # pmodel40 = plotmodel(rows=40)
    # pmodel100 = plotmodel(rows=100)
    drawpicture(pmodel10,pmodel20)
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
        self.cm = self.confusedmatrix()
        self.overallacc = self.overallacc()
        
        # self.precision = self.precision()
        # self.recall = self.recall()
        # self.fscore = self.fscore()

    def confusedmatrix(self):
        m = [ [ 0 for j in range(7) ] for i in range(7) ]
        for i in range(self.labelnum):
            c = np.argmax(self.pred_label[i])
            r = self.label[i]
            m[r][c] += 1
        return m

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

    def precisionbycm(self):
        l = len(self.cm)
        prec = [0.0 for i in range(l)]
        for i in range(l):
            colsum = sum(self.cm[r][i] for r in range(l))
            prec[i] = float(self.cm[i][i])/colsum
        return prec

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

    def recallbycm(self):
        l = len(self.cm)
        recall = [0.0 for i in range(l)]
        for i in range(l):
            rowsum = sum(self.cm[i])
            recall[i] = float(self.cm[i][i]) / rowsum
        return recall

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

def get100pcap(rows=100,cols=256):
    d = [] # data
    l = [] # label

    from pcap2image import pcap2packetspayload
    from pcap2image import transferpacket2matrixbycols
    for i,filename in enumerate(["skype.pcap","jumblo.pcap","xlite.pcap","zoiper.pcap","uu.pcap","alt.pcap","kc.pcap"]):
        (packets, max_len) = pcap2packetspayload("../data/unknow/"+filename)
        r = random.randint(10, 100)
        num = 0
        while num < 15:
            f = 0
            m = np.zeros((rows, cols), dtype="float32")
            while f < rows and num*r+f < len(packets):
                m[f] = transferpacket2matrixbycols(packets[r+f], cols)
                f += 1
            num += 1
            d.append(m)
            l.append(i)
    d = d[0:100]
    l = l[0:100]
    data = np.asarray(d, dtype="float32")
    data = data.reshape(data.shape[0],rows, cols, 1)
    label = np.asarray(l, dtype="float32")
    data /= np.max(data)
    data -= np.mean(data)
    return data,label

def getpretable(rows=100):
    model = model_from_json(
        open("../result/model_json_final/alexnet_model_architecture_" + str(rows) + ".json").read())
    model.load_weights("../result/model_json_final/alexnet_model_weights_" + str(rows) + ".h5")
    data ,label = get100pcap(rows=rows)
    pred_label = model.predict(data,batch_size=1, verbose=1)
    print(type(pred_label[0]))
    file = open('../result/pred100/100pred_label_'+str(rows), 'w')
    for pred in pred_label:
        file.write(str(pred[0])+" "+str(pred[1])+" "+str(pred[2])+" "+str(pred[3])+" "+str(pred[4])+" "+str(pred[5])+" "+str(pred[6])+"\r\n")
    file.close()
    # file_label = open('../result/pred/100label_'+str(rows), 'w')
    # for l in label:
    #     file_label.write(str(l)+"\r\n")
    # file_label.close()

if __name__=="__main__":
    # for row in [10,20,40,100]:
    #     getpretable(rows=row)
    pmodel10 = plotmodel(rows=10)
    pmodel20 = plotmodel(rows=20)
    pmodel40 = plotmodel(rows=40)
    pmodel100 = plotmodel(rows=100)
    # drawpicture(pmodel10,pmodel20,pmodel40,pmodel100)
    print(pmodel10.confusedmatrix())
    print(pmodel10.precision())
    print(pmodel10.precisionbycm())
    print(pmodel10.recall())
    print(pmodel10.recallbycm())
    # getpretable(rows=10)
    # getpretable(rows=20)
    # getpretable(rows=40)
    # getpretable(rows=100)


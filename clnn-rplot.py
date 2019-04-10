from data_voip import load_data
import random
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt

def getpretable(rows=100):
    model = model_from_json(
        open("../clnn/model_json/alexnet_model_architecture_" + str(rows) + ".json").read())
    model.load_weights("../clnn/model_json/alexnet_model_weights_" + str(rows) + ".h5")
    data ,label = load_data(rows=rows)
    import time
    start_time = time.time()
    pred_label = model.predict(data,batch_size=1, verbose=1)
    end_time = time.time()
    print str(rows)+"->"+str(end_time-start_time)
    print(type(pred_label[0]))
    file = open('../clnn/result/pred/pred_label_'+str(rows), 'w')
    for pred in pred_label:
        file.write(str(pred[0])+" "+str(pred[1])+" "+str(pred[2])+" "+str(pred[3])+" "+str(pred[4])+" "+str(pred[5])+" "+str(pred[6])+" "+str(pred[7])+" "+str(pred[8])+" "+str(pred[9])+"\r\n")
    file.close()
    file_label = open('../clnn/result/pred/label_'+str(rows), 'w')
    for l in label:
        file_label.write(str(l)+"\r\n")
    file_label.close()

class plotmodel:
    def __init__(self,rows=100):
        self.rows = rows
        pred = []
        file = open('../clnn/result/pred/pred_label_'+str(self.rows), 'r')
        for line in file:
            label = line.strip().split(" ")
            #folat_label = np.asarray(label,dtype="float")
            pred.append(label)
        self.pred_label = np.asarray(pred,dtype="float")
        tru = []
        file = open('../clnn/result/pred/label_'+str(self.rows), 'r')
        for line in file:
            label = line.strip()
            tru.append(label)
        floatlabel = np.asarray(tru,dtype="float")
        self.label = np.asarray(floatlabel,dtype="int")
        self.labelnum = len(self.pred_label)
        self.cm = self.confusedmatrix()
        self.overallacc = self.overallacc()
        
        self.precision = self.precision()
        self.recall = self.recall()
        #self.fpr = self.fpr()
	self.fpr = self.fprbycm()
        self.tpr = self.tprbycm()
	self.fnr = self.fnr()
        self.tnr = self.tnr()
        self.fscore = self.fscore()

    def confusedmatrix(self):
        m = [ [ 0 for j in range(10) ] for i in range(10) ]
        for i in range(self.labelnum):
            c = np.argmax(self.pred_label[i])  #col : pred label
            r = self.label[i]                  #row : real label
            m[r][c] += 1
        return m

    def overallacc(self):
        accuracy = len([1 for i in range(self.labelnum) if self.label[i]==np.argmax(self.pred_label[i])])/float(self.labelnum)
        return accuracy

    def acc4voip(self):   # Class detection rate
        voip_len = [0 for i in range(10)]
        for i in range(self.labelnum):
            if self.label[i] == np.argmax(self.pred_label[i]):
                index = self.label[i].astype(np.int64)
                voip_len[index] += 1
        voip_acc = [voip_len[i] / (self.labelnum / 10.0) for i in range(10)]
        print(voip_acc)
        return voip_acc
        
    def fprandfnr(self):

        return

    def precision(self):  #ppv Positive Predictive Value
        tp = [0.0 for i in range(10)]
        fp = [0.0 for i in range(10)]
        for i in range(self.labelnum):
            predict = np.argmax(self.pred_label[i])
            if self.label[i] == predict:
                tp[predict] += 1.0
            else:
                fp[self.label[i]] += 1.0
        voip_prec = [tp[i]/(tp[i]+fp[i]) for i in range(10)]
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
        tp = [0.0 for i in range(10)]
        fn = [0.0 for i in range(10)]
        for i in range(self.labelnum):
            predict = np.argmax(self.pred_label[i])
            if self.label[i] == predict:
                tp[predict] += 1.0
            else:
                index = self.label[i].astype(np.int64)
                fn[index] += 1.0
        voip_recall = [tp[i]/(tp[i]+fn[i]) for i in range(10)]
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
        fp = [0.0 for i in range(10)]
        tn = [0.0 for i in range(10)]
        for i in range(self.labelnum):
            predict = np.argmax(self.pred_label[i])
            if self.label[i] == predict:
                for j in range(10):
                    if j != predict:
                        tn[j] += 1
            else:
                fp[predict] += 1.0
        voip_fpr = [fp[i]/(fp[i]+tn[i]) for i in range(10)]
        return voip_fpr

    def tprbycm(self):
        l = len(self.cm)
        tpr = [0.0 for i in range(l)]
        for i in range(l):
            rowsum = sum(self.cm[i])   # rowsum : tp+fn 
            tpr[i] = float(self.cm[i][i])/rowsum
        return tpr

    def fprbycm(self):
        l = len(self.cm)
        fpr = [0.0 for i in range(l)]
        for i in range(l):
            # colsum = sum(self.cm[r][i] for r in range(l))
	    fp_num = sum(self.cm[r][i] for r in range(l) if r != i)
	    tn_num = sum(self.cm[r][r] for r in range(l) if r != i)
            fpr[i] = float(fp_num)/(fp_num+tn_num)
        return fpr

    def fnr(self):
        l = len(self.cm)
        fnr = [0.0 for i in range(l)]
        for i in range(l):
	    fnr[i] = 1.0-self.tpr[i]
	return fnr
    def tnr(self):
        l = len(self.cm)
        tnr = [0.0 for i in range(l)]
        for i in range(l):
	    tnr[i] = 1.0-self.fpr[i]
	return tnr


    def far(self):   # Class FAR or class FP rate
        f = [0.0 for i in range(10)]
        for i in range(self.labelnum):
            predict = np.argmax(self.pred_label[i])
            if self.label[i] != predict:
                index = self.label[i].astype(np.int64)
                f[index] += 1.0
        far = [f[i]/(self.labelnum/10.0*9.0) for i in range(10)]
        return far

    def fscore(self):
        l = len(self.cm)
        fscore = [0.0 for i in range(l)]
        for i in range(l):
	    fscore[i] = 2*self.precision[i]*self.recall[i]/(self.precision[i]+self.recall[i])
        return fscore

    def unknowpkt(self):
        return

def drawpicture(models):
    x = []
    y = []
    for m in models:
        x.append(m.rows)
        y.append(m.overallacc)
    plt.plot(x,y,".")
    plt.savefig("../result/plot_10/overallacc.eps",format="eps")
    plt.show()

#    acc10 =  tuple(pmodel10.acc4voip())
#    acc20 =  tuple(pmodel20.acc4voip())
#    acc40 =  tuple(pmodel40.acc4voip())
#    acc100 =  tuple(pmodel100.acc4voip())
#    n_groups = 10
#    index = np.arange(n_groups)  
#    bar_width = 0.15
#    opacity=0.4
#    rects10 = plt.bar(index, acc10, bar_width,alpha=opacity, color='b',label='10')
#    rects20 = plt.bar(index + bar_width, acc20, bar_width,alpha=opacity,color='r',label='20')
#    rects40 = plt.bar(index + 2*bar_width, acc40, bar_width,alpha=opacity, color='y',label='40')
#    rects200 = plt.bar(index + 3*bar_width, acc100, bar_width,alpha=opacity,color='m',label='100')
#
#    # plt.xlabel('Model')  
#    plt.ylabel('Accuracy')  
#    # plt.title('')  
#
#    plt.xticks(index+2*bar_width-0.075,('skype','jumblo','xlite','zoiper','uucall','altcall','kccall'),fontsize=18)
#    plt.yticks(fontsize =18)  #change the num axis size
#
#    plt.ylim(0.0,1.0)  #The ceil
#    plt.legend()  
#
#    plt.tight_layout()
#    plt.savefig("../result/plot_10/accbar.eps",format="eps")
#    plt.show()

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

def get100pretable(rows=100):
    model = model_from_json(
        open("../result/model_json_final/alexnet_model_architecture_" + str(rows) + ".json").read())
    model.load_weights("../result/model_json_final/alexnet_model_weights_" + str(rows) + ".h5")
    data ,label = get100pcap(rows=rows)
    pred_label = model.predict(data,batch_size=1, verbose=1)
    print(type(pred_label[0]))
    file = open('../result/pred100/100pred_label_'+str(rows), 'w')
    for pred in pred_label:
        file.write(str(pred[0])+" "+str(pred[1])+" "+str(pred[2])+" "+str(pred[3])+" "+str(pred[4])+" "+str(pred[5])+" "+str(pred[6])+" "+str(pred[7])+" "+str(pred[8])+" "+str(pred[9])+"\r\n")
    file.close()
    # file_label = open('../result/pred/100label_'+str(rows), 'w')
    # for l in label:
    #     file_label.write(str(l)+"\r\n")
    # file_label.close()

def saveresults(models):
    file = open('../result/evaluation/data', 'w')
    for model in models:
        file.write("rows->"+str(model.rows)+", overallacc->"+str(model.overallacc)+",\r\n precision->"+str(model.precision)+", \r\nrecall->"+str(model.recall)+", \r\nfpr->"+str(model.fpr)+", tpr->"+str(model.tpr)+"\r\n\r\n")
    file.close()

def save2xlsx(models):
    import xlwt
    workbook = xlwt.Workbook(encoding='utf-8')
    tables = []
    table = workbook.add_sheet('Skype') #0
    tables.append(table)
    table = workbook.add_sheet('Jumblo') #1
    tables.append(table)
    table = workbook.add_sheet('Xlite') #2
    tables.append(table)
    table = workbook.add_sheet('Zoiper') #3
    tables.append(table)
    table = workbook.add_sheet('UU') #4
    tables.append(table)
    table = workbook.add_sheet('ALT') #5
    tables.append(table)
    table = workbook.add_sheet('KC') #6
    tables.append(table)
    table = workbook.add_sheet('Eyebeam') #7
    tables.append(table)
    table = workbook.add_sheet('ExpressTalk') #8
    tables.append(table)
    table = workbook.add_sheet('Bria') #9
    tables.append(table)
    
    for sheet in tables:
	sheet.write(0,0,"Row")
	sheet.write(0,1,"Precision")
	sheet.write(0,2,"Recall")
	sheet.write(0,3,"FPR")
	sheet.write(0,4,"TPR")
	sheet.write(0,5,"FNR")
	sheet.write(0,6,"TNR")
	sheet.write(0,7,"F-Score")
	sheet.write(1,0,2)
	sheet.write(2,0,4)
	sheet.write(3,0,6)
	sheet.write(4,0,8)
	sheet.write(5,0,10)
	sheet.write(6,0,20)
	sheet.write(7,0,40)
	sheet.write(8,0,100)
    for i,sheet in enumerate(tables):
        for j,model in enumerate(models):
	    sheet.write(j+1,1,model.precision[i])
	    sheet.write(j+1,2,model.recall[i])
	    sheet.write(j+1,3,model.fpr[i])
	    sheet.write(j+1,4,model.tpr[i])
	    sheet.write(j+1,5,model.fnr[i])
	    sheet.write(j+1,6,model.tnr[i])
	    sheet.write(j+1,7,model.fscore[i])
    workbook.save("../clnn/result/results.xlsx")

def savepredict2xlsx(row):
    print row
    import xlwt
    workbook = xlwt.Workbook(encoding='utf-8')
    tables = []
    table = workbook.add_sheet('Skype') #0
    tables.append(table)
    table = workbook.add_sheet('Jumblo') #1
    tables.append(table)
    table = workbook.add_sheet('Xlite') #2
    tables.append(table)
    table = workbook.add_sheet('Zoiper') #3
    tables.append(table)
    table = workbook.add_sheet('UU') #4
    tables.append(table)
    table = workbook.add_sheet('ALT') #5
    tables.append(table)
    table = workbook.add_sheet('KC') #6
    tables.append(table)
    table = workbook.add_sheet('Eyebeam') #7
    tables.append(table)
    table = workbook.add_sheet('ExpressTalk') #8
    tables.append(table)
    table = workbook.add_sheet('Bria') #9
    tables.append(table)
    for sheet in tables:
	sheet.write(0,0,"P_index")
	sheet.write(0,1,"R_index")
	sheet.write(0,2,"P_pro")
	sheet.write(0,3,"R_pro")
    #for row in [2,4,6,8,10,20,40,100]:

    pred = []
    file = open('../result/pred_final/pred_1000/pred_label_'+str(row), 'r')
    for line in file:
        label = line.strip().split(" ")
            #folat_label = np.asarray(label,dtype="float")
        pred.append(label)
    pred_label = np.asarray(pred,dtype="float")
    tru = []
    file = open('../result/pred_final/pred_1000/label_'+str(row), 'r')
    for line in file:
        label = line.strip()
        tru.append(label)
    tru_label = np.asarray(tru,dtype="float")
    #num = len(tru_label)
    for i,sheet in enumerate(tables):
	r = 1
	for j in range(i*1000,(i+1)*1000):
	    #r = j-i*1000+1
	    pred = np.argmax(pred_label[j])
	    tru = int(tru_label[j])
	    if pred!=tru:
	        sheet.write(r,0,pred)
	        sheet.write(r,1,tru)
	        sheet.write(r,2,pred_label[j][pred])
	        sheet.write(r,3,pred_label[j][tru])
		r = r+1
    workbook.save("pred_"+str(row)+".xlsx")
	    

    

def snsdrawsub():
    import pandas as pd
    import seaborn as sns
    import xlrd
    #book = xlrd.open_workbook("results_unknown.xlsx")
    #sheet = book.sheet_by_name("Skype")
    #print sheet.name
    #for sheet in book.sheets():
	#print sheet.name
        #data = pd.read_excel("results_unknown.xlsx",sheet_name=sheet.name)
        #print data
    sns.set_style("dark")
    sns.despine(right=True)
    fig, axes = plt.subplots(3,4)
    book = xlrd.open_workbook("results_unknown.xlsx")
    for i,sheet in enumerate(book.sheets()):
        data = pd.read_excel("results_unknown.xlsx",sheet_name=i)
        x1 = pd.Series(np.array([2,4,6,8,10,20,40,100]))
        y1 = data.iloc[:,[4,6]]
        
        y1.plot(title=sheet.name,x=x1,style=["-gd","-bd"],ax=axes[i/4][i-4*(i/4)])
    plt.show()

def snsdraw():
    import pandas as pd
    import seaborn as sns
    import xlrd
    #book = xlrd.open_workbook("results_unknown.xlsx")
    #sheet = book.sheet_by_name("Skype")
    #print sheet.name
    #for sheet in book.sheets():
	#print sheet.name
        #data = pd.read_excel("results_unknown.xlsx",sheet_name=sheet.name)
        #print data
    sns.set_style("whitegrid")
    sns.despine()
    sns.set_palette("husl",2)
    #sns.despine(right=True)
    #fig, axes = plt.subplots(3,4)
    book = xlrd.open_workbook("results_unknown.xlsx")
    for i,sheet in enumerate(book.sheets()):
	data = pd.read_excel("results_unknown.xlsx",sheet_name=i)
	x1 = pd.Series(np.array([2,4,6,8,10,20,40,100]))
	y1 = data.iloc[:,[4,6]]
	y1.plot(title=sheet.name,x=x1,style=["-o","-o"]).legend(bbox_to_anchor=(1.0,0.16))
	plt.savefig("../result/plot_final/"+sheet.name+".eps",format="eps")

def falsepredict():
    import xlwt
    workbook = xlwt.Workbook(encoding='utf-8')
    for row in [2,4,6,8,10,20,40,100]:
        sheet = workbook.add_sheet(str(row)) #0
        sheet.write(0,0,"T_pro")
        sheet.write(0,1,"F_pro")
        #for row in [2,4,6,8,10,20,40,100]:
    
        pred = []
        file = open('../result/pred_final/10000/pred_label_'+str(row), 'r')
        for line in file:
            label = line.strip().split(" ")
                #folat_label = np.asarray(label,dtype="float")
            pred.append(label)
        pred_label = np.asarray(pred,dtype="float")
        tru = []
        file = open('../result/pred_final/10000/label_'+str(row), 'r')
        for line in file:
            label = line.strip()
            tru.append(label)
        tru_label = np.asarray(tru,dtype="float")
        #count = 0
        tr = 1
        fr = 1
        num = len(tru_label)
        for i in range(0,50000):
    	    pred = np.argmax(pred_label[i])
    	    tru = int(tru_label[i])
    	    if(tru==pred):
    	        sheet.write(tr,0,pred_label[i][pred])
		#if pred_label[i][pred]<0.999:
		    #count = count+1
		    #print "tpro->"+str(pred_label[i][pred])
    	        tr = tr+1
    	    else:
    	        sheet.write(fr,1,pred_label[i][pred])
		#if pred_label[i][pred]>0.999:
		    #print "fpro->"+str(pred_label[i][pred])
    	        fr = fr+1
	tr = 1
        fr = 1
        for i in range(50000,100000):
    	    pred = np.argmax(pred_label[i])
    	    tru = int(tru_label[i])
    	    if(tru==pred):
    	        sheet.write(tr,2,pred_label[i][pred])
    	        tr = tr+1
    	    else:
    	        sheet.write(fr,3,pred_label[i][pred])
    	        fr = fr+1
    workbook.save("pred.xlsx")
    
	

if __name__=="__main__":
    # generate pred_label_rows and label_rows
    for row in [2,4,6,8,10,20,40,100]:
        getpretable(rows=row)
    # pmodel8 = plotmodel(rows=8)
    # print(pmodel8.overallacc)
    # # for row in [10,20,40,100]:
    # #     getpretable(rows=row)
    # pmodel10 = plotmodel(rows=10)
    # pmodel20 = plotmodel(rows=20)
    # pmodel40 = plotmodel(rows=40)
    # pmodel100 = plotmodel(rows=100)
    # # drawpicture(pmodel10,pmodel20,pmodel40,pmodel100)
    # print(pmodel10.confusedmatrix())
    # print(pmodel10.precision())
    # print(pmodel10.precisionbycm())
    # print(pmodel10.recall())
    # print(pmodel10.recallbycm())
    # # getpretable(rows=10)
    # # getpretable(rows=20)
    # # getpretable(rows=40)
    # # getpretable(rows=100)

    models = []
    for r in [2,4,6,8,10,20,40,100]:
        models.append(plotmodel(rows=r))
    #---overallaccuary
    #saveresults(models)
    #---tfr tpr ...
    save2xlsx(models)
    #snsdraw()
    #for r in [2,4,6,8,10,20,40,100]:
    #    savepredict2xlsx(row=r)
    #falsepredict()


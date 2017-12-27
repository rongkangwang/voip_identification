from data_voip import load_data
import random
from keras.models import model_from_json
import numpy as np

class plotmodel:
    def __init__(self,rows=100):
        self.rows = rows
        model = model_from_json(
            open("../data/model_json/alexnet_model_architecture_" + str(rows) + ".json").read())
        model.load_weights("../data/model_json/alexnet_model_weights_" + str(rows) + ".h5")
        self.model = model
        self.data ,self.label = load_data(rows=rows)
        self.pred_label = model.predict(self.data,batch_size=1, verbose=1)

    def acc4voip(self):
        num = len(self.label)
        voip_len = [0 for i in range(7)]
        for i in range(num):
            if self.label[i] == np.argmax(self.pred_label[i]):
                index = self.label[i].astype(np.int64) - 1
                voip_len[index] += 1
        voip_acc = [voip_len[i] / (num / 7.0) for i in range(7)]
        print(voip_acc)

    def fprfnr(self):
        return

    def unknowpkt(self):
        return




if __name__=="__main__":
    pmodel = plotmodel(rows=100)
    pmodel.model.summary()
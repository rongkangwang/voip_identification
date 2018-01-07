from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from keras.models import Sequential
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Activation

img = Image.open("../data/cat/1.png")
cat = np.asarray(img)
# newcat = np.empty(cat.shape,dtype="float32")
# newcat[:,:,0] = cat[:,:,2]
# newcat[:,:,1] = cat[:,:,1]
# newcat[:,:,2] = cat[:,:,0]
# print(cat.shape)

model = Sequential()
model.add(Conv2D(1,(3,3),input_shape=cat.shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
cat_batch = np.expand_dims(cat,axis=0)
print(cat_batch.shape)
conv_cat = model.predict(cat_batch)
print(conv_cat.shape)
def v_cat(cat_batch):
    cat = np.squeeze(cat_batch,axis=0)
    print(cat.shape)
    cat = cat.reshape(cat.shape[:2])
    plt.imshow(cat)
    plt.show()

v_cat(conv_cat)


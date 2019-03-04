from keras.models import model_from_json
# import h5py
from PIL import Image
import numpy as np

def predict(image):
	img = Image.open(image)
	arr = np.asarray(img,dtype="float32")
	arr = arr.reshape(1,28,28,1)
	model = model_from_json(open("my_model_architecture.json").read())
	model.load_weights("my_model_weights.h5")
	print(model.predict_classes(arr))

if __name__=="__main__":
	predict("../data/mnist/1.184.jpg")
import numpy as np
# from PIL import Image
# from scipy import signal
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
#
# img = Image.open("./mnist/0.0.jpg")
# arr = np.asarray(img,dtype="float32")
#
# ker = np.random.random((4,5,5))
#
# a = [[1,2],[3,4]]
# b = [[-1,1],[-2,2]]
#
# #c = np.convolve([1,2,3,4],[1,1,3],mode="full")
# #c = signal.convolve2d(arr,ker,mode="valid")
#
# c = [[17,24,1,8,15],[23,5,7,14,16],[4,6,13,20,22],[10,12,19,21,3],[11,18,25,2,8]]
# d = [[8,1,6],[3,5,7],[4,9,2]]
#
# e = signal.convolve2d(c,d,mode="same")
# arr = arr.reshape( 28, 28, 1)
# arr = arr.reshape( 28, 28)
#
# print(np.shape(arr))
# plt.imshow(arr,cmap = cm.Greys_r)
# plt.show()

# import struct
#
# a = "\x04\x00"
# packet_len = struct.unpack('h', a)[0]
# print packet_len

# a = np.zeros((1,2,4),dtype="float32")
# b = np.zeros((1,2,4),dtype="float32")
# c = np.array([a,b])
# print c
#
# a = 111
# b = 111/4*3
# print b

a = "baidu__bre.pcap"
print a.split("_")

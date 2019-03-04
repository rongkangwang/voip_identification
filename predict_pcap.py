from keras.models import model_from_json
import h5py
from PIL import Image
import numpy as np
import struct
from paser_pcap import transferpacket2matrix

def predict(image):
	img = Image.open(image)
	arr = np.asarray(img,dtype="float32")
	arr = arr.reshape(1,28,28,1)
	model = model_from_json(open("my_model_architecture.json").read())
	model.load_weights("my_model_weights.h5")
	print(model.predict_classes(arr))


def getpackets(filename):
    fpcap = open(filename, 'rb')
    string_data = fpcap.read()
    # pcap header
    pcap_header = {}
    pcap_header['magic_number'] = string_data[0:4]
    pcap_header['version_major'] = string_data[4:6]
    pcap_header['version_minor'] = string_data[6:8]
    pcap_header['thiszone'] = string_data[8:12]
    pcap_header['sigfigs'] = string_data[12:16]
    pcap_header['snaplen'] = string_data[16:20]
    pcap_header['linktype'] = string_data[20:24]

    num = 0  # packet no.
    packets = []  # all of the packets
    pcap_packet_header = {}  # packet header
    i = 24  # pcap header takes 24 bytes
    while (i < len(string_data)):
        # paser packet header
        pcap_packet_header['GMTtime'] = string_data[i:i + 4]
        pcap_packet_header['MicroTime'] = string_data[i + 4:i + 8]
        pcap_packet_header['caplen'] = string_data[i + 8:i + 12]
        pcap_packet_header['len'] = string_data[i + 12:i + 16]
        # len is real, so use it
        packet_len = struct.unpack('I', pcap_packet_header['len'])[0]
        # add packet to packets
        packets.append(string_data[i + 16:i + 16 + packet_len])
        i = i + packet_len + 16
        num += 1
    fpcap.close()
    return packets

if __name__=="__main__":
    t = 0
    f = 0
    filename =  "/Users/kang/Documents/workspace/voip_identification/pcap/baidu_test.pcap"
    packets = getpackets(filename)
    model = model_from_json(open("my_model_architecture.json").read())
    model.load_weights("my_model_weights.h5")
    for p in packets:
        m = transferpacket2matrix(p)
        m = m.reshape(1, 256, 54, 1)
        result  = model.predict_classes(m)
        if(result == 0):
            t = t+1
        else:
            f = f+1
    print("True:"+str(t))
    print("False:"+str(f))
    print(t/(t+f))

from PIL import Image
import numpy as np
import struct
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def pcap2packets(filename):
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
    packets = []    # all of the packets
    pcap_packet_header = {}  # packet header
    i = 24    # pcap header takes 24 bytes
    max_len = 0
    while (i < len(string_data)):
        # paser packet header
        pcap_packet_header['GMTtime'] = string_data[i:i + 4]
        pcap_packet_header['MicroTime'] = string_data[i + 4:i + 8]
        pcap_packet_header['caplen'] = string_data[i + 8:i + 12]
        pcap_packet_header['len'] = string_data[i + 12:i + 16]
        # len is real, so use it
        packet_len = struct.unpack('I', pcap_packet_header['len'])[0]
        print(packet_len)
        if(packet_len>max_len):
            max_len = packet_len
        # add packet to packets
        packets.append(string_data[i + 16:i + 16 + packet_len])
        i = i + packet_len + 16
        num += 1
    fpcap.close()
    print("Max Length:"+str(max_len))
    return (packets,max_len)


def transferpacket2matrix(packet,max_len):
    p_len = len(packet)

    m = np.zeros(max_len,dtype="float32")
    for i in range(p_len):
        pc = packet[i]
        a_num = struct.unpack('B', pc)[0]
        m[i] = a_num
    return m

if __name__=="__main__":
    filename = "/Users/kang/Documents/workspace/voip_identification/ALT/alicall_voice.pcap"
    (packets,max_len) = pcap2packets(filename)
    m = np.zeros((len(packets), max_len), dtype="float32")
    for i in range(len(packets)):
        p = packets[i]
        m[i] = transferpacket2matrix(p,max_len)
    plt.imshow(m, cmap=cm.Greys_r)
    plt.show()
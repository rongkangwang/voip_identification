from PIL import Image
import numpy as np
import struct
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import platform

series_threshold = 2.0/3.0
series_pktnum = 15
column = 1000
row = 1040

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

def pcap2packetswithpktheader(filename):
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
        packets.append(string_data[i:i + 16 + packet_len])
        i = i + packet_len + 16
        num += 1
    fpcap.close()
    print("Max Length:"+str(max_len))
    return (packets,max_len)

def isudpwithpktheader(packet):   # isudp: the packet should with the header
    protocol = struct.unpack('B', packet[39])[0]
    if(protocol == 17):
        return True
    else:
        return False

def istcpwithpktheader(packet):
    protocol = struct.unpack('B', packet[39])[0]
    if(protocol == 6):
        return True
    else:
        return False

def ishttpwithpktheader(packet):
    if(istcpwithpktheader(packet)):
        flag = struct.unpack('B', packet[63])[0]
        if(hex(flag) == 0x018):
            port = struct.unpack('H', packet[53]+packet[52])[0]
            if(port==80):
                return True
            else:
                return False
        else:
            return False
    else:
        return False

def isudp(packet):
    # print protocol
    protocol = struct.unpack('B', packet[23])[0]
    if(protocol == 17):
        return True
    else:
        return False

def istcp(packet):
    protocol = struct.unpack('B', packet[23])[0]
    if(protocol == 6):
        return True
    else:
        return False

def ishttp(packet):    # ishttp: the packet should not with the header
    if(istcpwithpktheader(packet)):
        flag = struct.unpack('B', packet[47])[0]
        if(hex(flag) == 0x018):
            port = struct.unpack('H', packet[37]+packet[36])[0]
            if(port==80):
                return True
            else:
                return False
        else:
            return False
    else:
        return False

def getvoicestartandend(packets):
    start = 0
    end = 0
    for i,pkt in enumerate(packets):
        if(isudpwithpktheader(pkt)):
            if(isseries(packets[i:i+series_pktnum])):    # decide start position by next 20 packtes
                start = i
                break
    packets = packets[::-1]
    for i,pkt in enumerate(packets):
        if(isudpwithpktheader(pkt)):
            if(isseries(packets[i:i+series_pktnum])):    # decide start position by next 20 packtes
                end = i
                break
    end = len(packets) - end
    
    return (start,end-1)


def getvoiceend(packets):
    for i,pkt in enumerate(packets):
        if(not isudpwithpktheader(pkt)):
            if(len(packets)-1-i<series_pktnum):
                if(not isseries(packets[i:])): # decide end position by next 20 packets
                    end = i
                    break
            else:
                if(not isseries(packets[i:i+series_pktnum])): # decide end position by next 20 packets
                    end = i
                    break
    if(not end+series_pktnum > len(packets)):
        if(isseries(packets[end+series_pktnum:end+2*series_pktnum])):
            end = getvoiceend(packets[end+series_pktnum:]) + end + series_pktnum
    return end

def isseries(packets):      # check next 20 packets, if more than series_threshold, is udp series
    num = 0
    l = len(packets)
    for i,pkt in enumerate(packets):
        if(isudpwithpktheader(pkt)):
            num = num + 1
    print(num,l)
    if(float(num)/l>=series_threshold):
        return True
    else:
        return False

def getfinalmatrix(packets,start,end):   # not consider the end-start<1000 case
    # packets1 = packets[0:start-1]
    # packets2 = packets[start:end]
    # packets3 = packets[end:] 
    packets_end = packets[start-1-19:start-1]+packets[start:start+999]+packets[end+1:end+20]
    m = np.zeros((row, column), dtype="float32")
    if(start<20):
        for i in range(start+1):
            m[i+20-start] = transferpacket2matrix(packets[i])
    else:
        for i,pkt in enumerate(packets[start-1-19:start]):
            m[i] = transferpacket2matrix(pkt)
    if(len(packets)-end-1<20):
        for i in range(end+1,len(packets)):
            print i,end
            m[len(packets)-1-i+1000] = transferpacket2matrix(packets[i])
    else:
        for i,pkt in enumerate(packets[end+1:end+21]):
            m[end+20-i+1000] = transferpacket2matrix(pkt)
    for i,pkt in enumerate(packets[start:start+999]):
            m[i+20] = transferpacket2matrix(pkt)
    return m


def transferpacket2matrix(packet,max_len=1000):
    p_len = len(packet)

    m = np.zeros(max_len,dtype="float32")
    # for i in range(max_len):
    #     m[i] = 255
    if p_len>1000:
        p_len = 1000
    for i in range(p_len):
        pc = packet[i]
        a_num = struct.unpack('B', pc)[0]
        # m[i] = 255-a_num
        m[i] = a_num
    return m

if __name__=="__main__":
    if(platform.uname()[0]=="Linux"):
        filename = "/home/kang/Documents/data/alt/alt_voice.pcap"
    else:
        filename = "/Users/kang/Documents/workspace/data/alt/alt_voice.pcap"
    (packets_init,max_len) = pcap2packetswithpktheader(filename)
    (start,end) = getvoicestartandend(packets_init)
    print (start,end)
    m = getfinalmatrix(packets_init,start,end)
    #packets = getfinalpackets(packets_init,start,end)
    # for packet in packets:
    #     print isudp(packet)
    # m = np.zeros((len(packets), max_len), dtype="float32")
    # for i in range(len(packets)):
    #     p = packets[i]
    #     m[i] = transferpacket2matrix(p,max_len)
    plt.imshow(m, cmap=cm.Greys_r)
    plt.show()
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
#type = "zoiper"
types = ["xlite","skype","uu","zoiper","jumblo"]
types_loop = ["kc","alt","eyebeam","expresstalk","bria"]
pktnum = 224
rows_default = 120
cols_default = 256

def pcap2packetspayload(filename):
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
        #packet_len = struct.unpack('I', pcap_packet_header['len'])[0]
        packet_len = struct.unpack('I', pcap_packet_header['caplen'])[0]
        gtmtime = struct.unpack('=L', pcap_packet_header['GMTtime'])[0]
        microtime = struct.unpack('=L', pcap_packet_header['MicroTime'])[0]
        packet_time = float(gtmtime) + float(microtime) / 1000000.0
        #print(str(packet_len)+" "+str(packet_len1))
        # if(packet_len>1000):
        #     packet_len = packet_len1
        # if(packet_len>max_len):
        #     max_len = packet_len
        # add packet to packets
        packet = string_data[i + 16:i + 16 + packet_len]
        if(isudp(packet)):
            payload = packet[42:]
            packets.append((packet_time,payload))
        i = i + packet_len + 16
        num += 1
    fpcap.close()
    #print("Max Length:"+str(max_len))
    print(len(packets))
    return (packets,max_len)

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
        #packet_len = struct.unpack('I', pcap_packet_header['len'])[0]
        packet_len = struct.unpack('I', pcap_packet_header['caplen'])[0]
        #print(str(packet_len)+" "+str(packet_len1))
        # if(packet_len>1000):
        #     packet_len = packet_len1
        if(packet_len>max_len):
            max_len = packet_len
        # add packet to packets
        packets.append(string_data[i + 16:i + 16 + packet_len])
        i = i + packet_len + 16
        num += 1
    fpcap.close()
    #print("Max Length:"+str(max_len))
    print(len(packets))
    return (packets,max_len)

def skypepcap2imgpayload(filename):
    global imagecount
    fpcap = open(filename, 'rb')
    pcap_header = fpcap.read(24)
    packets = []
    i = 0
    packet_header = fpcap.read(16)
    while(len(packet_header)==16):
        # print(len(packet_header))
        # packet_len = struct.unpack('I', packet_header[12:16])[0]
        packet_len = struct.unpack('I', packet_header[8:12])[0]
        # print(packet_len)
        packet = fpcap.read(packet_len)
        if(isudp(packet)):
            packet = packet[42:]
            packets.append(packet)
            i = i+1
        if(i==224):
            get224imgsingle(packets)
            imagecount = imagecount+1
            packets = []
            i = 0
        packet_header = fpcap.read(16)

    fpcap.close()

def skypepcap2img(filename):
    global imagecount
    fpcap = open(filename, 'rb')
    pcap_header = fpcap.read(24)
    packets = []
    i = 0
    packet_header = fpcap.read(16)
    while(len(packet_header)==16):
        # print(len(packet_header))
        # packet_len = struct.unpack('I', packet_header[12:16])[0]
        packet_len = struct.unpack('I', packet_header[8:12])[0]
        # print(packet_len)
        packet = fpcap.read(packet_len)
        if(isudp(packet)):
            packets.append(packet)
            i = i+1
        if(i==224):
            get224imgsingle(packets)
            imagecount = imagecount+1
            packets = []
            i = 0
        packet_header = fpcap.read(16)

    fpcap.close()



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
    # print("Max Length:"+str(max_len))
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
    # print(num,l)
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
            # print i+20-start
            m[i+20-start] = transferpacket2matrix(packets[i])
    else:
        for i,pkt in enumerate(packets[start-1-19:start]):
            print i
            m[i] = transferpacket2matrix(pkt)
    for i,pkt in enumerate(packets[start:start+1000]):
        # print i
        m[i+20] = transferpacket2matrix(pkt)
    if(len(packets)-end-1<20):
        for i in range(end+1,len(packets)):
            m[i - end - 1 + 1000] = transferpacket2matrix(packets[i])
    else:
        for i,pkt in enumerate(packets[end+1:end+21]):
            m[i - end - 1 + 1000] = transferpacket2matrix(pkt)

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

def transferpacket2matrix224(packet,max_len=224):
    p_len = len(packet)

    m = np.zeros(max_len,dtype="float32")
    # for i in range(max_len):
    #     m[i] = 255
    if p_len>224:
        p_len = 224
    for i in range(p_len):
        pc = packet[i]
        a_num = struct.unpack('B', pc)[0]
        # m[i] = 255-a_num
        m[i] = a_num
    return m

def transferpacket2matrixbycols(packet,cols):
    p_len = len(packet)

    m = np.zeros(cols,dtype="float32")
    # for i in range(max_len):
    #     m[i] = 255
    if p_len>cols:
        p_len = cols
    for i in range(p_len):
        pc = packet[i]
        a_num = struct.unpack('B', pc)[0]
        # m[i] = 255-a_num
        m[i] = a_num
    return m

def get224imgwithpktheader(packets):
    pkts = []
    for packet in packets:
        if(isudpwithpktheader(packet)):
            pkts.append(packet)
    udplen = len(pkts)
    num = udplen/224
    print num
    for i in range(num):
        ps = pkts[0+i*224:223+i*224]
        m = np.zeros((224, 224), dtype="float32")
        for j in range(len(ps)):
            p = ps[j]
            m[j] = transferpacket2matrix224(p,224)
        plt.imshow(m, cmap=cm.Greys_r)
        plt.savefig("../data/"+str(pktnum)+"/"+type+"/"+type+str(i+1)+".png")
        #plt.show()
        #plt.close()

def get224imgsingle(packets):
    global imagecount
    m = np.zeros((224, 224), dtype="float32")
    for j in range(len(packets)):
        p = packets[j]
        m[j] = transferpacket2matrix224(p,224)
    im = np.array(m)
    im = im.reshape(224, 224)
    from scipy.misc import imsave
    # from matplotlib.pyplot import imsave
    imsave("../data/"+str(pktnum)+"/" + type + "/" + type + str(imagecount) + ".png", im)

def get224imgpayload(pkts):
    global imagecount
    udplen = len(pkts)
    num = udplen/pktnum
    print num
    for i in range(num):
        ps = pkts[0+i*pktnum:(pktnum-1)+i*pktnum]
        m = np.zeros((224, 224), dtype="float32")
        for j in range(len(ps)):
            p = ps[j]
            m[j] = transferpacket2matrix224(p,224)
        # plt.imshow(m, cmap=cm.Greys_r)
        # plt.savefig("../data/224/"+type+"/"+type+str(imagecount)+".png")
        im = np.array(m)
        im = im.reshape(224, 224)
        from scipy.misc import imsave
        # from matplotlib.pyplot import imsave
        imsave("../data/"+str(pktnum)+"/"+type+"/"+type+str(imagecount)+".png", im)
        imagecount = imagecount+1

def get224img(packets):
    global imagecount
    pkts = []
    for packet in packets:
        if(isudp(packet)):
            pkts.append(packet)
    udplen = len(pkts)
    num = udplen/224
    print num
    for i in range(num):
        ps = pkts[0+i*224:223+i*224]
        m = np.zeros((224, 224), dtype="float32")
        for j in range(len(ps)):
            p = ps[j]
            m[j] = transferpacket2matrix224(p,224)
        # plt.imshow(m, cmap=cm.Greys_r)
        # plt.savefig("../data/224/"+type+"/"+type+str(imagecount)+".png")
        im = np.array(m)
        im = im.reshape(224, 224)
        from scipy.misc import imsave
        # from matplotlib.pyplot import imsave
        imsave("../data/"+str(pktnum)+"/"+type+"/"+type+str(imagecount)+".png", im)
        imagecount = imagecount+1

def getimgbydims(pkts,rows=256,cols=256):
    global imagecount
    udplen = len(pkts)
    num = udplen/rows
    print num
    for i in range(num):
        if imagecount>30000:
            break
        ps = pkts[0+i*rows:(rows-1)+i*rows]
        m = np.zeros((rows, cols), dtype="float32")
        for j in range(len(ps)):
            p = ps[j]
            m[j] = transferpacket2matrixbycols(p,cols)
        # plt.imshow(m, cmap=cm.Greys_r)
        # plt.savefig("../data/224/"+type+"/"+type+str(imagecount)+".png")
        im = np.array(m)
        im = im.reshape(rows, cols)
        from scipy.misc import imsave
        # from matplotlib.pyplot import imsave
        basepath = "../data/"+str(rows)+"/"+type
        if(not os.path.exists(basepath)):
            os.makedirs(basepath)
        imsave(basepath+"/"+type+str(imagecount)+".png", im)
        imagecount = imagecount+1

imagecount = 1

def shufflepackets(packets):
    import random
    index = [i for i in range(len(packets))]
    random.shuffle(index)
    # packets = packets[index]
    pkts = []
    for i in index:
        pkts.append(packets[i])
    return pkts

import datetime

def saveitd2xlsx():
    import xlwt
    if (platform.uname()[0] == "Linux"):
        filepath = "/home/kang/Documents/data/"
    elif (platform.uname()[0] == "Darwin"):
        filepath = "/Users/kang/Documents/workspace/data/" 

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
    table = workbook.add_sheet('non-voip') #9
    tables.append(table)
    for sheet in tables:
        sheet.write(0,0,"length")
        if(sheet.name=="Skype"):
            filename = filepath + "/skype/skype1.pcap"
        if(sheet.name=="Jumblo"):
            filename = filepath + "/jumblo/jumblo1.pcap"
        if(sheet.name=="Xlite"):
            filename = filepath + "/xlite/xlite1.pcap"
        if(sheet.name=="Zoiper"):
            filename = filepath + "/zoiper/zoiper1.pcap"
        if(sheet.name=="UU"):
            filename = filepath + "/uu/uu1.pcap"
        if(sheet.name=="ALT"):
            filename = filepath + "/alt/alt_voice.pcap"
        if(sheet.name=="KC"):
            filename = filepath + "/kc/kc_voice.pcap"
        if(sheet.name=="Eyebeam"):
            filename = filepath + "/eyebeam/eyebeam1.pcap"
        if(sheet.name=="ExpressTalk"):
            filename = filepath + "/expresstalk/expresstalk1.pcap"
        if(sheet.name=="Bria"):
            filename = filepath + "/bria/bria1.pcap"
        if(sheet.name=="non-voip"):
            filename = filepath + "/non-voip/tencent.pcap"
        (packets_init, max_len) = pcap2packetspayload(filename)
        tr = 1
        for i in range(len(packets_init)-1):
            if tr>60000:
                break
            current_t = datetime.datetime.utcfromtimestamp(packets_init[i][0]).second+datetime.datetime.utcfromtimestamp(packets_init[i][0]).microsecond/1000000.0
            next_t = datetime.datetime.utcfromtimestamp(packets_init[i+1][0]).second+datetime.datetime.utcfromtimestamp(packets_init[i+1][0]).microsecond/1000000.0
            #print(next_t-current_t)
            sheet.write(tr,0,str(next_t-current_t))
            tr = tr+1
    workbook.save("itd.xlsx")

if __name__=="__main__":
    saveitd2xlsx()
                        

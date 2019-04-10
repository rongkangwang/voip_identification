import pcap
import pcap2image
import struct

def getsrcip(pkt):
    ip1 = struct.unpack('B', pkt[26])[0]
    ip2 = struct.unpack('B', pkt[27])[0]
    ip3 = struct.unpack('B', pkt[28])[0]
    ip4 = struct.unpack('B', pkt[29])[0]
    return "%d"%ip1+"."+"%d"%ip2+"."+"%d"%ip3+"."+"%d"%ip4

def getdstip(pkt):
    ip1 = struct.unpack('B', pkt[30])[0]
    ip2 = struct.unpack('B', pkt[31])[0]
    ip3 = struct.unpack('B', pkt[32])[0]
    ip4 = struct.unpack('B', pkt[33])[0]
    return "%d" % ip1 + "." + "%d" % ip2 + "." + "%d" % ip3 + "." + "%d" % ip4

def getsrcport(pkt):
    port = struct.unpack('H', pkt[35] + pkt[34])[0]
    return port

def getdstport(pkt):
    port = struct.unpack('H', pkt[37] + pkt[36])[0]
    return port

def isbigendian():
    a = 0x12345678
    result = struct.pack('i',a)
    if hex(ord(result[0])) == '0x78':
        print 'small endian'
    else:
        print 'big endian'

def get4tuple(pkt):
    return (getsrcip(pkt),getdstip(pkt),getsrcport(pkt),getdstport(pkt))

# class rtpheader:
#     def __init__(self):

def revertpkt2rtp(pkt):
    pkt = pkt[42:]
    version = pkt[0]
    port = struct.unpack('B', version)[0]
    #print bin(port)[2]
    print(int(str(bin(port)[2])+str(bin(port)[4]),2))


if __name__=="__main__":
    sniffer = pcap.pcap(name=None,promisc=True,immediate=True)
    for ts,pkt in sniffer:
        #print ts,`pkt`
        if pcap2image.isudp(pkt):
            revertpkt2rtp(pkt)


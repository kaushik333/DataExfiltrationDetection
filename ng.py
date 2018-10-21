from scapy.all import *
import numpy as np
import csv
import sys
import argparse

def main():

    descriptionText = '''Given a pcap file, generate unigrams from each packet
    '''
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description=descriptionText, formatter_class=argparse.RawDescriptionHelpFormatter)
    # mandatory arguments
    parser.add_argument("-f", "--infile", action='store', dest='pcapfile', type=str, required=True, help='Name of input PCAP file.')
    parser.add_argument("-o", "--outfile", action='store', dest='outfile', type=str, required=True, help='Name of output unigram CSV file.')
    parser.add_argument("-l", "--labelfile", action='store', dest='labelfile', type=str, required=True, help='Name of the label file.')
    # optional arguments
    parser.add_argument("--log", action='store', dest='logtype', type=str, required=False, default="e", help='Type of Logarithm to use. Valid choices are: None, e, 10, 2.')
    parser.add_argument("--csv-append", action='store', type=bool, dest='append', required=False, default=False, help='Append ngrams to an existing file. Default is "False"')
    args = parser.parse_args()

    # set the file mode to append or write based on args.append
    if(args.append == False):
        fileMode = "w"
    else:
        fileMode = "a"

    a = rdpcap(args.pcapfile)
    data_array = np.zeros([len(a),256])

    # Open the file to save ngrams to
    f = open(args.outfile, fileMode)
    writer = csv.writer(f)

    labels=[]
    with open(args.labelfile) as f:
        lab = csv.reader(f)
        for r in lab:
            labels.append(int(float(r[0])))

    for i in range(0,len(a)):
        if labels[i]==0:
            a[i]["DNSQR"].qname = a[i]["DNSQR"].qname[7:]
        c = str(a[i].payload.payload).encode("HEX")
        d = bytearray.fromhex(c)

        byte_list = []
        for k in d:
            byte_list.append(k)

        #############################################
        ### create 256 bins and add count occurences
        #############################################
        freq = []
        for j in range(0,256):
            num = byte_list.count(j)

            # Check if Logarithm is used.
            if(args.logtype != "None"):
                # Logarithm is used. Check which base.
                if num!=0:
                    if(args.logtype == "e"):
                        freq.append(1 + np.log(num))
                    elif(args.logtype == "10"):
                        freq.append(1 + np.log10(num))
                    elif(args.logtype == "2"):
                        freq.append(1 + np.log2(num))
                    else:
                        print("Invalid log type. Quit.")
                        sys.exit(1)
                else:
                    freq.append(0)

            # No logarithm is used. frequency is base 10
            else:
                freq.append(num)

        #data_array[i] = np.array(freq)
        ########################################
        ### SAVE N-GRAMS IN CSV FILE
        ########################################
        writer.writerow(freq)

    f.close()

if __name__ == "__main__":
    main()

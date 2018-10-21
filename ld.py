from scapy.all import *
import scapy
import numpy as np
import csv
import sys
import argparse


def main():
    
    #a = rdpcap('twitter_exfiltration.pcap')
    #print("Done loading")
    
    descriptionText = '''Given a pcap file, filter out the DNS packets and label each of them as Malicious or Non-malicious. 
                         The output is going to be a csv file with 1s and 0s; 1 being non-malicious and 0 being malicious
                      '''
    
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description=descriptionText, formatter_class=argparse.RawDescriptionHelpFormatter)
    
    ############################
    ### MANDATORY ARGUMENTS
    ############################
    parser.add_argument("-f", "--infile", action='store', dest='pcapfile', type=str, required=True, help='Name of input PCAP file.')
    parser.add_argument("-fn", "--output", action='store', dest='newpcapfile', type=str, required=True, help='Name of filtered PCAP file.')
    parser.add_argument("-l", "--logfile", action='store', dest='logfile', type=str, required=True, help='Path to the logfile which helps in labelling.')
    parser.add_argument("-o","--labelfile",action='store',dest='labelfile',type=str, required=True, help='Path to output-label-file.')
    args = parser.parse_args()
    dns_list = []
    
    ##############################
    ### FIREWALL
    ##############################
    myreader = PcapReader(args.pcapfile)
    while True:
        a = myreader.read_packet()
        if a is None:
            break
        if (a.haslayer("DNS")):
            try:
                if (a["DNSQR"].qtype != 28): #not AAAA type resource record
                    dns_list.append(a)
            except:
                pass
    
    del a        
    #STORE IN A SEPARATE PCAP FILE        
    wrpcap(args.newpcapfile,dns_list)
            
    time_stamp = [x.time for x in dns_list]
    
    ##############################
    ### GET LOGS OF VISIT
    ##############################
    
    with open(args.logfile) as f:
        websites = csv.reader(f)
        data = [r for r in websites] 
    
    ############################################
    ### GET LABELS AND TME STAMPS FROM LOG FILE
    ############################################
    
    time_list=[]
    labels = []
    for i in range(0,len(data)):
        time_list.append(float(data[i][2]))
        labels.append(data[i][0])
        
    f = open(args.labelfile, 'a')
    writer = csv.writer(f)
        
    #################################
    ### PACKET WISE LABELLING
    #################################
    for i in range(1,len(time_list)):
        start_time = time_list[i-1]
        end_time = time_list[i]
        for pack_num in range(0,len(dns_list)):            
            if (i-1)==0:
                if dns_list[pack_num].time < start_time:
                    writer.writerow([1]) #Normal data
            
            if i==len(time_list)-1:
                if dns_list[pack_num].time > end_time:
                    if labels[i]=="Exfiltraion":
                        writer.writerow([0]) #Exfiltration data
                    else:
                        writer.writerow([1]) #Normal data
                
    #        if dns_list[pack_num].time > end_time or dns_list[pack_num].time < start_time:
    #            continue
            
            if dns_list[pack_num].time >= start_time and dns_list[pack_num].time < end_time:
                if labels[i-1]=="Exfiltraion":
                    writer.writerow([0]) #Exfiltration data
                else:
                    writer.writerow([1]) #Normal data
                    
    f.close()            
        
    
if __name__ == "__main__":
    main()    

    

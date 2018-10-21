#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 10:30:39 2018

@author: Kaushik Koneripalli
"""

import os, random
import numpy as np
import csv
import time
import datetime
from dnslib import *
import sys
import argparse

#####################################################################
### Define the websites to generate normal traffic
#####################################################################
def main():
    descriptionText = ''' This file is used to generate normal DNS traffic along with some exfiltrations. 
        '''
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description=descriptionText, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-w", "--weblist", action='store', dest='weblist', type=str, required=True, help='Name of csv file with website list. Make sure its in the current directory.')
    parser.add_argument("-l", "--logfile", action='store', dest='logfile', type=str, required=True, help='Name of csv file which will store the logs of normal traffic and exfiltration.')
    parser.add_argument("-hr", "--hours", action='store', dest='hours', type=float, required=True, help='Number of hours of traffic generation.')
    
    args = parser.parse_args()
    web_path = "./"+args.weblist
    with open(web_path) as f:
        websites = csv.reader(f)
        data = [r for r in websites] 
        
    web_list = [item for sublist in data for item in sublist]
     
    start = time.time()
    num_hours = args.hours
    
    rand_num1 = round(num_hours*np.random.random(),2)
    rand_num2 = round(num_hours*np.random.random(),2)
    rand_num3 = round(num_hours*np.random.random(),2)
    print rand_num1
    print rand_num2
    print rand_num3
    
    log_file = "./"+args.logfile
    f = open(args.logfile, "a")
    writer = csv.writer(f)
    
    while(time.time() - start < 60*60*num_hours):
        #f = open("traffic_log.txt","a")
        
        condition1 =  (time.time() - start > 60*60*rand_num1) and (time.time() - start < 60*60*(rand_num1+0.005))
        condition2 =  (time.time() - start > 60*60*rand_num2) and (time.time() - start < 60*60*(rand_num2+0.005))
        #condition3 =  (time.time() - start > 60*60*rand_num3) and (time.time() - start < 60*60*(rand_num3+0.005))
        condition3 = False
        rand_num = np.random.random_sample()
        if (condition1 or condition2 or condition3):
            print("EXFILTRATION !!!")
            #####################################################################
            ### Define the data files to be exfiltrated
            #####################################################################
            file_name = random.choice(os.listdir("./ExfiltrationData/")) #change dir name to whatever   
            file_path = './ExfiltrationData/'+file_name
            
            ###########################
            ### exfiltrate data
            ##########################
            #st = datetime.datetime.fromtimestamp(time.time())#.strftime('%Y-%m-%d %H:%M:%S')
            st = time.time()
            writer.writerow(['Exfiltraion',file_name,str(st)])
            st = time.time()
            sudoPassword = 'labuser'
            command = 'sudo python det.py -f '+file_path+' -c ./config-client.json -p dns'
            os.system('echo %s|sudo -S %s' % (sudoPassword, command))
            #os.system('sudo python det.py -f {} -c ./config-client.json -p http'.format(file_path))
        else:
            ###########################
            ### generate normal traffic
            ########################## 
            RR_list=["MX","NS","A","TXT","CNAME"]
            rand_num = np.random.random()
            url = random.choice(web_list)
            print("Visiting {}".format(url))
            st = time.time()
            writer.writerow(['Normal traffic',url,str(st)])
            if rand_num < 0.4:
                os.system('wget {}'.format(url))
            else:
                d = DNSRecord.question(url, random.choice(RR_list))
    #        os.system('wget {}'.format(d))
                try:
                    d.send("192.168.220.141", 53, timeout=0.01)
                except:
                    pass
                time.sleep(0.5)

    f.close()

if __name__ == "__main__":
    main()

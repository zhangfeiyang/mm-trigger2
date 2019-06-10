#!/usr/bin/python3

from ROOT import TTree,TFile,TChain

import numpy as np
import random
from random import shuffle
import math

id_dic = {}
pi = 3.141592654

N = 15

def gen_id_dic():

    file0 = open('Map','r')

    for line in file0:
        datas = line.split()
        id_dic[int(datas[0])] = [float(datas[1]),float(datas[2]),float(datas[3])]

def id_2_theta_phi(Id):

    x = id_dic[Id][0]
    y = id_dic[Id][1]
    z = id_dic[Id][2]

    r = (x**2 + y**2 + z**2)**0.5
    costheta = z/r

    if y > 0:
        phi = math.acos(x/(x**2 + y**2)**0.5)
    else:
        phi = 2*pi - math.acos(x/(x**2 + y**2)**0.5)
    
    return (costheta,phi)
    
def gen_data(tag):

    t = TChain("t")
    t.Add("ML.root")
    
    datas = []
    labels = []
    entries = t.GetEntries()
    #entries = 500
    if tag == 0:
        Range = range(0,int(entries/2),1)
    else:
        Range = range(int(entries/2),entries,1)
    
    for i in Range:

        t.GetEntry(i)
        nhits = t.GetLeaf("nhits").GetValue(0)
        data = []
        nhits_C14 = int((t.GetLeaf("nhits_C14").GetValue(0)))

        if nhits_C14 == 0:
            seed = random.uniform(0,600)
            #seed = random.uniform(0,1)
            if seed > 1:
                continue
            labels.append([0])

        elif nhits_C14 > 15 and nhits_C14 < 25:
        #elif nhits_C14 > 20:
            labels.append([1])
        else:
            continue
    
        for k in range(0,N*N):
            data.append(0)

        for j in range(int(nhits)):
            pmtID = int((t.GetLeaf("pmtID").GetValue(j)))
            if pmtID > 20000:
                continue
            (costheta,phi) = id_2_theta_phi(pmtID)

            data[int((costheta + 1)/2.0*N)*N + int(phi/2.0/pi*N)] += 1
    
        datas.append(data)
    
    return (np.array(datas,'i'),np.array(labels,'i'))

if __name__ == "__main__":
    
    gen_id_dic()

    file = open('Data5_'+str(N),'w')
    datas = gen_data(0)
    for i in range(len(datas[0])):
        for j in range(N*N):
            file.write(str(datas[0][i][j])+'\t')
        file.write(str(datas[1][i][0])+'\n')

    datas = gen_data(1)
    for i in range(len(datas[0])):
        for j in range(N*N):
            file.write(str(datas[0][i][j])+'\t')
        file.write(str(datas[1][i][0])+'\n')


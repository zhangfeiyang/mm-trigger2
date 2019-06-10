#!/usr/bin/env python

from ROOT import TTree,TFile,TChain

import numpy as np
import random
from random import shuffle

def gen_data(tag):

    t = TChain("t")
    t.Add("ML.root")
    
    datas = []
    labels = []
    entries = t.GetEntries()
    #entries = 500
    if tag == 0:
        Range = range(0,int(entries/2),10)
    else:
        Range = range(int(entries/2),entries,10)
    
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
        #elif nhits_C14 > 50:
            labels.append([1])
        else:
            continue
    
        for k in range(0,17739):
            data.append(0)

        for j in range(int(nhits)):
            pmtID = int((t.GetLeaf("pmtID").GetValue(j)))
            if pmtID > 20000:
                continue
            data[pmtID] = 1 
    
        datas.append(data)
    
    return (np.array(datas,'i'),np.array(labels,'i'))

if __name__ == "__main__":
    datas = gen_data()
    file = open('Data1','w')
    for i in range(len(datas[0])):
        for j in range(17739):
            file.write(str(datas[0][i][j])+'\t')
        file.write('\n'+str(datas[1][i][0])+'\n')
    print(len(datas[0]))
    print("hello")

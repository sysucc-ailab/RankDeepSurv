# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
import h5py


def getDataset(textpath,factors,sheetname):
    DF = pd.read_excel(textpath,sheet_name=sheetname,)
    numrow = DF.iloc[:, 0].size  #2341
    index = range(0, numrow)
    random.shuffle(index)
    tindex = index[0:numrow]
    df = DF[factors] #u'首次治疗时间'
    data = np.array(df)

    # proprecessing
    ind = np.int32(data[:, 10])
    text = np.float32(data[:, 0:8])
    #text[:, 8] = np.log10(text[:, 8] + 1)  # EBVDNAs
    osM = np.float32(data[:, 9])


    dataset={}

    dataset['x']=text[tindex,:]
    dataset['e']=ind[tindex]
    dataset['t'] = osM[tindex]

    return dataset['x'],dataset['e'],dataset['t']





if __name__ == '__main__':
    textpath = '../excel file/npcData/npc_last_all_2ci.xlsx'

    factors = [u'gender', u'TUICC', u'NUICC', u'CRP', u'LDH',u'age', u'HGB',  u'BMI',u'EBVDNA','PFSmonths','outcome']
    #factors = [u'sex', u'TUICC', u'NUICC', u'CRP', u'LDH',u'age', u'HGB',  u'BMI','PFSmonths',u'outcome']  #cphA
    #factors = [u'sex', u'TUICC', u'NUICC', u'CRP', u'LDH',u'age', u'HGB',  u'BMI',u'EBVDNA','PFSmonths',u'outcome']  #cphB
    #, u'VCAIgA1', u'EAIgA1', u'ALB', u'GLOB', u'WBC', u'NE2', u'LY2',  u'PLT',
    #u'身高平方'
    text,e,t= getDataset(textpath=textpath,factors=factors,sheetname='train')
    tstext, tse, tst = getDataset(textpath=textpath,factors=factors,sheetname='test')

    file = h5py.File('../data/npcData/pfs_npc_A_ranksurv.h5', 'w')
    file.create_group('train')
    file['train'].create_dataset('x',data=text)
    file['train'].create_dataset('t',data=t)
    file['train'].create_dataset('e',data=e)
    file.create_group('test')
    file['test'].create_dataset('x',data=tstext)
    file['test'].create_dataset('t',data=tst)
    file['test'].create_dataset('e',data=tse)
    file.close()

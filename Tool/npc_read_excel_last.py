# -*- coding: utf-8 -*-

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image as pil_image
import pandas as pd
import random
import time
import os
from datetime import datetime
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
    textpath = '/media/sysucc99/data/ZTROOT/2017MainText/excel file/NPC/npc_last_all.xlsx'

    factors = [u'性别', u'T分期UICC七', u'N分期UICC七', u'CRP', u'LDH',u'年龄', u'HGB',  u'BMI',u'EBVDNA','OSmonths',u'患者死亡状态']
    #, u'VCAIgA1', u'EAIgA1', u'ALB', u'GLOB', u'WBC', u'NE2', u'LY2',  u'PLT',
    #u'身高平方'
    text,e,t= getDataset(textpath=textpath,factors=factors,sheetname='raw_train')
    tstext, tse, tst = getDataset(textpath=textpath,factors=factors,sheetname='raw_test')

    file = h5py.File('/media/sysucc99/data/ZTROOT/2017MainText/data/NPC/npc_A_excel_last.h5', 'w')
    file.create_group('train')
    file['train'].create_dataset('x',data=text)
    file['train'].create_dataset('t',data=t)
    file['train'].create_dataset('e',data=e)
    file.create_group('test')
    file['test'].create_dataset('x',data=tstext)
    file['test'].create_dataset('t',data=tst)
    file['test'].create_dataset('e',data=tse)
    file.close()

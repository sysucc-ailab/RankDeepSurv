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



def getDataset(textpath):
    DF = pd.read_excel(textpath,sheet_name='Sheet1')
    numrow = DF.iloc[:, 0].size
    index = range(0, numrow)
    random.shuffle(index)

    trainindex = index[0:4630]
    testindex = index[4630:-1]

    df = DF[[u'性别', u'T分期UICC七', u'N分期UICC七', u'CRP', u'LDH',u'年龄', u'HGB',  u'BMI',u'EBVDNA','OSmonths',u'患者死亡状态']]
    #, u'VCAIgA1', u'EAIgA1', u'ALB', u'GLOB', u'WBC', u'NE2', u'LY2',  u'PLT',
    #u'身高平方'

    data = np.array(df,dtype='float32')
    data[:, 8] = np.log10(data[:, 8] + 1)


    data_Train = {}
    data_Test = {}
    data_Train['x'] = np.float32(data[trainindex, 0:9])
    data_Train['t'] = np.float32(data[trainindex, 9])
    data_Train['e'] = np.int32(data[trainindex, 10])
    data_Test['x'] = np.float32(data[testindex, 0:9])
    data_Test['t'] = np.float32(data[testindex, 9])
    data_Test['e'] = np.int32(data[testindex, 10])

    return data_Train['x'],data_Train['e'],data_Train['t'], data_Test['x'], data_Test['e'], data_Test['t']



if __name__ == '__main__':
    textpath = '/media/sysucc99/data/ZTROOT/2017MainText/excel file/NPC/train_test.xlsx'
    text,e,t,tstext,tse,tst = getDataset(textpath=textpath)



    file = h5py.File('/media/sysucc99/data/ZTROOT/2017MainText/data/npc_B_excel.h5', 'w')
    file.create_group('train')
    file['train'].create_dataset('x',data=text)
    file['train'].create_dataset('t',data=t)
    file['train'].create_dataset('e',data=e)
    file.create_group('test')
    file['test'].create_dataset('x',data=tstext)
    file['test'].create_dataset('t',data=tst)
    file['test'].create_dataset('e',data=tse)
    file.close()
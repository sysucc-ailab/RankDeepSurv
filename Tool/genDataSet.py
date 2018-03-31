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
    DF = pd.read_csv(textpath)
    numrow = DF.iloc[:, 0].size  #2341
    index = range(0, numrow)
    random.shuffle(index)
    tindex = index[0:numrow]
    df = DF[factors] #u'首次治疗时间'
    data = np.array(df)

    # proprecessing
    ind = np.int32(data[:, 10])
    text = np.float32(data[:, 0:9])
    text[:, 8] = np.log10(text[:, 8] + 1)  # EBVDNAs
    osM = np.float32(data[:, 9])


    dataset={}

    dataset['x']=text[tindex,:]
    dataset['e']=ind[tindex]
    dataset['t'] = osM[tindex]

    return dataset['x'],dataset['e'],dataset['t']


class GeneratorImage():
    def __init__(self,dataset, width, height):
        self.width = width
        self.height = height
        self.generator = ImageDataGenerator(
            rescale=1. / 255,
            samplewise_center=True,
            samplewise_std_normalization=True,
            rotation_range=60,
            horizontal_flip=True,
            vertical_flip=True,
        )
        imagepath = dataset['image']
        text = dataset['text']
        e = dataset['e']
        t = dataset['t']
        self.imagep = []
        mm = -1
        for i in range(0,len(imagepath)):
            for j,path in enumerate(imagepath[i]):
                mm=mm+1
                self.imagep.append((path))
                if mm==0:
                    self.text=np.reshape(text[i,:],(1,4))
                    self.t = np.reshape(t[i],(1))
                    self.e = np.reshape(e[i],(1))
                else:
                    text0=np.reshape(text[i,:],(1,4))
                    t0 = np.reshape(t[i],(1))
                    e0 = np.reshape(e[i],(1))
                    self.text=np.concatenate((self.text,text0),axis=0)
                    self.t=np.concatenate((self.t,t0),axis=0)
                    self.e=np.concatenate((self.e,e0),axis=0)


    def _load_img(self, path, target_size, grayscale=False, Flag=True):

        if pil_image is None:
            raise ImportError('Could not import PIL.Image. '
                              'The use of `array_to_img` requires PIL.')
        img = pil_image.open(path)

        if grayscale:
            if img.mode != 'L':
                img = img.convert('L')
        else:
            if img.mode != 'RGB':
                img = img.convert('RGB')
        """
        if target_size:
            hw_tuple = (target_size[1], target_size[0])
            if img.size != hw_tuple:
                print img.size
                img = img.resize(hw_tuple)
        """
        h, w = img.size
        if w > h:
            center = w / 2
            bilen = h / 2
            box = (center - bilen + 80, 0 + 80, center + h - bilen - 80, h - 80)
            img = img.crop(box)
        elif w < h:
            center = h / 2
            bilen = w / 2
            box = (0 + 80, center - bilen + 80, w - 80, center + w - bilen - 80)
            img = img.crop(box)

        newimg = img.resize(target_size)
        return newimg

    def _img_to_array(self, img, data_format='channels_last'):
        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Unknown data_format: ', data_format)

        x = np.asarray(img, dtype=np.float32)
        if len(x.shape) == 3:
            if data_format == 'channels_first':
                x = x.transpose(2, 0, 1)
        elif len(x.shape) == 2:
            if data_format == 'channels_first':
                x = x.reshape((1, x.shape[0], x.shape[1]))
            else:
                x = x.reshape((x.shape[0], x.shape[1], 1))
        else:
            raise ValueError('Unsupported image shape: ', x.shape)
        # skimage.util.random_noise(x,mode = 'gaussian',clip=True)

        return x


    def loaddata(self):
        allnum = len(self.imagep)
        c_img = np.zeros([allnum, self.width, self.height, 3], dtype=np.float32)

        for i in range(0,allnum):
            data = self.imagep[i]
            #print 'loaddata',data
            img = self._load_img(path=data, target_size=(self.width, self.height))
            img = self._img_to_array(img=img)
            img = self.generator.random_transform(img)
            img = self.generator.standardize(img)
            c_img[i,:,:,:]= img
        print c_img.shape
        print self.text.shape
        print self.t.shape
        print self.e.shape

        t_p = np.zeros((allnum),dtype=np.float32)
        print t_p.shape

        return c_img,self.text,self.t,self.e,t_p

if __name__ == '__main__':
    trainpath = '/media/sysucc99/data/ZTROOT/2017MainText/excel file/NPC/train.csv'
    testpath = '/media/sysucc99/data/ZTROOT/2017MainText/excel file/NPC/test.csv'
    factors = ['sex', 'TUICC', 'NUICC', 'CRP', 'LDH', 'age', 'HGB', 'BMI', 'EBVDNA','PFSmonths','outcome']
    text,e,t= getDataset(textpath=trainpath,factors=factors,sheetname='train')
    tstext, tse, tst = getDataset(textpath=testpath,factors=factors,sheetname='test')

    file = h5py.File('/media/sysucc99/data/ZTROOT/2017MainText/data/npc_B.h5', 'w')
    file.create_group('train')
    file['train'].create_dataset('x',data=text)
    file['train'].create_dataset('t',data=t)
    file['train'].create_dataset('e',data=e)
    file.create_group('test')
    file['test'].create_dataset('x',data=tstext)
    file['test'].create_dataset('t',data=tst)
    file['test'].create_dataset('e',data=tse)
    file.close()
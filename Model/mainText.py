# -*- coding: utf-8 -*-

import numpy as np
from keras import initializers
from keras import  Model
from keras.callbacks import Callback,TensorBoard
from keras import backend as K
from keras.layers import  Input,BatchNormalization,Dense,Dropout,regularizers,Lambda,Concatenate,Activation
from keras.activations import relu
from keras.optimizers import SGD,Adam
from keras.losses import hinge
import random
import tensorflow as tf
import os
import copy
import h5py
import json
import scipy.stats as st
from lifelines.utils import concordance_index as cindex
from collections import defaultdict
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))



def standardize_dataset(dataset, offset, scale):
    norm_ds = copy.deepcopy(dataset)
    norm_ds['x'] = (norm_ds['x'] - offset) / scale
    return norm_ds

def load_datasets(dataset_file,standardize=True):
    datasets = defaultdict(dict)

    with h5py.File(dataset_file, 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]

    if standardize:
        norm_vals = {
            'mean': datasets['train']['x'].mean(axis=0),
            'std': datasets['train']['x'].std(axis=0)
        }
        datasets['train'] = standardize_dataset(datasets['train'], norm_vals['mean'], norm_vals['std'])
        datasets['test'] = standardize_dataset(datasets['test'], norm_vals['mean'], norm_vals['std'])

    return datasets

def _sort(dataset):
    e = dataset['e']
    t = dataset['t']
    sortindex = np.argsort(t)  # /?\#
    ind = e
    uncensorindex = np.where(ind == 1)
    uncensorindex = uncensorindex[0]
    uncensorsortindex = np.argsort(t[uncensorindex])
    uncensorsortindex = uncensorindex[uncensorsortindex]
    return sortindex, uncensorsortindex,t

def _pairwise(self,index,uncensorindex):
    k = 1
    flag = True
    index = index.tolist()
    for i in index:
        osZ = self.t[i]
        j = uncensorindex[0]
        if osZ < self.t[j]:
            del index[i]
            continue
        k = k - 1
        for j, mmm in enumerate(uncensorindex):
            osC = self.t[mmm]
            if osC > osZ:
                k = j - 1
                if k == 0:
                    k = 1
                break
        try:
            pj = random.randint(0, j - 1)
        except:
            pj = 0
            print 'except pairwise:(index,uncensorindex)', i, j
        nnn = uncensorindex[pj]
        pair = np.array([i, nnn], dtype=np.int32)
        pair = pair.reshape((1, 2))
        if flag:
            flag = False
            pairwise = pair
            pairwise = pairwise.reshape((1, 2))
        else:
            pairwise = np.concatenate((pairwise, pair), axis=0)

    pair_index = range(0, pairwise.shape[0])
    random.Random(0).shuffle(pair_index)
    newpair = pairwise[pair_index, :]
    self.c_index = pairwise[pair_index, 0]
    return newpair
def _pairwise_all(dataset):
    index, uncensorindex, t = _sort(dataset)
    flag = True
    index = index.tolist()
    pair_in = list()
    pair_un = list()
    for i in index:
        osZ = t[i]
        j = uncensorindex[0]
        if osZ <= t[j]:
            del index[i]
            continue
        for j, mmm in enumerate(uncensorindex):
            osC = t[mmm]
            if osC >= osZ:
                break
        # if j>=20:
        #     qj = j/4
        # else:
        #     qj=j
        # print qj
        for pj in range(0,j):
            nnn = uncensorindex[pj]
            pair_un.append(nnn)
            pair_in.append(i)
    pairwise = np.array(zip(pair_in, pair_un))
    pair_index = range(0, len(pair_in))
    seed = random.randint(0,1e+6)
    random.Random(4096).shuffle(pair_index)
    print seed

    newpair = pairwise[pair_index, :]
    #c_index = pairwise[pair_index, 0]
    return newpair
class Generator():
    def __init__(self,dataset,batchsize):
        self.batchsize = batchsize
        #self.image = dataset['image']
        self.text = dataset['x']
        self.e = dataset['e']
        self.t = dataset['t']
        #self.sortindex,self.uncensorsortindex = self._sort()




    def loaddata(self,batchnum,pairwise):
        while True:
            #pairwise = self._pairwise_all(self.sortindex, self.uncensorsortindex)
            c = pairwise[:, 0]
            u = pairwise[:, 1]
            for i in range(0, batchnum):
                index = c[i * self.batchsize:(i + 1) * self.batchsize]
                uindex = u[i * self.batchsize:(i + 1) * self.batchsize]
                #c_img = self.image[index, :, :, :]
                c_text = self.text[index, :]
                c_t = self.t[index]
                #c_t[c_t>=60]=60
                c_e = self.e[index]
                #u_img = self.image[uindex, :, :, :]
                u_text = self.text[uindex, :]
                u_t = self.t[uindex]
                #u_t[u_t>=60]=60
                u_e = self.e[uindex]

                input = {
                    #'c_img': c_img,
                    'c_text': c_text,
                    'c_t': c_t,
                    'c_e': c_e,
                    #'u_img': u_img,
                    'u_text': u_text,
                    'u_t': u_t,
                    'u_e': u_e,

                }
                output = {
                    'rankmseloss': np.zeros((self.batchsize), dtype=np.float32)
                }
                yield (input, output)

def mseloss(mm):
    #y_true, y_pred, ind, y_true_uncensor, y_pred_uncensor, ind_uncensor = mm
    #pred = y_pred - y_pred_uncensor
    #true = y_true - y_true_uncensor
    #rankloss = K.tensor_array_ops.array_ops.where(K.greater(true, pred), K.square(true - pred), pred * 0)
    y_true, y_pred, ind, y_true_uncensor, y_pred_uncensor, ind_uncensor = mm

    fuind = (ind - 1) * (-1)
    mseloss = K.square(y_true - y_pred) * ind + fuind * K.tensor_array_ops.array_ops.where(
        K.greater(y_true, y_pred), K.square(y_true - y_pred), y_true * 0)  # 1 ->uncensor, 0->censor,y+pred<=max?
    return mseloss


def rankmseloss(mm):
    y_true, y_pred, ind, y_true_uncensor, y_pred_uncensor, ind_uncensor = mm

    fuind = (ind - 1) * (-1)
    mseloss0 = K.square(y_true - y_pred) * ind + fuind * K.tensor_array_ops.array_ops.where(
        K.greater(y_true, y_pred), K.square(y_true - y_pred), y_true * 0)  # 1 ->uncensor, 0->censor,y+pred<=max?
    pred = y_pred - y_pred_uncensor
    true = y_true - y_true_uncensor
    rankloss = K.tensor_array_ops.array_ops.where(K.greater(true, pred), K.square(true - pred), pred * 0)
    return  mseloss0*11.5+rankloss*11.2



def lossfunction(y_true,y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)

def myact(x):
    return K.relu(x, alpha=0, max_value=60)
class MyModel():
    def __init__(self,loss,batchsize,batchnorm):
        self.batchsize = batchsize
        self.rankmseloss = loss
        self.batchnorm = batchnorm


    def fcn(self,batch,n_in, hidden_layers_sizes, activation, dropout, l2):
        #seed = 521 # GBSG,416304
        seed = random.randint(0,1e+6)
        #seed = 813314
        for i in range(3):
            print '---------------------------------\n',seed
        textinput = Input(batch_shape=(batch, n_in))

        inner = Dense(n_in, activation=activation, name='dense%d' % (1), kernel_initializer=initializers.glorot_normal(seed=seed))(
            textinput)
        for i, j in enumerate(hidden_layers_sizes):
            inner = Dense(j, activation=activation, name='dense%d' % (2+ i), kernel_initializer=initializers.glorot_normal(seed=seed))(
                inner)

        inner = Dropout(dropout, name='dropout')(inner)
        y_pred = Dense(1,activation=activation,name='pred', kernel_regularizer=regularizers.l2(l2),kernel_initializer=initializers.glorot_normal(seed=seed))(inner)
        fcn = Model(inputs=[textinput], outputs=[y_pred])
        #fcn.summary()

        return fcn

    def model(self,n_in,l2 = 0.0,activation = "elu",dropout = None, hidden_layers_sizes = None,):
        with tf.device('/cpu:0'):
            # print self.batchsize, n_in
            textinput = Input(batch_shape=(self.batchsize, n_in))
            fcn_model = self.fcn(self.batchsize,n_in,hidden_layers_sizes,activation,dropout,l2=l2)
            y_pred = fcn_model([textinput])
            predictmodel = Model(inputs=[ textinput], outputs=[y_pred])
            #predictmodel.summary()

        with tf.device('/cpu:0'):
            textdata = Input(name='c_text', batch_shape=[self.batchsize, n_in], dtype='float32')
            c_y_pred = fcn_model([textdata])
        with tf.device('/cpu:0'):
            textdata_uncensor = Input(name='u_text', batch_shape=[self.batchsize, n_in], dtype='float32')
            u_y_pred = fcn_model([textdata_uncensor])

        with tf.device('/cpu:0'):
            ind = Input(name='c_e', batch_shape=[self.batchsize,1], dtype='float32')
            y_true = Input(name='c_t', batch_shape=[self.batchsize,1], dtype='float32')
            ind_uncensor = Input(name='u_e', batch_shape=[self.batchsize,1], dtype='float32')
            y_true_uncensor = Input(name='u_t', batch_shape=[self.batchsize,1], dtype='float32')
            rankmse = Lambda(self.rankmseloss,output_shape=(1,),name='rankmseloss')([y_true, c_y_pred, ind,y_true_uncensor,u_y_pred,ind_uncensor])
        model = Model(inputs=[textdata, y_true, ind,textdata_uncensor,y_true_uncensor,ind_uncensor], outputs=[rankmse])
        # model.summary()
        return predictmodel, model

class VizCallBack(Callback):
    def __init__(self,trainset,testset,model,batch_size,weightname):
        self.trainset = trainset
        self.testset = testset
        self.predictmodel = model
        self.weight = weightname
        self.batchsize = batch_size
        ii = np.shape(trainset['t'])[0]
        testii = np.shape(testset['t'])[0]
        self.trainindex = (ii//batch_size)*batch_size
        self.testindex = (testii//batch_size)*batch_size


    def on_train_begin(self, logs=None):
        self.trainCindex = []
        self.testCindex = []
        #self.trmae = []
        #self.tsmae = []
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        if epoch % 1 == 0:
            P = self.predictmodel.predict([self.trainset['x'][0:self.trainindex]],
                                          batch_size=self.batchsize)
            CI = cindex(self.trainset['t'][0:self.trainindex], P, self.trainset['e'][0:self.trainindex])

            #trmae = mae(self.trainset['t'][0:self.trainindex],P,self.trainset['e'][0:self.trainindex])

            tsP = self.predictmodel.predict([self.testset['x'][0:self.testindex]],
                                          batch_size=self.batchsize)
            tsCI = cindex(self.testset['t'][0:self.testindex], tsP, self.testset['e'][0:self.testindex])

            #tsmae = mae(self.testset['t'][0:self.testindex],tsP,self.testset['e'][0:self.testindex])

            print CI,tsCI,trmae,tsmae
            self.trainCindex.append(CI)
            self.testCindex.append(tsCI)
            self.trmae.append(trmae)
            self.tsmae.append(tsmae)
            history = {'traincindex': self.trainCindex,'testcindex': self.testCindex, 'loss': self.losses}
            with open('../logs/' + self.weight + '_log.txt', 'w') as f:
                f.write(str(history))
                #print 'test Cindex', self.Cindex
                print '\n'
                #self.predictmodel.save_weights('./weight/bestCI/' + self.weight + '_CI_%02d.h5' % (epoch))
                print 'saveCI successful'
def mae(t,p,e):
    t=np.array(t).reshape(np.shape(t)[0],1)
    e=np.array(e).reshape(np.shape(e)[0],1)
    tm = np.mean(t*e)
    p=np.array(p)
    pm = np.mean(p*e)
    #ab = np.abs((t-tm) - (p-pm)) * e
    ab = np.abs(t - p) * e
    mae = np.sum(ab) / np.sum(e)
    return mae

def bootstrap_metric(predictmodel, dataset,batchsize, N=100):
    def sample_dataset(dataset, sample_idx):
        sampled_dataset = {}
        for (key, value) in dataset.items():
            sampled_dataset[key] = value[sample_idx]
        return sampled_dataset

    metrics = []
    size = len(dataset['x'])

    for _ in range(N):
        resample_idx = np.random.choice(size, size=size, replace=True)
        samp_dataset = sample_dataset(dataset, resample_idx)
        index = (len(resample_idx) // batchsize) * batchsize
        pred_y = predictmodel.predict(samp_dataset['x'][0:index],batch_size=batchsize)
        metric = cindex(samp_dataset['t'][0:index],pred_y,samp_dataset['e'][0:index])
        metrics.append(metric)

    # Find mean and 95% confidence interval
    mean = np.mean(metrics)
    conf_interval = st.t.interval(0.95, len(metrics) - 1, loc=mean, scale=st.sem(metrics))
    return {
        'mean': mean,
        'confidence_interval': conf_interval
    }

def evaluate_model(predictmodel,testset,name,batchsize,bootstrap=True):
    metrics = {}

    with open('../pResult/'+name+'.txt','w') as f:
        ii = np.shape(testset['t'])[0]
        index = (ii // batchsize) * batchsize
        P = predictmodel.predict([testset['x'][0:index]],batch_size=batchsize)
        for line in range(0,P.shape[0]):
            predline = '{}**{}**{}\n'.format(testset['t'][line], testset['e'][line], P[line][0])
            f.write(predline)
    metrics['C-index'] = cindex(testset['t'][0:index], P, testset['e'][0:index])
    #metrics = cindex(testset['t'][0:index], P, testset['e'][0:index])
    if bootstrap==True:
        metrics['C-index-bootstrap'] = bootstrap_metric(predictmodel,testset,batchsize)

    return metrics


def trainmodel(trainset,
               testset,
               start_epoch,
               batchsize,
               n_in,
               numepoch,
               l2,
               name,
               activation,
               dropout,
               hidden_layers_sizes,
               batchnorm,
               learning_rate,
               lr_decay,
               momentum ,):
    trainnum = trainset['x'].shape[0]
    print trainnum
    print  testset['x'].shape[0]
    pairwise = _pairwise_all(trainset)
    print pairwise.shape[0]
    batchnum = pairwise.shape[0]//batchsize-2  # del index[i]
    gen = Generator(trainset,batchsize)
    generator = gen.loaddata(batchnum,pairwise)


    mymodel = MyModel(loss=rankmseloss,batchsize=batchsize,batchnorm=batchnorm)
    predictmodel,model= mymodel.model(n_in,l2,activation=activation,dropout=dropout,hidden_layers_sizes=hidden_layers_sizes)
    optimizer = SGD(lr=learning_rate,momentum=momentum,decay=lr_decay,nesterov=True,clipnorm=5)
    #optimizer = Adam(lr=learning_rate,decay=lr_decay)
    model.compile(optimizer=optimizer,loss=lossfunction)

    zHis = VizCallBack(trainset,testset, predictmodel, batch_size=batchsize,weightname=name)
    #tsHis = VizCallBack(testset, predictmodel, batch_size=batchsize,weightname=name,evalname='traincindex')
    model.fit_generator(generator,steps_per_epoch=batchnum,epochs=numepoch,callbacks=[zHis],verbose=1)


    predictmodel.save_weights('../weight/'+name+'.h5')

    metrics = evaluate_model(predictmodel,testset=testset,name=name,batchsize=batchsize,bootstrap=True)

    print metrics



if __name__ == '__main__':

    SUPPORT = '../Parameter/SUPPORT_RankDeepSurv.json'
    METABRIC = '../Parameter/METABRIC_RankDeepSurv.json'
    GBSG = "../Parameter/GBSG_RankDeepSurv.json"
        #GBSG = "/media/sysucc99/data/ZTROOT/2017MainText/Parameter/GBSG_RankDeepSurv.json"
    whas = '../Parameter/WHAS_RankDeepSurv.json'
    npc_A_excel_last_2_PFSmonths = '../2017MainText/Parameter/npc_A_excel_last_2_PFSmonths_RankDeepSurv.json'
    npc_B_excel_last_2_PFSmonths = '../2017MainText/Parameter/npc_B_excel_last_2_PFSmonths_RankDeepSurv.json'


    jsonmodel = SUPPORT
    with open(jsonmodel, 'r') as fp:

        json_model = fp.read()
        hyperparams = json.loads(json_model)

    name = os.path.basename(jsonmodel)
    name,_ = os.path.splitext(name)
    print name

    datapath = hyperparams['datapath']
    for i in [1,2,3,4,5]:
        print datapath

    dataset = load_datasets(datapath)

    start_epoch=None
    batchsize=hyperparams['batch_size']
    numepoch=hyperparams['numepoch']
    l2=hyperparams['L2']
    activation = hyperparams['activation']
    dropout = hyperparams['dropout']
    hidden_layers_sizes = hyperparams['hidden_layers_sizes']
    print hidden_layers_sizes
    batchnorm = hyperparams['batch_norm']
    learning_rate = hyperparams['learning_rate']
    lr_decay = hyperparams['lr_decay']
    momentum = hyperparams['momentum']
    n_in = hyperparams['n_in']

    trainmodel(trainset=dataset['train'],testset=dataset['test'],start_epoch=start_epoch,batchsize=batchsize,n_in=n_in,
               numepoch=numepoch,l2=l2,name=name,activation=activation,dropout=dropout,hidden_layers_sizes=hidden_layers_sizes,
               batchnorm=batchnorm,learning_rate=learning_rate,lr_decay=lr_decay,momentum=momentum)



    print 'train end'
    # from Tool.plotplot import plo
    # plo(name)
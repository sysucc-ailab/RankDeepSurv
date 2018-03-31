# -*- coding: utf-8 -*-

import pandas as pd
import random
import numpy as np
import argparse
from easydict import EasyDict as edict
from collections import defaultdict

# Force matplotlib to not use any Xwindows backend.
import matplotlib
matplotlib.use('Agg')

import numpy as np
import h5py
import uuid
#import copy
import json

import sys, os


from deepsurv import viz
from deepsurv import utils
from deepsurv.deepsurv_logger import TensorboardLogger
from deepsurv.deep_surv import load_model_from_json

import time
localtime   = time.localtime()
TIMESTRING  = time.strftime("%m%d%Y%M", localtime)

def load_datasets(dataset_file):
    datasets = defaultdict(dict)

    with h5py.File(dataset_file, 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]
    trainset = datasets['train']
    valid = datasets['test']
    trainset = {'x':trainset['x'],'t':trainset['t'],'e':trainset['e']}
    validset = {'x':valid['x'],'t':valid['t'],'e':valid['e']}
    newset = {'train':trainset,'test':validset}
    return newset

parse_ARGS = edict()
parse_ARGS.experiment = 'train'
parse_ARGS.model = '/media/sysucc99/data/ZTROOT/2017MainText/Parameter/npc_B_excel_DeepSurv.json'
#parse_ARGS.dataset = './data/whas_train_test.h5'
parse_ARGS.update_fn = 'adam'
parse_ARGS.plot_error = True
parse_ARGS.treatment_idx = 1
parse_ARGS.results_dir = '../pResult/'
parse_ARGS.weights = None
parse_ARGS.num_epochs = 500


def evaluate_model(model, dataset, bootstrap=False):
    def mse(model):
        def deepsurv_mse(x, hr, **kwargs):
            hr_pred = np.squeeze(model.predict_risk(x))
            return ((hr_pred - hr) ** 2).mean()

        return deepsurv_mse

    metrics = {}

    # Calculate c_index
    metrics['c_index'] = model.get_concordance_index(**dataset)
    if bootstrap:
        metrics['c_index_bootstrap'] = utils.bootstrap_metric(model.get_concordance_index, dataset)

    # Calcualte MSE
    if 'hr' in dataset:
        metrics['mse'] = mse(model)(**dataset)
        if bootstrap:
            metrics['mse_bootstrap'] = utils.bootstrap_metric(mse(model), dataset)

    return metrics


def save_risk_surface_visualizations(model, dataset, norm_vals, output_dir, plot_error, experiment,
                                     trt_idx):
    if experiment == 'linear':
        clim = (-3, 3)
    elif experiment == 'gaussian' or experiment == 'treatment':
        clim = (-1, 1)
    else:
        clim = (0, 1)

    risk_fxn = lambda x: np.squeeze(model.predict_risk(x))
    color_output_file = os.path.join(output_dir, "deep_viz_color_" + TIMESTRING + ".pdf")
    viz.plot_experiment_scatters(risk_fxn, dataset, norm_vals=norm_vals,
                                 output_file=color_output_file, figsize=(4, 3), clim=clim,
                                 plot_error=plot_error, trt_idx=trt_idx)

    bw_output_file = os.path.join(output_dir, "deep_viz_bw_" + TIMESTRING + ".pdf")
    viz.plot_experiment_scatters(risk_fxn, dataset, norm_vals=norm_vals,
                                 output_file=bw_output_file, figsize=(4, 3), clim=clim, cmap='gray',
                                 plot_error=plot_error, trt_idx=trt_idx)


def save_treatment_rec_visualizations(model, dataset, output_dir,
                                      trt_i=1, trt_j=0, trt_idx=0):
    trt_values = np.unique(dataset['x'][:, trt_idx])
    print("Recommending treatments:", trt_values)
    rec_trt = model.recommend_treatment(dataset['x'], trt_i, trt_j, trt_idx)
    rec_trt = np.squeeze((rec_trt < 0).astype(np.int32))

    rec_dict = utils.calculate_recs_and_antirecs(rec_trt, true_trt=trt_idx, dataset=dataset)

    output_file = os.path.join(output_dir, '_'.join(['deepsurv', TIMESTRING, 'rec_surv.pdf']))
    print(output_file)
    viz.plot_survival_curves(experiment_name='DeepSurv', output_file=output_file, **rec_dict)


def save_model(model, output_file):
    model.save_weights(output_file)


if __name__ == '__main__':
    # args = parse_args()
    args = parse_ARGS
    # print("Arguments:",args)
    print args

    # Load Dataset
    #print("Loading datasets: " + args.dataset)
    #datasets = utils.load_datasets(args.dataset)
    datapath = '/media/sysucc99/data/ZTROOT/2017MainText/data/npc_B_excel.h5'
    datasets = load_datasets(datapath)
    norm_vals = {
        'mean': datasets['train']['x'].mean(axis=0),
        'std': datasets['train']['x'].std(axis=0)
    }

    # Train Model
    tensor_log_dir = "./logs/tensorboard_" + 'train' + "_" + str(uuid.uuid4())
    logger = TensorboardLogger("experiments.deep_surv", tensor_log_dir, update_freq=10)
    model = load_model_from_json(args.model, args.weights)
    if 'valid' in datasets:
        valid_data = datasets['valid']
    else:
        valid_data = None
    metrics = model.train(datasets['train'], valid_data, n_epochs=args.num_epochs, logger=logger,
                          update_fn=utils.get_optimizer_from_str(args.update_fn),
                          validation_frequency=100)

    # Evaluate Model
    with open(args.model, 'r') as fp:
        json_model = fp.read()
        hyperparams = json.loads(json_model)

    train_data = datasets['train']
    if hyperparams['standardize']:
        train_data = utils.standardize_dataset(train_data, norm_vals['mean'], norm_vals['std'])

    metrics = evaluate_model(model, train_data)
    print("Training metrics: " + str(metrics))
    if 'valid' in datasets:
        valid_data = datasets['valid']
        if hyperparams['standardize']:
            valid_data = utils.standardize_dataset(valid_data, norm_vals['mean'], norm_vals['std'])
            metrics = evaluate_model(model, valid_data)
        print("Valid metrics: " + str(metrics))

    if 'test' in datasets:
        test_dataset = utils.standardize_dataset(datasets['test'], norm_vals['mean'], norm_vals['std'])
        metrics = evaluate_model(model, test_dataset, bootstrap=True)
        print("Test metrics: " + str(metrics))

    if 'viz' in datasets:
        print("Saving Visualizations")
        save_risk_surface_visualizations(model, datasets['viz'], norm_vals=norm_vals,
                                         output_dir=args.results_dir, plot_error=args.plot_error,
                                         experiment=args.experiment, trt_idx=args.treatment_idx)

    if 'test' in datasets and args.treatment_idx is not None:
        print("Calculating treatment recommendation survival curvs")
        # We use the test dataset because these experiments don't have a viz dataset
        save_treatment_rec_visualizations(model, test_dataset, output_dir=args.results_dir,
                                          trt_idx=args.treatment_idx)

    if args.results_dir:
        # _, model = os.path.split(args.model)
        output_file = os.path.join(args.results_dir, "models/") + str(uuid.uuid4()) + ".h5"
        print("Saving model parameters to output file", output_file)
        save_model(model, output_file)

    exit(0)
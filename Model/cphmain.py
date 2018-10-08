import sys, os
import matplotlib

matplotlib.use('Agg')
from Tool import utils
import argparse

import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

import logging
import time
localtime = time.localtime()
TIMESTRING = time.strftime("%m%d%Y%M", localtime)
DURATION_COL = 'time'
EVENT_COL = 'censor'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('experiment', help='name of the experiment that is being run')

    parser.add_argument('dataset', help='.h5 File containing the train/valid/test datasets')

    parser.add_argument('--results_dir', default='/media/sysucc321/data/ZTROOT/2017MainText/pResult',
                        help='Directory to save resulting models and visualizations')

    parser.add_argument('--plot_error', action="store_true", help="If arg present, plot absolute error plots")

    parser.add_argument('--treatment_idx', default=None, type=int,
                        help='(Optional) column index of treatment variable in dataset. If present, run treatment visualizations.')

    return parser.parse_args()


def evaluate_model(model, dataset, bootstrap=False):
    def ci(model):

        def cph_ci(x, t, e, **kwargs):
            return concordance_index(

                event_times=t,

                predicted_event_times=-model.predict_partial_hazard(x),

                event_observed=e,

            )

        return cph_ci

    def mse(model):

        def cph_mse(x, hr, **kwargs):
            hr_pred = np.squeeze(-model.predict_partial_hazard(x).values)

            return ((hr_pred - hr) ** 2).mean()

        return cph_mse

    metrics = {}

    # Calculate c_index

    metrics['c_index'] = ci(model)(**dataset)

    if bootstrap:
        metrics['c_index_bootstrap'] = utils.bootstrap_metric(ci(model), dataset)

    # Calcualte MSE
    risk =  np.array(-model.predict_partial_hazard(dataset['x']))
    risk.tolist()
    t = dataset['t'].tolist()
    e = dataset['e'].tolist()

    with open('../pResult/'+'cph_B.txt','w') as f:
        for line in range(0,len(t)):
            predline = '{}**{}**{}\n'.format(t[line], e[line], risk[line][0])
            f.write(predline)


    #plot_curve(t,e,risk)

    if 'hr' in dataset:

        metrics['mse'] = mse(model)(**dataset)

        if bootstrap:
            metrics['mse_bootstrap'] = utils.bootstrap_metric(mse(model), dataset)

    return metrics



if __name__ == '__main__':

    logger = logging.getLogger(__name__)

    logger.setLevel(logging.DEBUG)

    args = parse_args()

    print("Arguments:", args)

    # Load Dataset

    print("Loading datasets: " + args.dataset)

    datasets = utils.load_datasets(args.dataset)

    # Train CPH model

    print("Training CPH Model")

    train_df = utils.format_dataset_to_df(datasets['train'], DURATION_COL, EVENT_COL)

    cf = CoxPHFitter()

    results = cf.fit(train_df, duration_col=DURATION_COL, event_col=EVENT_COL)

    cf.print_summary()

    print("Train Likelihood: " + str(cf._log_likelihood))


    if 'test' in datasets:
        metrics = evaluate_model(cf, datasets['test'], bootstrap=True)

        print("Test metrics: " + str(metrics))


    exit(0)

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import numpy as np
import pandas as pd
import pylab
import os

import time

localtime = time.localtime()

TIMESTRING = time.strftime("%m%d%Y%M", localtime)


def plot_survival_curves(A,B, output_dir='../pResult/pdf/'):
    # Set-up plots
    #pd.Series(low_t), pd.Series(low_e), pd.Series(high_t), pd.Series(high_e)
    output_file = os.path.join(output_dir, '_'.join(['pfs', TIMESTRING, 'rank_cph.pdf']))
    plt.figure(figsize=(12, 3))

    ax = plt.subplot(111)

    # Fit survival curves

    kmf = KaplanMeierFitter()

    kmf.fit(pd.Series(A[0]), event_observed=pd.Series(A[1]), label=' '.join(['rankdeepsurv', "high"]))

    kmf.plot(ax=ax, linestyle="-")

    kmf.fit(pd.Series(A[2]), event_observed=pd.Series(A[3]), label=' '.join(['rankdeepsurv', "low"]))

    kmf.plot(ax=ax, linestyle="--")

    kmf.fit(pd.Series(B[0]), event_observed=pd.Series(B[1]), label=' '.join(['cph', "high"]))

    kmf.plot(ax=ax, linestyle="-")

    kmf.fit(pd.Series(B[2]), event_observed=pd.Series(B[3]), label=' '.join(['cph', "low"]))

    kmf.plot(ax=ax, linestyle="--")

    # Format graph

    plt.ylim(0.5, 1);

    ax.set_xlabel('Timeline (months)', fontsize='large')

    ax.set_ylabel('Survival Rate', fontsize='large')

    # Calculate p-value

    results_rank = logrank_test(pd.Series(A[0]), pd.Series(A[2]), pd.Series(A[1]), pd.Series(A[3]), alpha=.95)
    results_cph = logrank_test(pd.Series(B[0]), pd.Series(B[2]), pd.Series(B[1]), pd.Series(B[3]), alpha=.95)

    results_rank.print_summary()
    results_cph.print_summary()

    # Location the label at the 1st out of 9 tick marks
    print results_rank
    print results_cph


    xloc = max(np.max(pd.Series(A[0])), np.max(pd.Series(A[2])),np.max(pd.Series(B[0])), np.max(pd.Series(B[2]))) / 9

    if results_rank.p_value < 1e-5 :

        ax.text(xloc, .2, '$p < 1\mathrm{e}{-5}$', fontsize=30)

    else:

        ax.text(xloc, .2, '$p=%f$' % results_rank.p_value, fontsize=12)

    plt.legend(loc='best', prop={'size': 10})

    if output_file:
        plt.tight_layout()

        pylab.savefig(output_file)

def highlow(t,e,p):
    print type(t),type(e),type(p)
    aa3 = np.array(p)
    med = np.median(aa3)
    low = [i for i,j in enumerate(p) if j >med]
    high = [i for i,j in enumerate(p) if j <=med]

    print len(high),len(low)
    print low
    print len(e)
    high_t = [t[i] for i in high]
    high_e = [e[i] for i in high]
    low_t = [t[i] for i in low]
    low_e = [e[i] for i in low]
    return (high_t,high_e,low_t,low_e)

def readdata(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    t = list()
    e = list()
    p = list()
    print len(lines)
    for line in lines:
        a1,a2,a3 = line.split('**')
        t.append(float(a1))
        e.append(int(a2))
        try:
            p.append(float(a3))
        except:
            print a3
            assert 1==2

    return  t,e,p
if __name__ == '__main__':

    path_ranksurv='../2017MainText/pResult/npc_B_excel_last_2_PFSmonths_RankDeepSurv.txt'
    path_cph = '../ZTROOT/2017MainText/pResult/cph_B.txt'
    #path_ranksurv = '/media/sysucc321/data/ZTROOT/2017MainText/pResult/cph_B.txt'
    r_t,r_e,r_p = readdata(path_ranksurv)
    cph_t,cph_e,cph_p = readdata(path_cph)
    ranksurv = highlow(r_t,r_e,r_p)
    #orgin = highlow(r_t,r_e,r_t)
    cph = highlow(cph_t,cph_e,cph_p)

    plot_survival_curves(ranksurv,cph)

    # low_t = [45 if i>=45 else i for i in antirec_t]
    # low_e = [0 if i in high else 1 for i,j in enumerate(antirec_e)]
    # plot_survival_curves(pd.Series(rec_t),pd.Series(e),pd.Series(low_t),pd.Series(low_e))



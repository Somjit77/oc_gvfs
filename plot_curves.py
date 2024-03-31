import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D

import matplotlib
font = {'size': 18,
        'weight' : 'bold'}
matplotlib.rc('font', **font)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('axes', labelsize="22")

import os
import tensorflow as tf
import numpy as np

def running_mean(X, N):
    cumsum = np.cumsum(np.insert(X, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

#Get latest file
def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    if len(paths) == 0:
      return None
    else:
      return max(paths, key=os.path.getctime)

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

summary_iterator = tf.compat.v1.train.summary_iterator

vers = ['0.0']
algs = ['dqn', 'gvf', 'dis', 'hc_gvf', 'gvf_esp', 'hc_gvf_esp', 'dis_esp', 'sa_esp', 'sa']
alg_label = {'dqn': 'DDQN', 'gvf': 'Random-GVFs', 'hc_gvf': 'HC-GVFs', 'dis': 'Dis-Aux-GVFs',
             'gvf_esp': 'Random-GVFs+', 'hc_gvf_esp': 'HC-GVFs+', 'dis_esp': 'Dis-Aux-GVFs+',
             'sa_esp': 'OC-GVFs', 'sa': 'without SA'}
fig, ax = plt.subplots(figsize=(32, 8), nrows=1, ncols=1)
for ver in vers:
    print(ver)
    title = 'CoinRun (Easy)'
    run_no = [0, 1, 2, 3, 4]
    runs = len(run_no)
    episodes = 5000
    N = 200
    X_axis = running_mean(np.arange(episodes), N)
    name = "Set1"
    cmap = get_cmap(name)
    colors = iter(cmap.colors)  # type: list
    labels = ['DDQN', 'Random-GVFs', 'Hand-Crafted GVFs', 'Dis-Aux-GVFs', 'OC-GVFs']
    custom_lines = [Line2D([0], [0], color=list(cmap.colors)[0], lw=1),
                Line2D([0], [0], color=list(cmap.colors)[1], lw=1),
                Line2D([0], [0], color=list(cmap.colors)[3], lw=1),
                Line2D([0], [0], color=list(cmap.colors)[2], lw=1),
                Line2D([0], [0], color=list(cmap.colors)[7], lw=1)]
    for alg in algs:
        total_reward = []
        color = next(colors)
        skip_alg = True
        for run in run_no:
            alg_ver = ver
            print(f'Alg:{alg}, Run:{run}')
            save_dir = os.getcwd() + f'/Results/logs-{alg_ver}/{alg}_run_{run+1000}/'  # Save Directory
            rewards = []
            try:
                latest_checkpoint = newest(save_dir)
                for e in summary_iterator(newest(save_dir)):    
                    for v in e.summary.value:
                        if v.tag == 'Training_returns':            
                            rewards.append(v.simple_value)
                skip_alg = False
            except:
                print('Skipping Run')
                continue
            print(len(rewards))
            total_reward.append(rewards)

        if not skip_alg:
            curve_mean, curve_std = tolerant_mean(total_reward)
            curve_std = curve_std/(np.sqrt(runs))
            ax.plot(running_mean(curve_mean, N), color=color, label=alg_label[alg])
            ax.fill_between(np.arange(len(running_mean(curve_mean, N))), running_mean(curve_mean + curve_std, N), running_mean(curve_mean - curve_std, N),
                        color=color, alpha=0.25)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Rewards')
    
plt.tight_layout()
fig.subplots_adjust(bottom=0.2)
fig.legend(custom_lines, labels, loc="lower center", ncol=5)
plt.savefig(f'coinrun.png', dpi=120)
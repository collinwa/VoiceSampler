import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import importlib as lib
import ChordClassifier as cc

from torch.nn import Conv1d, BatchNorm1d, ReLU, Linear
from utils import *
from parameters import *

lib.reload(cc)

BATCH_SIZE = 10 

def make_batch(df, chord, b_s=32):
    match_dfs = df[df.chord == chord]
    match_dfs = match_dfs.sample(b_s)
    diff_dfs = df[df.chord != chord]
    diff_dfs = diff_dfs.sample(b_s)

    match_x1 = []
    match_x2 = []

    diff_x1 = []
    diff_x2 = []

    for i in range(0, len(match_dfs), 2):
        _, t = read(match_dfs.iloc[i].fname)
        match_x1.append(t)

    for j in range(1, len(match_dfs), 2):
        _, t = read(match_dfs.iloc[i].fname)
        match_x2.append(t)

    for i in range(0, len(diff_dfs), 2):
        _, t = read(diff_dfs.iloc[i].fname)
        diff_x1.append(t)

    for i in range(1, len(diff_dfs), 2):
        _, t = read(diff_dfs.iloc[i].fname)
        diff_x2.append(t)

    x1 = torch.cat(match_x1 + diff_x1, 0)
    x2 = torch.cat(match_x2 + diff_x2, 0)

    y = torch.cat((torch.ones(len(match_x1), 1), torch.zeros(len(diff_x2), 1)), 0)

    # indices = np.arange(len(match_x1) + len(diff_x2))
    # np.random.shuffle(indices)

    x1_out = x1[:,:,:]
    x2_out = x2[:,:,:]
    y_out = y[:,:]

    return x1_out, x2_out, y_out

if __name__ == '__main__':
    net = cc.SiameseArchitecture(**SETTINGS)    

    refset_df = pd.read_csv(REF_CSV, header=None, sep=',', names=['fname', 'chord', 'key'])    
    refset_df.loc[:, 'fname'] = REF_DIR + refset_df[['fname']]
    refset_df['chord'] = refset_df['chord'] + '_' + refset_df['key']
    refset_df = refset_df.drop(columns=['key'])

    df = pd.read_csv(ALT_CSV, header=None, sep=',', names=['fname', 'chord', 'key'])
    df.loc[:, 'fname'] = ALT_DIR + df[['fname']]
    df['chord'] = df['chord'] + '_' + df['key']
    df = df.drop(columns=['key'])    

    df = refset_df.append(df)
	
	optimizer = torch.optim.Adam(net.parameters(), lr=0.000001,  amsgrad=True)
	optimizer.zero_grad()

	for _ in range(0, 10):
	    x1, x2, y = make_batch(df, 'd_sharp_major', b_s=BATCH_SIZE)
	    probs = net(x1, x2)
	    loss = F.binary_cross_entropy(probs, y.detach())
	    thresh = probs > 0.5
	    acc = torch.sum(thresh == y).item() / torch.numel(y)

	    print('loss: {} acc: {}'.format(loss, acc))

	    loss.backward()
	    optimizer.step()
	    optimizer.zero_grad()


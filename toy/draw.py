#!/bin/bash
import torch
import torch.utils.data as data

import os
import numpy as np
import warnings
import random
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from config import opt
from model import STRODE
from parse_datasets import parse_datasets
from utils import *

seed = opt.seed
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_dataset = parse_datasets(opt, 'test', device)

test_dataloader = data.DataLoader(
    dataset=test_dataset,
    shuffle=True,
    batch_size=opt.test_batch_size,
    num_workers=0)

model = STRODE(in_dim=1, h_dim=opt.h_dim, kernel_size=5, num_layers=2, device=device).to(device)
state_dict = torch.load(opt.exp+'/best_nnet.pth')
model.load_state_dict(state_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
log_path = opt.exp + '/' + str(seed) + '_out.log'
logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))

def cal_cs(t_pred, t_gt):
    n = t_gt.shape[0]
    total = 0
    for i in range(n):
        cs = cosine_similarity(t_pred[i:i+1], t_gt[i:i+1])
        total += cs
    cs = total/n
    return cs

def test(model, test_data_loader):
    model.eval()
    loss_total = 0
    total_cs = 0.
    total_mse = 0.

    t_pred_ls = []
    t_gt_ls = []
    for i, (data, _) in enumerate(test_data_loader):
        data = data.to(device)
        data_test = data[:, :, :1]
        t_gt = data[:, :, 1]

        with torch.no_grad():
            loss, t_pred, mse, _ = model(data_test, 'test')
            loss_total += float(loss)
            t_pred = t_pred.detach()
            cs = cal_cs(t_pred, t_gt)

            t_pred_ls.append(t_pred)
            t_gt_ls.append(t_gt)

            total_cs += cs
            total_mse += mse

    cs = total_cs/len(test_data_loader)
    mse = total_mse/len(test_data_loader)

    t_pred = torch.cat(t_pred_ls, dim=0)
    t_gt = torch.cat(t_gt_ls, dim=0)

    import seaborn as sns
    import pandas as pd
    t_pred = t_pred.reshape(-1)
    t_pred = (t_pred - min(t_pred))/(max(t_pred) - min(t_pred))
    t_gt = t_gt.reshape(-1)

    df = {'Ground-truth Boundary Time': t_gt, "Inferred Boundary Time": t_pred}
    df = pd.DataFrame(df)

    sns.set(style='white', font_scale=1.2)

    g = sns.JointGrid(data=df, x='Ground-truth Boundary Time', y='Inferred Boundary Time',
                      xlim=(0, 2), ylim=(0, 1), height=5)
    g = g.plot_joint(sns.regplot, color="xkcd:muted blue", scatter_kws={'alpha': 0.2})
    g = g.plot_marginals(sns.distplot, kde=False, bins=12, color="xkcd:bluey grey")
    g.ax_joint.text(1.2, 0.1, 'CS = '+str(round(cs[0][0], 3)), fontstyle='italic')
    plt.tight_layout()
    plt.savefig("pp.pdf")

    logger.info("TEST COS Sim: %f MSE: %f" % (cs, mse))

    return mse

mse = test(model, test_dataloader)
  


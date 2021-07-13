#!/bin/bash
import torch
import torch.utils.data as data

import os
import numpy as np
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='hks', choices=['pois', 'hks'])
parser.add_argument('--num_epoch', default=50, type=int)

args = parser.parse_args()

# model name and path 
opt.model = 'strode_toy_'+args.dataset
opt.model_path = './exp/'+opt.model
if not os.path.exists(opt.model_path):
    os.mkdir(opt.model_path)
# path for a specific exp
opt.exp = opt.model_path+'/'+str(opt.lr)+'_kl'+str(opt.w_kl) + \
    '_step'+str(opt.ode1_step)+'_eps'+str(opt.ode2_eps)+'_k'+str(opt.k)
if not os.path.exists(opt.exp):
    os.mkdir(opt.exp)

train_dataset = parse_datasets(opt, 'train', args.dataset, device)
val_dataset = parse_datasets(opt, 'val', args.dataset, device)
test_dataset = parse_datasets(opt, 'test', args.dataset, device)

train_dataloader = data.DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=opt.train_batch_size,
    num_workers=0)

val_dataloader = data.DataLoader(
    dataset=val_dataset,
    shuffle=True,
    batch_size=opt.val_batch_size,
    num_workers=0)

test_dataloader = data.DataLoader(
    dataset=test_dataset,
    shuffle=True,
    batch_size=opt.test_batch_size,
    num_workers=0)

model = STRODE(in_dim=1, h_dim=opt.h_dim, kernel_size=5, num_layers=2, device=device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
log_path = opt.exp+'/out.log'
logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))


def cal_cs(t_pred, t_gt):
    n = t_gt.shape[0]
    total = 0
    for i in range(n):
        cs = cosine_similarity(t_pred[i:i+1], t_gt[i:i+1])
        total += cs
    cs = total/n
    return cs


def train(model, train_data_loader, optimizer, epoch):
    model.train()
    for i, (data, _) in enumerate(train_data_loader):
        optimizer.zero_grad()
        data = data.to(device)
        data_train = data[:, :, :1]
        t_gt = data[:, :, 1]

        loss, t_pred, mse, kl = model(data_train)
        loss.backward()
        optimizer.step()

        if i % 30 == 0:
            logger.info("Epoch: %d, Step: %d, Loss: %f MSE: %f, KL: %f" % (epoch, i, loss, mse, kl))
            t_pred = t_pred.detach()
            cs = cal_cs(t_pred, t_gt)
            logger.info("COS Sim: %f " % (cs))

def val(model, val_data_loader):
    model.eval()
    loss_total = 0
    total_cs = 0.
    total_mse = 0.

    for i, (data, _) in enumerate(val_data_loader):
        data = data.to(device)
        data_test = data[:, :, :1]
        t_gt = data[:, :, 1]

        with torch.no_grad():
            loss, t_pred, mse, _ = model(data_test, 'test')
            loss_total += float(loss)
            t_pred = t_pred.detach()
           
            cs = cal_cs(t_pred, t_gt)
            total_cs += cs
            total_mse += mse

    cs = total_cs/len(val_data_loader)
    mse = total_mse/len(val_data_loader)
    logger.info("Val COS Sim: %f MSE: %f" % (cs, mse))
    
    return mse

def test(model, test_data_loader):
    model.eval()
    loss_total = 0
    total_cs = 0.
    total_mse = 0.

    for i, (data, _) in enumerate(test_data_loader):
        data = data.to(device)
        data_test = data[:, :, :1]
        t_gt = data[:, :, 1]

        with torch.no_grad():
            loss, t_pred, mse, _ = model(data_test, 'test')
            loss_total += float(loss)
            t_pred = t_pred.detach()
           
            cs = cal_cs(t_pred, t_gt)
            total_cs += cs
            total_mse += mse

    cs = total_cs/len(test_data_loader)
    mse = total_mse/len(test_data_loader)
    logger.info("TEST COS Sim: %f MSE: %f" % (cs, mse))
    
    return mse

def test_draw(model, test_data_loader):
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
    plt.savefig("out.pdf")

    logger.info("TEST COS Sim: %f MSE: %f" % (cs, mse))

    return mse

min_mse = 10000
for e in range(opt.epoch):
    train(model, train_dataloader, optimizer, e)
    mse = val(model, val_dataloader)
    mse_t = test(model, test_dataloader)
    if mse < min_mse:
        min_mse = mse
        torch.save(model.state_dict(), opt.exp + '/best_nnet.pth')

    logger.info("Best MSE: %f" % (min_mse))

state_dic = torch.load(opt.exp + '/best_nnet.pth')
model.load_state_dict(state_dic)

mse = test_draw(model, test_dataloader)

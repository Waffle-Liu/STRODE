#!/bin/bash
import torch
from torch.functional import split
import torch.utils.data as data

import os
import numpy as np
import warnings
import random
import argparse
from sklearn.metrics.pairwise import cosine_similarity

from config import opt
from rotateMNIST import RotateMNIST
from model import NODE, NODE_RNN, STRODE, STRODE_RNN
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='strode', choices=['node', 'node_rnn', 'strode', 'strode_rnn'])
parser.add_argument('--dataset', default='exp', choices=['exp', 'hks'])
parser.add_argument('--num_epoch', default=50, type=int)

args = parser.parse_args()

model_pool = {
    'node': NODE,
    'node_rnn': NODE_RNN,
    'strode': STRODE,
    'strode_rnn': STRODE_RNN
}

opt.model = args.model
opt.model_path = './exp/'+opt.model
if not os.path.exists(opt.model_path):
    os.mkdir(opt.model_path)
# path for a specific exp
opt.exp = opt.model_path+'/'+str(opt.lr)+'_kl'+str(opt.w_kl) + \
        '_step'+str(opt.ode1_step)+'_eps'+str(opt.ode2_eps)+'_k'+str(opt.k)+'_'+args.dataset
if not os.path.exists(opt.exp):
    os.mkdir(opt.exp)

seed = opt.seed
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpunumber)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
warnings.filterwarnings('ignore')
device = torch.device('cuda:'+str(opt.gpunumber) if torch.cuda.is_available() else 'cpu')

train_dataset = RotateMNIST(opt, args.dataset, mode='train')
val_dataset = RotateMNIST(opt, args.dataset, mode='val')
test_dataset = RotateMNIST(opt, args.dataset, mode='test')


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

# define model
model = model_pool[args.model](in_dim=1, h_dim=opt.h_dim, kernel_size=5, device=device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
log_path = opt.exp + '/out.log'
logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))

def cal_cs(t_pred, t_gt):
    n = t_gt.shape[0]
    total = 0
    for i in range(n):
        cs = cosine_similarity(t_pred[i:i+1], t_gt[i:i+1])
        total += cs
    cs_avg = total/n

    return cs_avg


def train(model, train_data_loader, optimizer, epoch):
    model.train()

    for i, (data, _, gt_time) in enumerate(train_data_loader):
        optimizer.zero_grad()
        data = data.to(device)

        loss, mse, kl, t_pred = model(data)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            logger.info("Epoch: %d, Step: %d, Loss: %f, MSE: %f, KL: %f" % (epoch, i, loss, mse, kl))
            mono_inc = np.array([np.arange(opt.video_length) for i in range(50)])
            cs_ours = cal_cs(t_pred.detach().cpu(), gt_time)
            cs_baseline = cal_cs(mono_inc, gt_time)
            logger.info("Train Cos Sim(ours/baseline): %f, %f" % (cs_ours, cs_baseline))


def val(model, val_data_loader):
    model.eval()
    mse_total = 0
    cs_ours_total = 0
    cs_baseline_total = 0
    
    for i, (data, _, gt_time) in enumerate(val_data_loader):
        data = data.to(device)

        with torch.no_grad():
            _, mse, _, t_pred = model(data, 'test')
           
            mse_total += float(mse)
            mono_inc = np.array([np.arange(opt.video_length) for i in range(50)])
            cs_ours = cal_cs(t_pred.detach().cpu(), gt_time)
            cs_baseline = cal_cs(mono_inc, gt_time)
            cs_ours_total += float(cs_ours)
            cs_baseline_total += float(cs_baseline)

    cur_mse = mse_total/len(val_data_loader)
    cur_cs_ours = cs_ours_total/len(val_data_loader)
    cur_cs_baseline = cs_baseline_total/len(val_data_loader)
    logger.info("Val MSE: %f, Cos Sim(ours/baseline): %f, %f" % (cur_mse, cur_cs_ours, cur_cs_baseline))
    
    return cur_mse

def test(model, test_data_loader):
    model.eval()
    mse_total = 0
    cs_ours_total = 0
    cs_baseline_total = 0
    
    for i, (data, _, gt_time) in enumerate(test_data_loader):
        data = data.to(device)

        with torch.no_grad():
            _, mse, _, t_pred = model(data, 'test')
           
            mse_total += float(mse)
            mono_inc = np.array([np.arange(opt.video_length) for i in range(opt.test_batch_size)])
            cs_ours = cal_cs(t_pred.detach().cpu(), gt_time)
            cs_baseline = cal_cs(mono_inc, gt_time)
            cs_ours_total += float(cs_ours)
            cs_baseline_total += float(cs_baseline)

    cur_mse = mse_total/len(test_data_loader)
    cur_cs_ours = cs_ours_total/len(test_data_loader)
    cur_cs_baseline = cs_baseline_total/len(test_data_loader)
    logger.info("Test MSE: %f, Cos Sim(ours/baseline): %f, %f" % (cur_mse, cur_cs_ours, cur_cs_baseline))

    return cur_mse


best_mse = 10000
for e in range(args.num_epoch):
    train(model, train_dataloader, optimizer, e)
    mse = val(model, val_dataloader)
    mse_t = test(model, test_dataloader)
    if mse < best_mse:
        best_mse = mse
        torch.save(model.state_dict(), opt.exp + '/best_nnet.pth')
    logger.info("Best MSE: %f" % (best_mse))
    # torch.save(model.state_dict(), opt.exp + '/dnn_nnet.pth')


state_dic = torch.load(opt.exp + '/best_nnet.pth')
model.load_state_dict(state_dic)
test_mse = test(model, test_dataloader)
logger.info("Test MSE: %f" % (test_mse))


#!/usr/bin/env python
# coding: utf-8

from pickle import TRUE
from easydict import EasyDict
import os

opt = EasyDict()

# hyper parameters for models traning and testing
opt.lr = 3e-4
opt.epoch = 20
# weight for kl2 ptqt
opt.w_kl = 0.0000001
# k steps
opt.k = 10
# ode1 for zt cal
opt.ode1_step = 10
# ode2 for kl2 loss cal (0.1, 0.01, 0.001 ...)
opt.ode2_eps = 0.1
opt.seed = 112
opt.h_dim = 8

opt.train_batch_size = 50
opt.test_batch_size = 50

# load data path for training and testing 
opt.isLoad = True

###########################################################################################
opt.n_samples = {"train":5000, "test":100}
opt.n_total_tp = 10
opt.dataset_name = 'sin'



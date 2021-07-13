
#!/usr/bin/env python
# coding: utf-8

from easydict import EasyDict
import os

opt = EasyDict()

opt.gpunumber = 0

# hyper parameters for models traning and testing
opt.lr = 5e-4
opt.epoch = 50
opt.class_num = 10
# weight for kl ptqt
opt.w_kl = 0.00005
# k steps
opt.k = 10
# ode1 for zt cal
opt.ode1_step = 10
# ode2 for kl2 loss cal 
opt.ode2_eps = 0.1
opt.seed = 112
opt.h_dim = 128

opt.train_batch_size = 50
opt.val_batch_size = 50
opt.test_batch_size = 50

# load data path for training and testing 
opt.isLoad = False

# MNIST data path for training and testing
opt.train_image_path = './data/MNIST/processed/training.pt'
opt.test_image_path = './data/MNIST/processed/test.pt'

###########################################################################################
# parameters for dataset generation

# dataset size for training and testing
opt.num_mnist_sample_train = 5000
opt.num_mnist_sample_val = 1000
opt.num_mnist_sample_test = 1000
# the number of frames in one sequence(namely, one video)
opt.video_length = 6
opt.label_length = opt.video_length 

# size of each frame  
opt.canvas_size = (64, 64)
# parameters for initial angle
opt.angles = (0, 1)  # (mean, var)
# parameters for initial position
opt.positions = (18, 18), (2.0, 2.0)  # (mean_px, mean_py), (var_px, var_py)



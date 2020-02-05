# coding: utf-8

"""
PPAP : Pre-Processing And Prediction
"""

# Author: Taketo Kimura <mucun.wuxian@gmail.com>
# License: BSD 3 clause


# 
import os
import pickle
import cloudpickle
import sys
import cv2
import base64
import json
import requests
import gc
import Levenshtein
import time
import math

import numpy               as np
import matplotlib.pyplot   as plt
import japanize_matplotlib

from datetime              import datetime
from datetime              import timedelta

from sklearn.metrics       import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.datasets      import fetch_mldata

np.random.seed(0)

import warnings
warnings.filterwarnings('ignore')

# 
from . import image_utility as ppap_img_utl


# 
import torch
import torch.nn            as nn
import torch.nn.functional as F
from torch.utils.data      import Dataset, DataLoader, TensorDataset
from torch.optim           import Adam
from torch.optim.optimizer import Optimizer, required


# feed forward
def feed_forward(model, device, 
                 X_image=None, X_table=None, 
                 batch_size=128, train_mode=True, classify_mode=True):
    
    # training mode
    if (train_mode == True):
        # set training mode to model
        model.train() 
        
        # feed forward
        output = _feed_forward(model, device, X_image, X_table, batch_size)
    
    # predicting mode
    else:
        # set predicting mode to model
        model.eval() 

        # not need grad
        with torch.no_grad():
            # feed forward
            output = _feed_forward(model, device, X_image, X_table, batch_size)
    
    # 
    if (train_mode == False):
        # 
        output = output.to('cpu').numpy()
    # 
    if ((train_mode == False) & (classify_mode == True)):
        # 
        output = np.exp(output)

    # 
    return output


# sub function of feed forward 
def _feed_forward(model, device, X_image, X_table, batch_size=128):
    
    # adjust when image is one-record or gray
    if (X_image is not None):
        if (len(np.shape(X_image)) == 2):
            X_image = X_image[np.newaxis, :, :]
        if (len(np.shape(X_image)) == 3):
            X_image = X_image[:, np.newaxis, :, :]
    # adjust when table is one-record
    if (X_table is not None):
        if (len(np.shape(X_table)) == 1):
            X_table = X_table[np.newaxis, :]
    
    # 
    if ((X_image is not None) & (X_table is None)):
        # 
        X_image = torch.Tensor(X_image)
        # feed forward
        output  = _feed_forward_single(model, device, X_image, batch_size)
    # 
    elif ((X_image is None) & (X_table is not None)):
        # 
        X_table = torch.Tensor(X_table)
        # feed forward
        output  = _feed_forward_single(model, device, X_table, batch_size)
    # 
    elif ((X_image is not None) & (X_table is not None)):
        # 
        X_image = torch.Tensor(X_image)
        X_table = torch.Tensor(X_table)
        # feed forward
        output  = _feed_forward_double(model, device, X_image, X_table, batch_size)
    
    # 
    return output


# sub function of feed forward (simple NN)
def _feed_forward_single(model, device, X, batch_size=128):
    
    # define variable for output
    output = None
    
    # loop of building-up
    for x_i in np.arange(0, len(X), batch_size):
        # pick up building-up
        x          = X[x_i:(x_i + batch_size)]
        # print('np.shape(x) = (%d, %d, %d, %d)' % np.shape(x))
        # feed forward and tranport output to cpu
        x          = x.to(device) 
        output_tmp = model(x) # forward
        # adding
        if (output is None):
            output = output_tmp
        else:
            output = torch.cat((output, output_tmp), dim=0)
 
    # 
    return output
    

# sub function of feed forward (double multimodal NN)
def _feed_forward_double(model, device, X_1, X_2, batch_size=128):
    
    # define variable for output
    output = None
    
    # loop of building-up
    for x_i in np.arange(0, len(X_1), batch_size):
        # pick up building-up
        x_1        = X_1[x_i:(x_i + batch_size)]
        x_2        = X_2[x_i:(x_i + batch_size)]
        # feed forward and tranport output to cpu
        x_1        = x_1.to(device) 
        x_2        = x_2.to(device) 
        output_tmp = model(x_1, x_2) # forward
        # adding
        if (output is None):
            output = output_tmp
        else:
            output = torch.cat((output, output_tmp), dim=0)
 
    # 
    return output


# train by numpy input data
def train(model, device, 
          loss_func, optimizer, 
          X_image=None, X_table=None, y=None, 
          batch_size=128, 
          classify_mode=True):

    # # 
    # print(np.shape(X_image))
    # print(np.shape(X_table))
    # print(np.shape(y))

    # 
    output = feed_forward(model=model,  
                          device=device, 
                          X_image=X_image, 
                          X_table=X_table, 
                          batch_size=batch_size, 
                          train_mode=True, 
                          classify_mode=classify_mode)

    # ここ、ムズいな… 2クラスのsegmentationの場合…
    if ((len(np.shape(output)) == 4) & 
        (len(np.shape(y)) == 3)):
        y = y[:, np.newaxis, :, :]
    
    # set y to variable of pytorch
    if (classify_mode == True):
        y = torch.LongTensor(y).to(device)
    else:
        y = torch.Tensor(y).float().to(device)
    
    # print(np.shape(output)) # gradがcatできているっぽい
    # print(np.shape(y)) 

    # train 1 mini batch
    optimizer.zero_grad()       # init grad
    loss = loss_func(output, y) # calc loss
    loss.backward()             # back prop
    optimizer.step()            # update weight


# 
def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


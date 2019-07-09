# coding: utf-8

"""
PPAP : Pre-Processing And Prediction
"""

# Author: Taketo Kimura <taketo_kimura@micin.jp>
# License: BSD 3 clause


# 
import numpy               as np
import math
import pandas              as pd
import sys
import os
import os.path
import matplotlib.pyplot   as plt 
import matplotlib.cm       as cm
import matplotlib.gridspec as gridspec
import sklearn
import inspect
# import lime
# import lime.lime_tabular
# import termcolor

# 
from datetime                  import datetime
from datetime                  import timedelta
from copy                      import copy, deepcopy
# from dateutil.relativedelta    import relativedelta
# from IPython.core.display      import display
# from scipy.sparse.csgraph      import connected_components
from sklearn                   import metrics
from sklearn.model_selection   import train_test_split
from sklearn.decomposition     import PCA, KernelPCA as KPCA
from sklearn.manifold          import TSNE
from umap                      import UMAP
from sklearn.cluster           import KMeans

# 
from . import tabular_util as ppap_tab_ut


# 
def draw_prediction_scat(y_hat, 
                         y, 
                         data_type, 
                         font_size = 12, 
                         alpha     = None):

    # not create figure on this method

    # sort for visualization
    sort_idx   = np.argsort(y_hat)
    y_hat_sort = y_hat[sort_idx]
    y_sort     = y[sort_idx]
    y_hat_sort = np.concatenate([y_hat_sort[(y_sort == 0)], y_hat_sort[(y_sort == 1)]])
    y_sort     = np.concatenate([y_sort[(y_sort == 0)],     y_sort[(y_sort == 1)]])

    if (alpha is None):
        alpha  = np.max([0.005, np.min([0.1, (100 / len(y_sort))])])

    ############################################################
    plt.scatter(np.arange(len(y_sort)),     y_sort,     alpha=alpha, label='actual')
    plt.scatter(np.arange(len(y_hat_sort)), y_hat_sort, alpha=alpha, label='predict')
    plt.scatter([(sum(y_sort == 0) / 2), (sum(y_sort == 0) + (sum(y_sort == 1) / 2))], 
                [np.mean(y_hat_sort[(y_sort == 0)]), np.mean(y_hat_sort[(y_sort == 1)])], alpha=0.5, s=100, label='mean of predict')
    if (np.sum(y == 0) > np.sum(y == 1)):
        plt.legend(loc='upper left')
    else:
        plt.legend(loc='lower right')
    plt.ylabel('value of actual and predict\nat %s data' % data_type)
    plt.xlabel('data index (sorted for visibility)')
    plt.rcParams["font.size"] = font_size
    plt.grid(True)
    ############################################################

# 
def draw_prediction_dist(y_hat, 
                         y, 
                         data_type, 
                         font_size = 12):

    # not create figure on this merhod

    # make histogram
    bins_value      = np.linspace(0, 1, num=41)
    bins_value[-1] += 1e-20
    
    bins_mean       = np.zeros(len(bins_value) - 1)
    y_hat_hist_0    = np.zeros(len(bins_value) - 1)
    y_hat_hist_1    = np.zeros(len(bins_value) - 1)
    for bins_i in range(len(bins_mean)):
        bins_mean[bins_i]    = (bins_value[bins_i] + bins_value[bins_i+1]) / 2
        y_hat_hist_0[bins_i] = np.sum((bins_value[bins_i] <= y_hat[y == 0]) & (y_hat[y == 0] < bins_value[bins_i + 1]))
        y_hat_hist_1[bins_i] = np.sum((bins_value[bins_i] <= y_hat[y == 1]) & (y_hat[y == 1] < bins_value[bins_i + 1]))
    
    ############################################################
    plt.bar(bins_mean, (y_hat_hist_0 / np.sum(y_hat_hist_0)), alpha=0.5, label='y == 0', width=((bins_mean[1] - bins_mean[0]) * 0.9))
    plt.bar(bins_mean, (y_hat_hist_1 / np.sum(y_hat_hist_1)), alpha=0.5, label='y == 1', width=((bins_mean[1] - bins_mean[0]) * 0.9))
    plt.legend(loc='upper center')
    plt.ylabel('frequency ratio of %s data' % data_type)
    plt.xlabel('y hat value')
    plt.rcParams["font.size"] = font_size
    plt.grid(True)
    ############################################################

# y_hat 1 dim -> (data index)
# y_hat 2 dim -> (data index, cross validation index)
# y_hat 3 dim -> (data index, cross validation index, model index)
def draw_roc(y_hat, 
             y, 
             model_name, 
             data_type, 
             draw_cv_ratio = 1.0, 
             draw_legend   = True, 
             font_size     = 12):

    # not create figure on this method

    # 
    if (type(model_name) == str):
        model_name = [model_name]
    if (type(y_hat) == np.ndarray):
        y_hat = [y_hat]
        y     = [y]
    if (type(y_hat[0]) == np.ndarray):
        y_hat = [y_hat]
        y     = [y]

    # 
    model_num = np.shape(y_hat)[0]
    cv_num    = np.shape(y_hat)[1]
    auc       = np.zeros([model_num, cv_num])

    draw_cv   = np.random.permutation(np.arange(cv_num))[:int(np.ceil(cv_num * draw_cv_ratio))]

    for model_i in range(model_num):
        
        legend_unset = True
        color_tmp    = cm.hsv(model_i/(model_num * 1.1))

        for cv_i in range(cv_num):

            # calc FPR, TPR and threshold
            fpr, tpr, threshold = metrics.roc_curve(y[model_i][cv_i], y_hat[model_i][cv_i])

            # calc AUC or ROC curve
            auc[model_i, cv_i] = metrics.auc(fpr, tpr)

            if (cv_i in draw_cv):
                ################################################
                if (legend_unset):
                    plt.plot(fpr, tpr, label=('%s' % model_name[model_i]), color=color_tmp, alpha=(0.8/len(draw_cv)), linewidth=3)
                    legend_unset = False
                else:
                    plt.plot(fpr, tpr, color=color_tmp, alpha=(0.8/len(draw_cv)), linewidth=3)
                # plt.scatter(fpr, tpr, color=color_tmp, alpha=(0.8/len(draw_cv)))
                ################################################

        ########################################################
        if (draw_legend):
            plt.legend(loc='lower right', fontsize='small')
        plt.title('ROC curve on %s data' % data_type)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.rcParams["font.size"] = font_size
        plt.grid(True)
        ########################################################

    return auc

# y_hat 1 dim -> (data index)
# y_hat 2 dim -> (data index, cross validation index)
# y_hat 3 dim -> (data index, cross validation index, model index)
def draw_pr(y_hat, 
            y, 
            model_name, 
            data_type, 
            draw_cv_ratio = 1.0, 
            draw_legend   = True, 
            font_size     = 12):

    # not create figure on this method

    # 
    if (type(model_name) == str):
        model_name = [model_name]
    if (type(y_hat) == np.ndarray):
        y_hat = [y_hat]
        y     = [y]
    if (type(y_hat[0]) == np.ndarray):
        y_hat = [y_hat]
        y     = [y]

    # 
    model_num = np.shape(y_hat)[0]
    cv_num    = np.shape(y_hat)[1]
    auc       = np.zeros([model_num, cv_num])

    draw_cv   = np.random.permutation(np.arange(cv_num))[:int(np.ceil(cv_num * draw_cv_ratio))]

    for model_i in range(model_num):
        
        legend_unset = True
        color_tmp    = cm.hsv(model_i/(model_num * 1.1))

        for cv_i in range(np.shape(y_hat)[1]):

            # calc Precision, Recall and threshold
            precision, recall, threshold = metrics.precision_recall_curve(y[model_i][cv_i], y_hat[model_i][cv_i])

            # calc AUC or ROC curve
            auc[model_i, cv_i] = metrics.auc(recall, precision)

            if (cv_i in draw_cv):
                ################################################
                if (legend_unset):
                    plt.plot(recall, precision, label=('%s' % model_name[model_i]), color=color_tmp, alpha=(0.8/len(draw_cv)), linewidth=3)
                    legend_unset = False
                else:
                    plt.plot(recall, precision, color=color_tmp, alpha=(0.8/len(draw_cv)), linewidth=3)
                # plt.scatter(recall, precision, color=color_tmp, alpha=(0.8/len(draw_cv)))
                ################################################

        ########################################################
        if (draw_legend):
            plt.legend(loc='lower left', fontsize='small')
        plt.title('Precision Recall curve on %s data' % data_type)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.ylim(-0.05, 1.05)
        plt.rcParams["font.size"] = font_size
        plt.grid(True)
        ########################################################

    return auc

# auc 1 dim -> (cross validation index)
# auc 2 dim -> (cross validation index, model index)
def draw_auc(auc, 
             model_name, 
             data_type, 
             watch_rank = 20,
             watch_mode = 'TOP',  
             font_size  = 12):
    
    # not create figure on this method

    # adjust
    if (len(np.shape(auc)) == 1):
        auc = auc[:, np.newaxis]
    if (type(model_name) == str):
        model_name = [model_name]
    if (type(model_name) == list):
        model_name = np.array(model_name)
    
    # 
    model_num      = np.shape(auc)[0]
    cv_num         = np.shape(auc)[1]

    # adjust
    if (watch_rank > model_num):
        watch_rank = model_num

    #
    if (cv_num > 1):
        auc_mean  = np.mean(auc, axis=1)
        auc_std   = np.std(auc, axis=1)
        auc_p1sig = auc_mean + auc_std
        auc_m1sig = auc_mean - auc_std
        auc_max   = np.max(auc, axis=1)
        auc_min   = np.min(auc, axis=1)
    else:
        auc_mean  = auc[0]
        auc_std   = auc[0] * 0
        auc_p1sig = auc[0]
        auc_m1sig = auc[0]
        auc_max   = auc[0]
        auc_min   = auc[0]
    #
    if (model_num > 1):
        idx_sort_auc = np.argsort(-auc_mean)
        auc_mean     = auc_mean[idx_sort_auc[:watch_rank]]
        auc_std      = auc_std[idx_sort_auc[:watch_rank]]
        auc_max      = auc_max[idx_sort_auc[:watch_rank]]
        auc_min      = auc_min[idx_sort_auc[:watch_rank]]
        model_name_  = model_name[idx_sort_auc[:watch_rank]]
    else:
        idx_sort_auc = np.array([0])
        model_name_  = model_name
    
    ############################################################
    if (watch_mode == 'TOP'):
        model_name_tmp = []
        for model_i in range(watch_rank):
            color_tmp = cm.hsv(idx_sort_auc[model_i]/(model_num * 1.1))
            plt.bar(model_i, auc_mean[model_i], color=color_tmp, alpha=0.5)
            plt.plot([model_i, model_i, model_i, model_i], 
                    [auc_max[model_i], auc_p1sig[model_i], auc_m1sig[model_i], auc_min[model_i]], color=color_tmp, alpha=0.9)
            plt.scatter([model_i, model_i, model_i, model_i], 
                        [auc_max[model_i], auc_p1sig[model_i], auc_m1sig[model_i], auc_min[model_i]], color=color_tmp, alpha=0.9)
            model_name_tmp.append('%s (AUC m=%.2f, s=%.2f, t=%.2f, b=%.2f)' % (model_name_[model_i], auc_mean[model_i], auc_std[model_i], auc_max[model_i], auc_min[model_i]))
        plt.xticks(np.arange(watch_rank))
        plt.xlim(-0.9, (watch_rank -0.1))
        ax = plt.gca()
        ax.set_xticklabels(model_name_tmp, rotation=90)
        plt.xlabel('model index')
        plt.ylabel('AUC value of %s' % data_type)
        plt.ylim(-0.05, 1.05)
        plt.rcParams["font.size"] = font_size
        plt.grid(True)
    else:
        for model_i in range(model_num):
            color_tmp = cm.hsv(model_i/(model_num * 1.1))
            plt.bar(model_i, auc_mean[model_i], color=color_tmp, alpha=0.5)
            plt.plot([model_i, model_i, model_i, model_i], 
                     [auc_max[model_i], auc_p1sig[model_i], auc_m1sig[model_i], auc_min[model_i]], color=color_tmp, alpha=0.9)
            plt.scatter([model_i, model_i, model_i, model_i], 
                        [auc_max[model_i], auc_p1sig[model_i], auc_m1sig[model_i], auc_min[model_i]], color=color_tmp, alpha=0.9)
        plt.xlim(-0.9, (model_num - 0.1))
        plt.xlabel('model index')
        plt.ylabel('AUC value of %s' % data_type)
        plt.ylim(-0.05, 1.05)
        plt.rcParams["font.size"] = font_size
        plt.grid(True)
    ############################################################

    return idx_sort_auc

# 
def train_and_predict(X_train, 
                      y_train, 
                      X_test, 
                      y_test, 
                      model, 
                      sample_balance = False):

    # random seed adjust
    np.random.seed(0)

    # model copy and random seed adjust
    model_tmp = deepcopy(model)
    if hasattr(model_tmp, 'random_state'):
        model_tmp.random_state = 0
    
    # 
    if (sample_balance):
        X_train_up    = copy(X_train)
        y_train_up    = copy(y_train)
        neg_train_num = np.sum(y_train == 0)
        pos_train_num = np.sum(y_train == 1)
        if (neg_train_num > pos_train_num):
            copy_num  = neg_train_num - pos_train_num
            while (copy_num > 0):
                X_train_pos = X_train_up[(y_train_up == 1), :]
                X_train_pos = np.random.permutation(X_train_pos)
                if (copy_num < len(X_train_pos)):
                    X_train_pos = X_train_pos[:copy_num]
                y_train_pos = np.ones([len(X_train_pos), 1], dtype='float')
                X_train_up  = np.concatenate([X_train_up, X_train_pos], axis=0)
                y_train_up  = np.concatenate([y_train_up[:, np.newaxis], y_train_pos], axis=0).reshape(-1)
                copy_num   -= len(X_train_pos)
        else:
            copy_num  = pos_train_num - neg_train_num
            while (copy_num > 0):
                X_train_neg = X_train_up[(y_train_up == 0), :]
                X_train_neg = np.random.permutation(X_train_neg)
                if (copy_num < len(X_train_neg)):
                    X_train_neg = X_train_neg[:copy_num]
                y_train_neg = np.zeros([len(X_train_neg), 1], dtype='float')
                X_train_up  = np.concatenate([X_train_up, X_train_neg], axis=0)
                y_train_up  = np.concatenate([y_train_up[:, np.newaxis], y_train_neg], axis=0).reshape(-1)
                copy_num   -= len(X_train_neg)
    else:
        # 
        X_train_up = np.nan
        y_train_up = np.nan

    #
    if ('verbose' in inspect.getargspec(model_tmp.fit)):
        model_tmp.fit(X_train, y_train, verbose=False)
    else:
        model_tmp.fit(X_train, y_train)
    # 
    y_train_hat = model_tmp.predict_proba(X_train).T[1]
    # 
    y_test_hat  = model_tmp.predict_proba(X_test).T[1]

    return (model_tmp, 
            y_train_hat, 
            y_test_hat, 
            X_train_up, 
            y_train_up)

def train_predict_and_measure(X_train, 
                              y_train, 
                              X_test, 
                              y_test, 
                              model, 
                              column_name    = None,
                              sample_balance = False, 
                              alpha          = None):

    # random seed adjust
    np.random.seed(0)
    
    # model copy and random seed adjust
    model_tmp = deepcopy(model)
    if hasattr(model_tmp, 'random_state'):
        model_tmp.random_state = 0
    
    # 
    X_dim = np.shape(X_train)[1]

    # adjust    
    if (column_name is None):
        column_name = [('column_%d' % i) for i in range(X_dim)]
    if (type(column_name) == list):
        column_name = np.array(column_name)

    # exec
    (model_tmp, 
     y_train_hat, 
     y_test_hat, 
     X_train_up, 
     y_train_up) = train_and_predict(X_train        = X_train,
                                     y_train        = y_train,
                                     X_test         = X_test,
                                     y_test         = y_test, 
                                     model          = model_tmp, 
                                     sample_balance = sample_balance)

    ############################################################
    fig = plt.figure(figsize=(12,8),dpi=100)
    # 
    plt.subplot(2, 2, 1)
    if (sample_balance):
        draw_prediction_scat(y_hat     = y_train_hat, 
                             y         = y_train_up, 
                             data_type = 'train', 
                             alpha     = alpha)
    else:
        draw_prediction_scat(y_hat     = y_train_hat, 
                             y         = y_train, 
                             data_type = 'train', 
                             alpha     = alpha)
    # 
    plt.subplot(2, 2, 2)
    draw_prediction_scat(y_hat     = y_test_hat, 
                         y         = y_test, 
                         data_type = 'test', 
                         alpha     = alpha)
    # 
    plt.subplot(2, 2, 3)
    if (sample_balance):
        draw_prediction_dist(y_hat     = y_train_hat, 
                             y         = y_train_up, 
                             data_type = 'train', 
                             alpha     = alpha)
    else:
        draw_prediction_dist(y_hat     = y_train_hat, 
                             y         = y_train, 
                             data_type = 'train')
    # 
    plt.subplot(2, 2, 4)
    draw_prediction_dist(y_hat     = y_test_hat, 
                         y         = y_test, 
                         data_type = 'test')
    # 
    fig = plt.figure(figsize=(12,4),dpi=100)
    #
    if hasattr(model_tmp, 'coef_'):
        importance_tmp = model_tmp.coef_[0].astype('float')
    else:
        importance_tmp = model_tmp.feature_importances_.astype('float')
    # 
    plt.subplot(1, 1, 1)
    ppap_tab_ut.draw_importance(importance  = importance_tmp, 
                                column_name = column_name)

    # plot
    fig = plt.figure(figsize=(12,8),dpi=100)

    ax = plt.subplot(2, 2, 1)
    if (sample_balance):
        auc_train = draw_roc(y_hat      = y_train_hat, 
                             y          = y_train_up, 
                             model_name = 'lightGBM', 
                             data_type  = 'train')
    else:
        auc_train = draw_roc(y_hat      = y_train_hat, 
                             y          = y_train, 
                             model_name = 'lightGBM', 
                             data_type  = 'train')
    ax = plt.subplot(2, 2, 2)
    auc_test = draw_roc(y_hat      = y_test_hat, 
                        y          = y_test, 
                        model_name = 'lightGBM', 
                        data_type  = 'test')
    ax = plt.subplot(2, 2, 3)
    draw_auc(auc        = auc_train, 
             model_name = 'lightGBM', 
             data_type  = 'train')
    ax = plt.subplot(2, 2, 4)
    draw_auc(auc        = auc_test, 
             model_name = 'lightGBM', 
             data_type  = 'test')

    # plot
    fig = plt.figure(figsize=(12,4),dpi=100)

    ax = plt.subplot(1, 2, 1)
    if (sample_balance):
        auc_train = draw_pr(y_hat      = y_train_hat, 
                            y          = y_train_up, 
                            model_name = 'lightGBM', 
                            data_type  = 'train')
    else:
        auc_train = draw_pr(y_hat      = y_train_hat, 
                            y          = y_train, 
                            model_name = 'lightGBM', 
                            data_type  = 'train')
    ax = plt.subplot(1, 2, 2)
    auc_test = draw_pr(y_hat      = y_test_hat, 
                       y          = y_test, 
                       model_name = 'lightGBM', 
                       data_type  = 'test')
    # 
    plt.show()
    ############################################################

    return (model_tmp, 
            y_train_hat, 
            y_test_hat, 
            X_train_up, 
            y_train_up)

# 
def cv_random(X, 
              y, 
              model, 
              model_name, 
              sample_balance       = False, 
              column_name          = None, 
              cv_train_ratio       = 0.5, 
              cv_num               = 10,
              draw_auc_flg         = True,  
              draw_cv_ratio        = 0.1, 
              draw_importance_rank = 30):

    # random seed adjust
    np.random.seed(0)

    # 
    X_dim = np.shape(X)[1]

    # 
    draw_cv = np.random.permutation(np.arange(cv_num))[:int(np.ceil(cv_num * draw_cv_ratio))]
    
    # 
    if (type(model_name) == str):
        model_name = [model_name]
    if (type(model) != list):
        model = [model]
    model_num = len(model)
    if (column_name is None):
        column_name = [('column_%d' % i) for i in range(X_dim)]
    if (type(column_name) == list):
        column_name = np.array(column_name)
    
    # 
    y_train_hat_stock = []
    y_train_stock     = []
    y_test_hat_stock  = []
    y_test_stock      = []
    importance_stock  = np.zeros([X_dim, cv_num, model_num])

    ############################################################
    # display processing progress 
    process_num   = cv_num # set number of process
    process_break = np.round(np.linspace(1, process_num, 50)) 
    ############################################################
    
    # 
    for model_i in range(len(model)):
        
        # model copy and random seed adjust
        model_tmp = deepcopy(model[model_i])
        if hasattr(model_tmp, 'random_state'):
            model_tmp.random_state = 0

        ########################################################
        process_i        = 0  
        str_now          = datetime.now()
        print('train on cv (model:%s, feature dim:%d) [start time is %s]' % (model_name[model_i], X_dim, str_now)) 
        print('--------------------------------------------------')
        print('START                                          END') 
        print('----+----1----+----2----+----3----+----4----+----5') 
        ########################################################
        
        # 
        y_train_hat_stock_of_cv = []
        y_train_stock_of_cv     = []
        y_test_hat_stock_of_cv  = []
        y_test_stock_of_cv      = []
        # 
        auc_train = 0
        auc_test  = 0

        # 
        for cv_i in range(cv_num):

            ####################################################
            # update processing progress
            process_i = process_i + 1   
            if (sum(process_break == process_i) > 0):
                for print_i in range(sum(process_break == process_i)): 
                    print('*', end='', flush=True)                              
            ####################################################

            # 
            (X_train, X_test, 
             y_train, y_test) = train_test_split(X, y, 
                                                 test_size    = cv_train_ratio, 
                                                 random_state = cv_i)
            # 
            (model_tmp, 
             y_train_hat, 
             y_test_hat, 
             _, 
             y_train_up) = train_and_predict(X_train        = X_train,
                                             y_train        = y_train,
                                             X_test         = X_test,
                                             y_test         = y_test, 
                                             model          = model_tmp, 
                                             sample_balance = sample_balance) 
            
            # 
            y_train_hat_stock_of_cv.append(y_train_hat)
            if (sample_balance):
                y_train_stock_of_cv.append(y_train_up)
            else:
                y_train_stock_of_cv.append(y_train)
            y_test_hat_stock_of_cv.append(y_test_hat)
            y_test_stock_of_cv.append(y_test)
            # 
            if hasattr(model_tmp, 'coef_'):
                importance_tmp = model_tmp.coef_[0].astype('float')
            else:
                importance_tmp = model_tmp.feature_importances_.astype('float')
            importance_stock[:, cv_i, model_i] = importance_tmp

            # calc AUC
            if (sample_balance):
                fpr, tpr, _ = metrics.roc_curve(y_train_up, y_train_hat)
            else:
                fpr, tpr, _ = metrics.roc_curve(y_train, y_train_hat)
            auc_tmp     = metrics.auc(fpr, tpr)
            auc_train  += (auc_tmp / cv_num)
            fpr, tpr, _ = metrics.roc_curve(y_test, y_test_hat)
            auc_tmp     = metrics.auc(fpr, tpr)
            auc_test   += (auc_tmp / cv_num)

            ####################################################
            if (cv_i in draw_cv):
                # 
                fig = plt.figure(figsize=(12,8),dpi=100)
                # 
                plt.subplot(2, 2, 1)
                if (sample_balance):
                    draw_prediction_scat(y_hat     = y_train_hat, 
                                         y         = y_train_up, 
                                         data_type = 'train', 
                                         alpha     = alpha)
                else:
                    draw_prediction_scat(y_hat     = y_train_hat, 
                                         y         = y_train, 
                                         data_type = 'train', 
                                         alpha     = alpha)
                # 
                plt.subplot(2, 2, 2)
                draw_prediction_scat(y_hat     = y_test_hat, 
                                     y         = y_test, 
                                     data_type = 'test', 
                                     alpha     = alpha)
                # 
                plt.subplot(2, 2, 3)
                if (sample_balance):
                    draw_prediction_dist(y_hat     = y_train_hat, 
                                         y         = y_train_up, 
                                         data_type = 'train')
                else:
                    draw_prediction_dist(y_hat     = y_train_hat, 
                                         y         = y_train, 
                                         data_type = 'train')
                # 
                plt.subplot(2, 2, 4)
                draw_prediction_dist(y_hat     = y_test_hat, 
                                     y         = y_test, 
                                     data_type = 'test')
                # 
                fig = plt.figure(figsize=(12,4),dpi=100)
                # 
                plt.subplot(1, 1, 1)
                ppap_tab_ut.draw_importance(importance  = importance_tmp, 
                                            column_name = column_name, 
                                            watch_rank  = draw_importance_rank)
            ####################################################

        # 
        y_train_hat_stock.append(y_train_hat_stock_of_cv)
        y_train_stock.append(y_train_stock_of_cv)
        y_test_hat_stock.append(y_test_hat_stock_of_cv)
        y_test_stock.append(y_test_stock_of_cv)

        ########################################################
        print('')
        str_now = datetime.now()  
        print('AUC mean on train = %.2f, test = %.2f [end time is %s]' % (auc_train, auc_test, str_now)) 
        print('\n')
        plt.show()
        ########################################################
    
    ############################################################
    if (draw_auc_flg):
        # 
        print('\n\nsummary result')
        # 
        fig = plt.figure(figsize=(12,12),dpi=100)
        # 
        plt.subplot(3, 2, 1)
        auc_train = draw_roc(y_hat      = y_train_hat_stock, 
                             y          = y_train_stock, 
                             model_name = model_name, 
                             data_type  = 'train')
        # 
        plt.subplot(3, 2, 2)
        auc_test = draw_roc(y_hat      = y_test_hat_stock, 
                            y          = y_test_stock, 
                            model_name = model_name, 
                            data_type  = 'test')
        # 
        plt.subplot(3, 2, 3)
        draw_auc(auc        = auc_train, 
                 model_name = model_name, 
                 watch_mode = 'ALL',
                 data_type  = 'train')
        # 
        plt.subplot(3, 2, 4)
        draw_auc(auc        = auc_test, 
                 model_name = model_name, 
                 watch_mode = 'ALL',
                 data_type  = 'test')
        # 
        plt.subplot(3, 2, 5)
        draw_auc(auc        = auc_train, 
                 model_name = model_name, 
                 data_type  = 'train')
        # 
        plt.subplot(3, 2, 6)
        draw_auc(auc        = auc_test, 
                 model_name = model_name, 
                 data_type  = 'test')
        # 
        fig = plt.figure(figsize=(12,4),dpi=100)
        # 
        plt.subplot(1, 2, 1)
        auc_train = draw_pr(y_hat      = y_train_hat_stock, 
                            y          = y_train_stock, 
                            model_name = model_name, 
                            data_type  = 'train')
        # 
        plt.subplot(1, 2, 2)
        auc_test = draw_pr(y_hat      = y_test_hat_stock, 
                           y          = y_test_stock, 
                           model_name = model_name, 
                           data_type  = 'test')
        #
        plt.show()
    ############################################################

    return (y_train_hat_stock, 
            y_train_stock, 
            y_test_hat_stock, 
            y_test_stock, 
            importance_stock)

# 
def rfe_and_cv(X, 
               y, 
               model, 
               model_name,
               sample_balance       = False, 
               column_name          = None, 
               rfe_step             = None,  
               cv_train_ratio       = 0.5, 
               cv_num               = 10, 
               draw_cv_ratio        = 0.1, 
               draw_importance_rank = 30):

    # random seed adjust
    np.random.seed(0)

    # 
    X_dim = np.shape(X)[1]

    # 
    if (type(model_name) == str):
        model_name = [model_name]
    if (type(model) != list):
        model = [model]
    model_num = len(model)
    if (rfe_step is None):
        rfe_step = [X_dim]
    if (type(rfe_step) != list):
        rfe_step = [rfe_step]
    rfe_step     = np.unique([X_dim] + rfe_step)
    sort_idx     = np.argsort(-rfe_step)
    rfe_step     = rfe_step[sort_idx]
    rfe_step     = np.array(rfe_step)
    rfe_step     = rfe_step[rfe_step <= X_dim]
    rfe_step_num = len(rfe_step)
    # 
    train_num    = int(np.floor(len(y) * cv_train_ratio))
    test_num     = int(np.ceil(len(y) * cv_train_ratio))
    # 
    y_train_hat_stock = []
    y_train_stock     = []
    y_test_hat_stock  = []
    y_test_stock      = []
    model_name_stock  = []
    importance_stock  = np.zeros([X_dim, cv_num, model_num, rfe_step_num])
    remain_idx        = np.zeros([X_dim, model_num, rfe_step_num], dtype='bool')

    #
    for model_i in range(len(model)):
        # 
        for rfe_step_i in range(len(rfe_step)):
            # 
            y_train_hat_stock_of_cv = []
            y_train_stock_of_cv     = []
            y_test_hat_stock_of_cv  = []
            y_test_stock_of_cv      = []
            #
            if (rfe_step_i == 0):
                remain_idx[:, model_i, rfe_step_i] = True
            else:
                importance_tmp = importance_stock[:, :, model_i, (rfe_step_i - 1)]
                importance_tmp = np.sum(importance_tmp, axis=1)
                sort_idx       = np.argsort(-importance_tmp)
                remain_idx[sort_idx[:rfe_step[rfe_step_i]], model_i, rfe_step_i] = True
            # 
            X_slim = X[:, remain_idx[:, model_i, rfe_step_i]]

            # model copy and random seed adjust
            model_tmp = deepcopy(model[model_i])
            if hasattr(model_tmp, 'random_state'):
                model_tmp.random_state = 0
            #
            model_name_tmp = ('[m:%d,r:%d] %s_RFE%d' % (model_i, rfe_step_i, model_name[model_i], rfe_step[rfe_step_i]))
            
            #  
            (y_train_hat_stock_tmp, 
             y_train_stock_tmp, 
             y_test_hat_stock_tmp, 
             y_test_stock_tmp, 
             importance_stock_tmp) = cv_random(X                    = X_slim, 
                                               y                    = y, 
                                               model                = model_tmp, 
                                               model_name           = model_name_tmp, 
                                               sample_balance       = sample_balance, 
                                               column_name          = column_name, 
                                               cv_train_ratio       = cv_train_ratio, 
                                               cv_num               = cv_num,  
                                               draw_auc_flg         = False, 
                                               draw_cv_ratio        = draw_cv_ratio, 
                                               draw_importance_rank = draw_importance_rank)
            # 
            for cv_i in range(cv_num):
                # 
                y_train_hat_stock_of_cv.append(y_train_hat_stock_tmp[0][cv_i])
                y_train_stock_of_cv.append(y_train_stock_tmp[0][cv_i])
                y_test_hat_stock_of_cv.append(y_test_hat_stock_tmp[0][cv_i])
                y_test_stock_of_cv.append(y_test_stock_tmp[0][cv_i])
                # 
                remain_idx_tmp                                              = remain_idx[:, model_i, rfe_step_i]
                importance_stock[remain_idx_tmp, cv_i, model_i, rfe_step_i] = importance_stock_tmp[:, cv_i, 0]

            # 
            y_train_hat_stock.append(y_train_hat_stock_of_cv)
            y_train_stock.append(y_train_stock_of_cv)
            y_test_hat_stock.append(y_test_hat_stock_of_cv)
            y_test_stock.append(y_test_stock_of_cv)
            # 
            model_name_stock.append(model_name_tmp)

    ############################################################
    # 
    print('\n\nsummary result')
    # 
    fig = plt.figure(figsize=(12,8),dpi=100)
    # 
    plt.subplot(2, 2, 1)
    auc_train = draw_roc(y_hat       = y_train_hat_stock, 
                         y           = y_train_stock, 
                         model_name  = model_name_stock,
                         draw_legend = False,  
                         data_type   = 'train')
    # 
    plt.subplot(2, 2, 2)
    auc_test = draw_roc(y_hat       = y_test_hat_stock, 
                        y           = y_test_stock, 
                        model_name  = model_name_stock, 
                        draw_legend = False,  
                        data_type   = 'test')
    # 
    plt.subplot(2, 2, 3)
    idx_sort_auc_train = draw_auc(auc        = auc_train, 
                                  model_name = model_name_stock, 
                                  data_type  = 'train')
    # 
    plt.subplot(2, 2, 4)
    idx_sort_auc_test = draw_auc(auc        = auc_test, 
                                 model_name = model_name_stock, 
                                 data_type  = 'test')
    # 
    fig = plt.figure(figsize=(12,4),dpi=100)
    # 
    plt.subplot(1, 2, 1)
    auc_train = draw_pr(y_hat       = y_train_hat_stock, 
                        y           = y_train_stock, 
                        model_name  = model_name_stock, 
                        draw_legend = False,  
                        data_type   = 'train')
    # 
    plt.subplot(1, 2, 2)
    auc_test = draw_pr(y_hat       = y_test_hat_stock, 
                       y           = y_test_stock, 
                       model_name  = model_name_stock, 
                       draw_legend = False,  
                       data_type   = 'test')
    # 
    plt.show()
    ############################################################

    return (importance_stock, 
            rfe_step, 
            remain_idx, 
            model_name_stock, 
            idx_sort_auc_test) 


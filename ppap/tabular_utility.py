# coding: utf-8

"""
PPAP : Pre-Processing And Prediction
"""

# Author: Taketo Kimura <taketo_kimura@micin.jp>
# License: BSD 3 clause


import numpy               as np
import math
import pandas              as pd
import sys
import os
import os.path
import matplotlib.pyplot   as plt 
import matplotlib.cm       as cm
import matplotlib.gridspec as gridspec
# import lime
# import lime.lime_tabular

from datetime                        import datetime
from datetime                        import timedelta
from copy                            import copy
from sklearn.decomposition           import PCA
from sklearn.cluster                 import KMeans
from sklearn.feature_extraction.text import CountVectorizer

STR_NAN = '(nan)' 

MCLASS_DEFINE = 50

# 
def check_nan_ratio_vert(X, 
                         column_name = None, 
                         watch_rank  = 20, 
                         font_size   = 12):
    
    # escape
    X_           = copy(X)
    column_name_ = copy(column_name)

    # if X is pandas dataframe then convert to numpy array
    if (str(type(X)).find('DataFrame') > -1):
        X_ = X_.values

    # adjust
    if (len(np.shape(X_)) == 1):
        X_ = X_[:, np.newaxis]

    # get basic info
    X_num = np.shape(X_)[0]
    X_dim = np.shape(X_)[1]
    
    # adjust
    if (column_name is None):
        if (str(type(X)).find('DataFrame') > -1):
            column_name_ = X.columns
        else:
            column_name_ = [('column_%d' % i) for i in range(X_dim)]
    if (type(column_name_) == str):
        column_name_ = [column_name_]
    if (type(column_name_) == list):
        column_name_ = np.array(column_name_)
    if (watch_rank > X_dim):
        watch_rank = X_dim
    
    # calc nan ratio
    nan_ratio = np.sum(((X_ == X_) == False), axis=0) / X_num
    # nan_ratio = nan_ratio + np.sum((X_ == 'nan'), axis=0) / X_num

    # print including nan column
    print('\ninclude nan column is ...')
    print('\n------------------------------------------------------------')
    for column_i in range(X_dim):
        if (nan_ratio[column_i] > 0):
            print('  - %s -> %.5f' % (column_name_[column_i], nan_ratio[column_i]))

    ############################################################
    if (np.sum(nan_ratio) > 0):
        # make figure
        plt.figure(figsize=(12,8),dpi=100)
        # 
        sort_idx = np.argsort(-nan_ratio)
        plt.subplot(2, 1, 1)
        ax = plt.gca()
        plt.bar(np.arange(len(nan_ratio)), nan_ratio[sort_idx])
        plt.ylim(-0.05, 1.05)
        plt.xlim(-0.9, (len(nan_ratio) -0.1))
        plt.title('nan ratio of X of all columns')
        plt.ylabel('nan exist ratio')
        plt.xlabel('column index')
        plt.rcParams["font.size"] = font_size
        plt.grid(True)
        # 
        sort_idx = sort_idx[:watch_rank]
        plt.subplot(2, 2, 3)
        ax = plt.gca()
        plt.bar(np.arange(watch_rank), nan_ratio[sort_idx])
        plt.ylim(-0.05, 1.05)
        plt.xlim(-0.9, (watch_rank -0.1))
        plt.xticks(np.arange(watch_rank))
        ax.set_xticklabels(column_name_[sort_idx], rotation=90, fontsize='small')
        plt.title('nan ratio of X (TOP%d)' % watch_rank)
        plt.ylabel('nan exist ratio')
        plt.rcParams["font.size"] = font_size
        plt.grid(True)
        # 
        sort_idx = np.argsort(nan_ratio)[:watch_rank]
        for i in range(len(sort_idx) // 2):
            sort_idx[i], sort_idx[-1-i] = sort_idx[-1-i], sort_idx[i]
        plt.subplot(2, 2, 4)
        ax = plt.gca()
        plt.bar(np.arange(watch_rank), nan_ratio[sort_idx])
        plt.ylim(-0.05, 1.05)
        plt.xlim(-0.9, (watch_rank -0.1))
        plt.xticks(np.arange(watch_rank))
        ax.set_xticklabels(column_name_[sort_idx], rotation=90, fontsize='small')
        plt.title('nan ratio of X (BOTTOM%d)' % watch_rank)
        plt.ylabel('nan exist ratio')
        plt.rcParams["font.size"] = font_size
        plt.grid(True)
        # 
        plt.show()
    else:
        print('nan is not exist. check OK!')
    ############################################################

    return nan_ratio

# 
def check_nan_ratio_horz(X, 
                         watch_rank  = 20, 
                         font_size   = 12):
    
    # escape
    X_ = copy(X)

    # if X is pandas dataframe then convert to numpy array
    if (str(type(X)).find('DataFrame') > -1):
        X_ = X_.values

    # adjust
    if (len(np.shape(X_)) == 1):
        X_ = X_[:, np.newaxis]

    # get basic info
    X_num = np.shape(X_)[0]
    X_dim = np.shape(X_)[1]
    
    # calc nan ratio
    nan_ratio = np.sum(((X_ == X_) == False), axis=1) / X_dim
    # nan_ratio = nan_ratio + np.sum((X_ == 'nan'), axis=1) / X_dim

    ############################################################
    if (np.sum(nan_ratio) > 0):
        # make figure
        plt.figure(figsize=(12,8),dpi=100)
        # 
        max_view_num = 20000
        if (len(nan_ratio) > max_view_num):
            nan_ratio_   = np.random.permutation(nan_ratio)[:max_view_num]
            sampling_flg = True
        else:
            nan_ratio_   = nan_ratio
            sampling_flg = False
        #
        sort_idx = np.argsort(-nan_ratio_)
        plt.subplot(2, 1, 1)
        ax = plt.gca()
        plt.bar(np.arange(len(nan_ratio_)), nan_ratio_[sort_idx])
        plt.ylim(-0.05, 1.05)
        plt.xlim(-0.9, (len(nan_ratio_) -0.1))
        if (sampling_flg):
            plt.title('nan ratio of X of all columns (view sampling %d rec)' % max_view_num)
        else:
            plt.title('nan ratio of X of all columns')
        plt.ylabel('nan exist ratio')
        plt.xlabel('column index')
        plt.rcParams["font.size"] = font_size
        plt.grid(True)
        # 
        sort_idx = sort_idx[:watch_rank]
        plt.subplot(2, 2, 3)
        ax = plt.gca()
        plt.bar(np.arange(watch_rank), nan_ratio_[sort_idx])
        plt.ylim(-0.05, 1.05)
        plt.xlim(-0.9, (watch_rank -0.1))
        plt.title('nan ratio of X (TOP%d)' % watch_rank)
        plt.ylabel('nan exist ratio')
        plt.rcParams["font.size"] = font_size
        plt.grid(True)
        # 
        sort_idx = np.argsort(nan_ratio_)[:watch_rank]
        for i in range(len(sort_idx) // 2):
            sort_idx[i], sort_idx[-1-i] = sort_idx[-1-i], sort_idx[i]
        plt.subplot(2, 2, 4)
        ax = plt.gca()
        plt.bar(np.arange(watch_rank), nan_ratio_[sort_idx])
        plt.ylim(-0.05, 1.05)
        plt.xlim(-0.9, (watch_rank -0.1))
        plt.title('nan ratio of X (BOTTOM%d)' % watch_rank)
        plt.ylabel('nan exist ratio')
        plt.rcParams["font.size"] = font_size
        plt.grid(True)
        # 
        plt.show()
    else:
        print('nan is not exist. check OK!')
    ############################################################

    return nan_ratio

# 
def check_value_count(X, 
                      column_name = None):
    
    # escape
    X_           = copy(X)
    column_name_ = copy(column_name)

    # if X is pandas dataframe then convert to numpy array
    if (str(type(X)).find('DataFrame') > -1):
        X_ = X_.values

    # adjust
    if (len(np.shape(X_)) == 1):
        X_ = X_[:, np.newaxis]

    # get basic info
    X_num = np.shape(X_)[0]
    X_dim = np.shape(X_)[1]
    
    # adjust
    if (column_name is None):
        if (str(type(X)).find('DataFrame') > -1):
            column_name_ = X.columns
        else:
            column_name_ = [('column_%d' % i) for i in range(X_dim)]
    if (type(column_name_) == str):
        column_name_ = [column_name_]
    if (type(column_name_) == list):
        column_name_ = np.array(column_name_)

    # detect almost numeric column
    print('\nnumeric ratio is ...')
    print('\n------------------------------------------------------------')
    # calc nan ratio
    for column_i in range(X_dim):
        val_cnt = pd.Series(X_[:, column_i]).value_counts(dropna=True)
        str_num = 0
        for val_cnt_i in range(len(val_cnt)):
            if (str(val_cnt.index[val_cnt_i]).replace('.', '').replace('+', '').replace('-', '').isnumeric() == False):
                str_num += val_cnt.values[val_cnt_i]
        num_ratio = (val_cnt.sum() - str_num) / val_cnt.sum()
        print('  - %s -> %.5f' % (column_name_[column_i], num_ratio))
    print('')

    # print value pattern
    print('\n')
    print('value pattern is ...')
    for column_i in range(X_dim):
        print('\n------------------------------------------------------------')
        print('[%s]' % column_name_[column_i])
        print(pd.Series(X_[:, column_i]).value_counts(dropna=False))

# 
def organize_data(X, 
                  column_name       = None, 
                  threshold_overlap = 0.99):
    
    # escape
    X_           = copy(X)
    column_name_ = copy(column_name)

    # if X is pandas dataframe then convert to numpy array
    if (str(type(X)).find('DataFrame') > -1):
        X_ = X_.values

    # adjust
    if (len(np.shape(X_)) == 1):
        X_ = X_[:, np.newaxis]

    # get basic info
    X_num = np.shape(X_)[0]
    X_dim = np.shape(X_)[1]
    
    # adjust
    if (column_name is None):
        if (str(type(X)).find('DataFrame') > -1):
            column_name_ = X.columns
        else:
            column_name_ = [('column_%d' % i) for i in range(X_dim)]
    if (type(column_name_) == str):
        column_name_ = [column_name_]
    if (type(column_name_) == list):
        column_name_ = np.array(column_name_)

    print('\ndelete one-pattern column...')

    # if value is one-pattern then reject auto
    column_remain     = []
    result_all_remain = True
    for column_i in range(X_dim):

        val_unique = np.array(pd.Series(X_[:, column_i]).unique()) # pick up nan
        val_count  = pd.Series(X_[:, column_i]).value_counts()     # not pick up nan
        if (len(val_unique) > 1):
            column_remain.append(column_i)
        else:
            print('  - %s is delete (only [%s])' % (column_name_[column_i], val_unique))
            result_all_remain = False
    if (result_all_remain):
        print('  - none ...')
    
    # drop
    X_           = X_[:, column_remain]
    column_name_ = column_name_[column_remain]
    X_dim        = len(column_remain)

    # If there is a redundant column set then reject auto
    print('\ndelete overlap column...')
    column_reject     = []
    result_all_remain = True
    for column_i in range(X_dim - 1):
        for column_j in range((column_i + 1), X_dim):
            # 
            value_left_tmp  = X_[:, column_i]
            value_right_tmp = X_[:, column_j]
            # 
            overlap_ratio   = np.sum(value_left_tmp == value_right_tmp) / X_num
            if (overlap_ratio >= threshold_overlap):
                if (column_j in column_reject):
                    pass
                else:
                    column_reject.append(column_j)
                    print('  - %s is delete ([%s] and [%s] is %.3f overlap)' % (column_name_[column_j], column_name_[column_i], column_name_[column_j], overlap_ratio))
                    result_all_remain = False
    if (result_all_remain):
        print('  - none ...')

    # drop
    column_remain                = np.ones(X_dim, dtype='bool')
    column_remain[column_reject] = False
    X_                           = X_[:, column_remain]
    column_name_                 = column_name_[column_remain]

    # adjust
    if (str(type(X)).find('DataFrame') > -1):
        X_ = pd.DataFrame(X_, columns=column_name_)

    return X_

# 
def one_k_code_and_disc(X, 
                        column_name   = None,
                        column_1k     = [], # 1-k code
                        column_disc   = [], # discretization
                        column_bow    = [], # bag of words
                        disc_bins_num = 10):
    
    # escape
    X_           = copy(X)
    column_name_ = copy(column_name)

    # if X is pandas dataframe then convert to numpy array
    if (str(type(X)).find('DataFrame') > -1):
        X_ = X_.values

    # adjust
    if (len(np.shape(X_)) == 1):
        X_ = X_[:, np.newaxis]

    # get basic info
    X_num = np.shape(X_)[0]
    X_dim = np.shape(X_)[1]
    
    # adjust
    if (column_name is None):
        if (str(type(X)).find('DataFrame') > -1):
            column_name_ = X.columns
        else:
            column_name_ = [('column_%d' % i) for i in range(X_dim)]
    if (type(column_name_) == str):
        column_name_ = [column_name_]
    if (type(column_name_) == list):
        column_name_ = np.array(column_name_)

    # check exist or not
    if (len(column_1k) > 0):
        column_1k   = [s for s in column_1k   if s in column_name_]
    if (len(column_disc) > 0):
        column_disc = [s for s in column_disc if s in column_name_]
    if (len(column_bow) > 0):
        column_bow  = [s for s in column_bow  if s in column_name_]
    
    # prepare output var as dataframe
    df_X = pd.DataFrame(index=np.arange(X_num), columns=[])
    
    ###############################################################
    process_num   = X_dim
    process_break = np.round(np.linspace(1, process_num, 50))
    process_i     = 0  
    str_now       = datetime.now()  
    print('\n1-k conding or discretization [start time is %s]' % (str_now)) 
    print('--------------------------------------------------')
    print('START                                          END') 
    print('----+----1----+----2----+----3----+----4----+----5') 
    ###############################################################

    for column_i in range(X_dim):
        
        ###########################################################
        process_i = process_i + 1   
        if (sum(process_break == process_i) > 0):
            for print_i in range(sum(process_break == process_i)): 
                print('*', end='', flush=True)                              
        ###########################################################

        # 1-k coding
        if (column_name_[column_i] in column_1k):
            X_tmp          = X_[:, column_i].reshape(-1)
            nan_idx        = ((X_tmp == X_tmp) == False)
            X_tmp[nan_idx] = STR_NAN
            df_X_tmp       = pd.get_dummies(pd.Series(X_tmp).astype(str).str.lower())
            df_X_tmp       = df_X_tmp.rename(columns=lambda s: column_name_[column_i]+'_c['+str(s)+']')
            if (df_X_tmp.shape[1] == 2):
                df_X_tmp   = df_X_tmp.drop(columns=df_X_tmp.columns[1], axis=1)
            df_X           = pd.concat([df_X, df_X_tmp], axis=1)

        # discretization 
        elif (column_name_[column_i] in column_disc):
            (X_tmp, 
             column_name_tmp) = disc_plus_alpha(X             = X_[:, column_i], 
                                                column_name   = column_name_[column_i], 
                                                disc_bins_num = disc_bins_num)
            df_X_tmp          = pd.DataFrame(X_tmp, columns=column_name_tmp)
            df_X              = pd.concat([df_X, df_X_tmp], axis=1)

        # bag of words (Half way there!!!)
        elif (column_name_[column_i] in column_bow):
            df_X_tmp, _ = bow(df_in.loc[:, [column_name_[column_i]]])
            df_X        = pd.concat([df_X, df_X_tmp], axis=1)

    ###############################################################
    str_now = datetime.now()  
    print('\n                              [end time is %s]' % (str_now)) 
    print('\n') 
    ###############################################################

    print('\nafter quantize size = (%d, %d)' % df_X.shape)
    
    # adjust
    column_name_ = df_X.columns
    if (str(type(X)).find('DataFrame') == -1):
        df_X = df_X.values

    return (df_X, 
            column_name_)

#
def disc_plus_alpha(X, 
                    column_name    = None, 
                    disc_bins_num  = 10, 
                    disc_bins_even = True):
    
    # escape
    X_           = copy(X)
    column_name_ = copy(column_name)

    # if X is pandas dataframe then convert to numpy array
    if ((str(type(X)).find('Series') > -1) | 
        (str(type(X)).find('DataFrame') > -1)):
        X_ = X_.values

    # adjust
    if (len(np.shape(X_)) == 2):
        X_ = X_.reshape(-1)
    
    # get basic info
    X_num = len(X_)

    # adjust
    if (column_name is None):
        if (str(type(X)).find('DataFrame') > -1):
            column_name_ = X.columns
        else:
            column_name_ = 'column_XXX'
    if (type(column_name_) == np.ndarray):
        column_name_ = column_name_[0]
    if (type(column_name_) == list):
        column_name_ = column_name_[0]

    # prepare output var as dataframe
    df_X = pd.DataFrame(index=np.arange(X_num), columns=[])
    
    # get index of number or string
    number_idx  = pd.Series(pd.Series([str(s) for s in X.astype('str')]).str.replace('.', '').replace('+', '').values).str.isdecimal()
    string_idx  = ~number_idx

    if (np.sum(string_idx) > 0):
        # dummy for string
        X_tmp           = X_
        nan_idx         = ((X_tmp == X_tmp) == False)
        X_tmp           = pd.Series(pd.Series(X_tmp).astype(str).str.lower().values).astype(object).values
        X_tmp[nan_idx]  = STR_NAN
        X_tmp[~nan_idx] = np.nan
        df_X_tmp        = pd.get_dummies(pd.Series(X_tmp))
        df_X_tmp        = df_X_tmp.rename(columns=lambda s: column_name_+'_c['+str(s)+']')
        df_X            = pd.concat([df_X, df_X_tmp], axis=1)
    
    # hist for number
    number_val = X_[number_idx].astype('float')
    if (disc_bins_even):
        bins       = (np.arange(disc_bins_num + 1) / disc_bins_num) * np.max(number_val)
    else:
        bins_ratio = np.arange(disc_bins_num + 1)
        bins_ratio = bins_ratio / np.max(bins_ratio) * 100
        bins       = np.zeros(len(bins_ratio))
        bins       = np.percentile(np.sort(number_val), bins_ratio)
    # make histgram
    number_hist        = np.zeros([np.sum(number_idx), disc_bins_num], dtype='float')
    column_number_hist = []
    for bins_i in range(disc_bins_num):
        if (bins_i == 0):
            number_hist[:, bins_i] = (((bins[bins_i] - 1e-10) < number_val) & (number_val <= bins[bins_i+1]))
        else:
            number_hist[:, bins_i] = ((bins[bins_i] < number_val) & (number_val <= bins[bins_i+1]))
        column_number_hist.append('%s_v[%.3f-%.3f]' % (column_name_, np.round(bins[bins_i], decimals=3), np.round(bins[bins_i+1], decimals=3)))
    # 
    df_X_tmp    = pd.DataFrame(data=np.zeros([X_num, len(column_number_hist)]), columns=column_number_hist)
    number_idx_ = np.where(number_idx)[0]
    for row_i in range(len(number_idx_)):
        df_X_tmp.iloc[number_idx_[row_i], :] = number_hist[row_i, :]
    # 
    df_X = pd.concat([df_X, df_X_tmp], axis=1)

    # adjust
    column_name_ = df_X.columns
    if ((str(type(X)).find('Series')    == -1) & 
        (str(type(X)).find('DataFrame') == -1)):
        df_X = df_X.values

    return (df_X, 
            column_name_)

# 
def bow(df_in):

    # (Half way there!!!)

    sr_in = df_in.iloc[:, 0]

    # reject number
    for num_i in range(10):
        sr_in = sr_in.str.replace(('%d' % num_i), '0')
    for num_i in range(100):
        sr_in = sr_in.str.replace('00', '0')

    # convert to lower
    sr_in = sr_in.str.lower()

    word_in = sr_in.values

    # reject stopwords
    stop_word       = stopwords.words('english')
    remain_word_idx = np.ones(len(word_in)).astype('bool')
    for stop_word_i in range(len(stop_word)):
        for row_i in range(len(word_in)):
            if (stop_word[stop_word_i] == word_in[row_i]):
                remain_word_idx[row_i] = False
                break
    
    word_in = word_in[remain_word_idx]

    # bow
    model_tmp  = CountVectorizer(max_features=3000)
    # print(np.unique(word_in))
    model_tmp.fit(word_in)
    word_bow   = np.array([d[0] for d in model_tmp.vocabulary_.items()])

    column_bow = np.array([('%s_w[%s]' % (sr_in.name, d[0])) for d in model_tmp.vocabulary_.items()])
    value_bow  = model_tmp.transform(sr_in.values).toarray()

    sort_idx   = np.argsort(-np.sum(value_bow, axis=0))

    df_out = pd.DataFrame(data=value_bow[:, sort_idx], columns=column_bow[sort_idx])
    return df_out, word_bow

# 
def bag_of_vote(df_in, 
                column_key):

    if (len(column_key) == 1):
        key_value     = df_in.loc[:, column_key].values
    else:
        key_value     = df_in.loc[:, column_key[0]].values
        for column_key_i in range(1, len(column_key)):
            key_value = np.core.defchararray.add(key_value, df_in.loc[:, column_key[column_key_i]].values)

    key_break_row = (np.where(key_value[1:] != key_value[:-1])[0]) + 1
    key_break_row = np.concatenate([[0], key_break_row, [len(key_value)]])

    num_value = df_in.values[:, 1:]
    sum_value = np.zeros([(len(key_break_row) - 1), np.shape(num_value)[1]])

    ###############################################################
    process_num   = len(key_break_row) - 1
    process_break = np.round(np.linspace(1, process_num, 50))
    process_i     = 0  
    str_now       = datetime.now()  
    print('summary to 1 row [start time is %s]' % (str_now)) 
    print('--------------------------------------------------')
    print('START                                          END') 
    print('----+----1----+----2----+----3----+----4----+----5') 
    ###############################################################

    for key_i in range(len(key_break_row) - 1):
        
        ###########################################################
        process_i = process_i + 1   
        if (sum(process_break == process_i) > 0):
            for print_i in range(sum(process_break == process_i)): 
                print('*', end='', flush=True)                              
        ###########################################################

        if ((key_break_row[key_i+1] - key_break_row[key_i]) == 1):
            # substitute    
            sum_value[key_i, :] = num_value[key_break_row[key_i], :]
        else:
            # sum
            sum_value[key_i, :] = np.sum(num_value[key_break_row[key_i]:key_break_row[key_i+1], :], axis=0)

    sum_value = np.concatenate([key_value[key_break_row[:-1]][:, np.newaxis], sum_value], axis=1)
    df_out    = pd.DataFrame(data=sum_value, columns=df_in.columns)

    ###############################################################
    str_now = datetime.now()  
    print('\n                 [end time is %s]' % (str_now)) 
    print('\n') 
    ###############################################################

    print('after summarize size    = (%d, %d)' % df_out.shape)
    print('number of %s\'s var = %d' % (column_key, len(df_out.loc[:, column_key].unique())))

    return df_out

#
def X_regularization(X, min=None, max=None, eps=1e-20):

    if (min is None):
        min = np.min(X, axis=0)

    if (max is None):
        max = np.max(X, axis=0)

    return ((X - min) / (max - min + eps)), min, max

# 
def X_normalization(X, mean=None, std=None, eps=1e-20):

    if (mean is None):
        mean = np.mean(X, axis=0)

    if (std is None):
        std = np.std(X, axis=0)
    
    return ((X - mean) / (std + eps)), mean, std

# 
def cooc_feat(X, 
              column_name):
    #

    # FIND
    D            = np.shape(X)[1]
    D_           = int((D * D) - (D * (D + 1) / 2))
    N            = len(X)
    X_           = np.zeros([N, D_], dtype='float32')
    column_name_ = []

    set_i = 0

    for column_i in range(D - 1):
        for column_j in range((column_i + 1), D):
            column_name_.append('%s_x_%s' % (column_name[column_i], column_name[column_j]))

    ###############################################################
    process_num   = D - 1
    process_break = np.round(np.linspace(1, process_num, 50))
    process_i     = 0  
    str_now       = datetime.now()  
    print('make cooc feat [start time is %s]' % (str_now)) 
    print('--------------------------------------------------')
    print('START                                          END') 
    print('----+----1----+----2----+----3----+----4----+----5') 
    ###############################################################
        
    set_i = 0

    for column_i in range(D - 1):
        
        ###########################################################
        process_i = process_i + 1   
        if (sum(process_break == process_i) > 0):
            for print_i in range(sum(process_break == process_i)): 
                print('*', end='', flush=True)                              
        ###########################################################
        
        for column_j in range(column_i + 1, D):
            
            # print('%d, %d' % (column_i, column_j))
            X_[:, set_i] = X[:, column_i] * X[:, column_j]
            set_i += 1

    ###############################################################
    str_now = datetime.now()  
    print('\n                       [end time is %s]' % (str_now)) 
    print('') 
    ###############################################################

    X           = np.concatenate([X, X_], axis=1)
    column_name = np.concatenate([column_name, np.array(column_name_)])

    return (X, column_name)

# 
def check_X_and_y_status(X,                        
                         y,                        
                         column_name,           
                         abs_flg           = True, 
                         cross_plot        = True, 
                         cross_plot_assign = None, 
                         font_size         = 12):
    
    # adjust    
    if (column_name is None):
        column_name = [('column_%d' % i) for i in range(X_dim)]
    if (type(column_name) == list):
        column_name = np.array(column_name)

    ############################################################
    print('Total data num    :%7d件' % len(X))
    ############################################################
   
    # 2class
    if (len(np.unique(y)) == 2):
        # calc pos and neg ratio
        X_pos = X[(y == 1), :]
        y_pos = y[(y == 1)]
        
        X_neg = X[(y == 0), :]
        y_neg = y[(y == 0)]
        
        ########################################################
        print('Positive data num :%7d件 (%5.2f%%)' % (len(X_pos), (len(X_pos) / len(X) * 100)))
        print('Negative data num :%7d件 (%5.2f%%)' % (len(X_neg), (len(X_neg) / len(X) * 100)))
        print('')
        print('')
        ########################################################

    elif ((2 < len(np.unique(y))) & (len(np.unique(y)) <= MCLASS_DEFINE)):
        
        ########################################################
        print('Total data num    :%7d件' % len(X))
        ########################################################
        
        for (class_i, class_value_tmp) in enumerate(np.unique(y)):
            # calc pos and neg ratio
            X_tmp = X[(y == class_value_tmp), :]
            y_tmp = y[(y == class_value_tmp)]

            ####################################################
            print('class %d data num :%7d件 (%5.2f%%)' % (class_i, np.sum(y == class_value_tmp), (np.sum(y == class_value_tmp) / len(X) * 100)))
        print('')
        print('')
        ########################################################

    if ((2 < len(np.unique(y))) & (len(np.unique(y)) <= MCLASS_DEFINE)):
        # multi class
        sort_idx    = np.zeros([len(column_name)])
        corr_with_y = []
        for (class_i, class_value_tmp) in enumerate(np.unique(y)):
            # calc pos and neg ratio
            y_tmp                       = y.copy()
            y_tmp[y != class_value_tmp] = 0
            y_tmp[y == class_value_tmp] = 1
            print('class %d' % class_value_tmp)
            (sort_idx_tmp, corr_with_y_tmp) = correlation_visualize(X, 
                                                                    y_tmp, 
                                                                    column_name, 
                                                                    abs_flg)
            sort_idx += sort_idx_tmp
            corr_with_y.append(corr_with_y_tmp)
        sort_idx = np.argsort(sort_idx)
        corr_with_y = np.array(corr_with_y)
    else:
        # 2 class or regression
        (sort_idx, corr_with_y) = correlation_visualize(X, 
                                                        y, 
                                                        column_name, 
                                                        abs_flg=abs_flg)

    if (cross_plot):
        ########################################################
        print('cross plot')
        # cross plot, boxplot, histogram and ratio at range
        if (cross_plot_assign is None):
            cross_plot_assign = np.arange(X.shape[1])[sort_idx]
            cross_plot_num    = 10
            if (len(cross_plot_assign) > cross_plot_num):
                cross_plot_assign = cross_plot_assign[:cross_plot_num]
                print('(high corr TOP%d)' % cross_plot_num)
        else:
            cross_plot_num = len(cross_plot_assign)

        for item_i in cross_plot_assign:
            
            # get values of 1 column
            X_       = X[:, item_i]
            
            # sort for visibility
            sort_idx = np.argsort(X_)
            X_sort   = X_[sort_idx]
            y_sort   = y[sort_idx]
            X_sort_  = np.empty([0])
            y_sort_  = np.empty([0])
            for (class_i, class_value_tmp) in enumerate(np.unique(y)):
                X_sort_  = np.concatenate([X_sort_, X_sort[(y_sort == class_value_tmp).ravel()]], axis=0)
                y_sort_  = np.concatenate([y_sort_, y_sort[(y_sort == class_value_tmp).ravel()]], axis=0)
            
            # make histogram
            X_hist             = []
            # at first...
            X_hist_info        = np.histogram(X_, bins=20)
            X_hist_info[1][-1] = X_hist_info[1][-1] + 1e-10
            # 
            bins_mean          = np.zeros(len(X_hist_info[1]) - 1)
            for bins_i in range(len(bins_mean)):
                bins_mean[bins_i] = (X_hist_info[1][bins_i] + X_hist_info[1][bins_i + 1]) / 2
            # 
            for (class_i, class_value_tmp) in enumerate(np.unique(y)):
                # then after...
                X_hist_tmp     = np.zeros(len(bins_mean))
                for bins_i in range(len(bins_mean)):
                    X_hist_tmp[bins_i] = np.sum((X_hist_info[1][bins_i] <= X_[(y == class_value_tmp).ravel()]) & (X_[(y == class_value_tmp).ravel()] < X_hist_info[1][bins_i + 1]))
                X_hist.append(X_hist_tmp)
            
            # 各変数ごとのクロスプロットやら、箱ヒゲ図やらを出力する
            fig  = plt.figure(figsize=(12,6),dpi=100)
            
            ax11 = plt.subplot(2, 2, 1)
            plt.scatter(np.arange(len(X_sort_)), X_sort_, alpha=0.1, label='x')
            plt.ylabel('x value')
            plt.ylabel('data index')
            plt.title('%s' % column_name[item_i])
            plt.rcParams["font.size"] = font_size
            plt.grid(True)
            ax12 = ax11.twinx()  # 2つのプロットを関連付ける
            plt.scatter([0], [0], alpha=0.05, label='x')
            plt.scatter(np.arange(len(y_sort_)), y_sort_, alpha=0.1, label='y')
            plt.yticks([])
            plt.legend(loc='upper left', fontsize=8)
            
            ax21 = plt.subplot(2, 2, 2)
            boxplot_table  = []
            boxplot_legend = []
            for (class_i, class_value_tmp) in enumerate(np.unique(y)):
                boxplot_table.append(X_[(y == class_value_tmp).ravel()])
                boxplot_legend.append('y == %s' % class_value_tmp)
            plt.boxplot(boxplot_table)
            ax21.set_xticklabels(boxplot_legend)
            plt.ylabel('x value')
            plt.rcParams["font.size"] = font_size
            plt.grid(True)
            
            plt.subplot(2, 2, 3)
            for (class_i, class_value_tmp) in enumerate(np.unique(y)):
                plt.bar(bins_mean, X_hist[class_i], alpha=0.5, label=('y == %s' % class_value_tmp), width=((bins_mean[1] - bins_mean[0]) * 0.9))
            plt.legend(loc='lower right', fontsize=8)
            plt.ylabel('frequency')
            plt.xlabel('x value')
            plt.rcParams["font.size"] = font_size
            plt.grid(True)
            
            plt.subplot(2, 2, 4)
            X_hist_sum       = np.zeros(len(X_hist[0]))
            X_hist_ratio_sum = np.zeros(len(X_hist[0]))
            for (class_i, class_value_tmp) in enumerate(np.unique(y)):
                X_hist_sum = X_hist_sum + X_hist[class_i]
            for (class_i, class_value_tmp) in enumerate(np.unique(y)):
                X_hist_ratio_tmp = X_hist[class_i] / (X_hist_sum + 1e-10)
                plt.bar(bins_mean, X_hist_ratio_tmp, bottom=X_hist_ratio_sum, label=('y == %s' % class_value_tmp), width=((bins_mean[1] - bins_mean[0]) * 0.9))
                X_hist_ratio_sum = X_hist_ratio_sum + X_hist_ratio_tmp
            plt.legend(loc='lower right', fontsize=8)
            plt.ylabel('frequency ratio')
            plt.xlabel('x value')
            plt.rcParams["font.size"] = font_size
            plt.grid(True)
            
            print('---------------------------------------------')
            print('[%d]' % item_i)
            plt.show()
            print('')
            print('')
        ########################################################
        
    return corr_with_y 

# 
def correlation_visualize(X, 
                          y, 
                          column_name, 
                          eps             = 1e-20, 
                          abs_flg         = True, 
                          watch_corr_rank = 30, 
                          font_size       = 12):
    # adjust
    if (watch_corr_rank > np.shape(X)[1]):
        watch_corr_rank = np.shape(X)[1]

    # X is matrix/vector, y is vector 

    # leakage and predictable check
    X_          = X - np.mean(X, axis=0)
    if (len(np.shape(X)) == 1):
        X_      = X_[:, np.newaxis]
    y_          = y - np.mean(y, axis=0)
    y_          = y_[:, np.newaxis]
    
    corr_with_y = np.dot(X_.T, y_).T / (np.sqrt(np.sum((X_ ** 2), axis=0)) * np.sqrt(np.sum((y_ ** 2), axis=0)) + eps)
    if (abs_flg):
        corr_with_y = np.abs(corr_with_y)
    corr_with_y = corr_with_y.ravel()

    ############################################################
    sort_idx = np.argsort(-np.abs(corr_with_y))[:watch_corr_rank]
    
    fig = plt.figure(figsize=(12,8),dpi=100)

    ax = plt.subplot(2, 1, 1)
    if (len(corr_with_y) > 1000):
        width = 1.6
    else:
        width = 0.8
    plt.bar(np.arange(len(corr_with_y)), corr_with_y, width=width)
    if (abs_flg):
        plt.ylim(-0.05, 1.05)
    plt.xlim(-0.9, (len(corr_with_y) - 0.1))
    plt.title('correlation with X and y of all column')
    plt.ylabel('absolute correlation value')
    # plt.xlabel('column index')
    plt.rcParams["font.size"] = font_size
    plt.grid(True)

    ax = plt.subplot(2, 1, 2)
    plt.bar(np.arange(watch_corr_rank), corr_with_y[sort_idx])
    if (abs_flg):
        plt.ylim(-0.05, 1.05)
    plt.xlim(-0.9, (watch_corr_rank - 0.1))
    plt.xticks(np.arange(watch_corr_rank))
    ax.set_xticklabels(column_name[sort_idx], rotation=90)
    plt.title('correlation with X and y (TOP%d)' % watch_corr_rank)
    plt.ylabel('absolute correlation value')
    plt.rcParams["font.size"] = font_size
    plt.grid(True)

    print('leakage check')
    plt.show()
    print('')
    print('')
    ############################################################

    return (sort_idx, 
            corr_with_y)

# 
def understandable_visualize(X,                     
                             y,                     
                             model           = PCA(), 
                             visualize_dim   = [1, 2], 
                             X_normalize     = True, 
                             X_outlier_care  = False, 
                             Xy_random_crop  = None,
                             freq_regularize = False, 
                             font_size       = 12):
    #
    X_ = copy(X)
    y_ = copy(y)

    # get basic info
    X_num = np.shape(X_)[0]

    # 
    if (Xy_random_crop is not None):
        idx_random_crop = np.random.permutation(np.arange(X_num))
        idx_random_crop = idx_random_crop[:int(np.round(X_num * Xy_random_crop))]
        X_              = X_[idx_random_crop, :]
        y_              = y_[idx_random_crop]

    # normalize 
    if (X_normalize):
        X_, X_mean, X_std = X_normalization(X=X_) 
        if (X_outlier_care):
            # suppress outlier
            for column_i in range(np.shape(X_)[1]):
                X_norm_tmp = X_[:, column_i]
                X_[(X_norm_tmp >  3), column_i] =  3
                X_[(X_norm_tmp < -3), column_i] = -3

    # model.fit(X)
    # X_proj = model.transform(X)
    X_proj = model.fit_transform(X_)

    X_proj_horz = []
    X_proj_vert = []
    for (class_i, class_value_tmp) in enumerate(np.unique(y_)):
        X_proj_horz.append(X_proj[(y_ == class_value_tmp), (visualize_dim[0] - 1)])
        X_proj_vert.append(X_proj[(y_ == class_value_tmp), (visualize_dim[1] - 1)])

    ############################################################
    print('2dim visualization')

    # plot PCA
    fig = plt.figure(figsize=(12,8),dpi=100)
    gs  = gridspec.GridSpec(9,9)

    plt.subplot(gs[:6, :6])
    for (class_i, class_value_tmp) in enumerate(np.unique(y_)):
        plt.scatter(X_proj_horz[class_i], X_proj_vert[class_i], alpha=0.5)
    plt.title('principal component')
    plt.xlabel('pc%d' % visualize_dim[0])
    plt.ylabel('pc%d' % visualize_dim[1])
    plt.rcParams["font.size"] = font_size
    plt.grid(True)

    # plot histogram
    plt.subplot(gs[-2:, :6])
    for (class_i, class_value_tmp) in enumerate(np.unique(y_)):
        plt.hist(X_proj_horz[class_i], alpha=0.5, bins=20, density=True)
    plt.xlabel('pc%d value' % visualize_dim[0])
    plt.rcParams["font.size"] = font_size
    plt.grid(True)

    plt.subplot(gs[:6, -2:])
    for (class_i, class_value_tmp) in enumerate(np.unique(y_)):
        plt.hist(X_proj_vert[class_i], alpha=0.5, bins=20, density=True, orientation="horizontal")
    plt.xlabel('pc%d value' % visualize_dim[1])
    plt.gca().invert_xaxis()
    plt.rcParams["font.size"] = font_size
    plt.grid(True)

    plt.show()
    ############################################################

    return model, X_, X_proj

#
def kmeans_classification(X, 
                          y, 
                          column_name    = None, 
                          k              = 10, 
                          random_state   = 0, 
                          X_normalize    = True, 
                          X_outlier_care = False, 
                          mode_draw      = True, 
                          font_size      = 12):
    
    #
    X_ = copy(X)

    # normalize 
    if (X_normalize):
        X_, X_mean, X_std = X_normalization(X=X_) 
        if (X_outlier_care):
            # suppress outlier
            for column_i in range(np.shape(X_)[1]):
                X_norm_tmp = X_[:, column_i]
                X_[(X_norm_tmp >  3), column_i] =  3
                X_[(X_norm_tmp < -3), column_i] = -3

    #
    kmeans        = KMeans(n_clusters=k, random_state=random_state, init='k-means++')
    kmeans_result = kmeans.fit(X_)

    # get cluster index
    kmeans_label  = kmeans_result.labels_

    # 
    data_num_of_cluster   = np.zeros([k, len(np.unique(y))])

    # loop of kmeans cluster
    for kmeans_label_i in range(k):
        X_of_cluster                        = X_[(kmeans_label == kmeans_label_i), :]
        y_of_cluster                        = y[(kmeans_label == kmeans_label_i)]
        # loop of class value
        for (class_i, class_value_tmp) in enumerate(np.unique(y)):
            data_num_of_cluster[kmeans_label_i, class_i]   = np.sum(y_of_cluster == class_value_tmp)
    
    # 
    data_ratio_of_cluster = data_num_of_cluster / (np.sum(data_num_of_cluster, axis=1)[:, np.newaxis])

    if (mode_draw):
        #
        sort_value_tmp = []
        for (class_i, _) in enumerate(np.unique(y)):
            sort_value_tmp.append(-data_ratio_of_cluster[:, class_i])
        for (class_i, _) in enumerate(np.unique(y)):
            sort_value_tmp.append(-data_num_of_cluster[:, class_i])
        # provisional lexsort specification...
        idx_k_sort = np.lexsort(sort_value_tmp[::-1])

        ####################################################
        # 
        fig = plt.figure(figsize=(12,12),dpi=100)
        # 
        ax = plt.subplot(2, 1, 1)
        data_num_of_cluster_sum = np.zeros([k])
        for (class_i, class_value_tmp) in enumerate(np.unique(y)):
            plt.bar(np.arange(k), data_num_of_cluster[idx_k_sort, class_i], bottom=data_num_of_cluster_sum[idx_k_sort])
            data_num_of_cluster_sum = data_num_of_cluster_sum + data_num_of_cluster[:, class_i]
        plt.xticks(np.arange(k))
        ax.set_xticklabels(idx_k_sort, rotation=90)
        plt.xlabel('cluster index')
        plt.ylabel('number of data')
        plt.title('K[%d]-means separate' % k)
        plt.rcParams["font.size"] = font_size
        plt.grid(True)
        # 
        ax = plt.subplot(2, 1, 2)
        data_ratio_of_cluster_sum = np.zeros([k])
        for (class_i, class_value_tmp) in enumerate(np.unique(y)):
            plt.bar(np.arange(k), data_ratio_of_cluster[idx_k_sort, class_i], bottom=data_ratio_of_cluster_sum[idx_k_sort])
            data_ratio_of_cluster_sum = data_ratio_of_cluster_sum + data_ratio_of_cluster[:, class_i]
        plt.xticks(np.arange(k))
        ax.set_xticklabels(idx_k_sort, rotation=90)
        plt.xlabel('cluster index')
        plt.ylabel('ratio of positive and negative')
        plt.rcParams["font.size"] = font_size
        plt.grid(True)
        ####################################################

        ####################################################
        fig  = plt.figure(figsize=(12,8),dpi=100)
        ax   = plt.subplot(2, 1, 1)
        cmap = plt.get_cmap("tab10")
        ####################################################

        # 
        for (class_i, class_value_tmp) in enumerate(np.unique(y)):
            # provisional lexsort specification...
            idx_k_tmp = np.lexsort([-data_num_of_cluster[:, class_i], 
                                    -data_ratio_of_cluster[:, class_i]])[0]
            # 
            X_tmp     = X_[(kmeans_result.labels_ == idx_k_tmp), :]
            # 
            X_mean_tmp         = np.mean(X_tmp, axis=0)
            X_std_tmp          = np.std(X_tmp, axis=0)
            X_1sigma_minus_tmp = X_mean_tmp - X_std_tmp
            X_1sigma_plus_tmp  = X_mean_tmp + X_std_tmp

            ################################################
            # 
            for column_i in range(len(column_name)):
                #
                plt.plot([column_i, column_i, column_i], [X_1sigma_minus_tmp[column_i], X_mean_tmp[column_i], X_1sigma_plus_tmp[column_i]], color=cmap(class_i), alpha=0.5)
                if (column_i == 1):
                    plt.scatter([column_i, column_i, column_i], [X_1sigma_minus_tmp[column_i], X_mean_tmp[column_i], X_1sigma_plus_tmp[column_i]], color=cmap(class_i), alpha=0.5, label=('cluster %d' % idx_k_tmp))
                else:
                    plt.scatter([column_i, column_i, column_i], [X_1sigma_minus_tmp[column_i], X_mean_tmp[column_i], X_1sigma_plus_tmp[column_i]], color=cmap(class_i), alpha=0.5)
        # 
        plt.legend(loc='best')
        plt.xticks(np.arange(len(column_name)))
        ax.set_xticklabels([]) # column_name, rotation=90)
        plt.rcParams["font.size"] = font_size
        plt.title('distribtion of cluster (prior class data ratio)')
        if (X_normalize):
            plt.ylabel('normalized value (mean, ±1σ)')
        else:
            plt.ylabel('value (mean, ±1σ)')
        plt.xlabel('column index')
        plt.rcParams["font.size"] = font_size
        plt.grid(True)
        # 
        ####################################################

        ####################################################
        ax   = plt.subplot(2, 1, 2)
        ####################################################

        # 
        for (class_i, class_value_tmp) in enumerate(np.unique(y)):
            # provisional lexsort specification...
            idx_k_tmp = np.lexsort([-data_ratio_of_cluster[:, class_i], 
                                    -data_num_of_cluster[:, class_i]])[0]
            # 
            X_tmp     = X_[(kmeans_result.labels_ == idx_k_tmp), :]
            # 
            X_mean_tmp         = np.mean(X_tmp, axis=0)
            X_std_tmp          = np.std(X_tmp, axis=0)
            X_1sigma_minus_tmp = X_mean_tmp - X_std_tmp
            X_1sigma_plus_tmp  = X_mean_tmp + X_std_tmp

            ################################################
            # 
            for column_i in range(len(column_name)):
                #
                plt.plot([column_i, column_i, column_i], [X_1sigma_minus_tmp[column_i], X_mean_tmp[column_i], X_1sigma_plus_tmp[column_i]], color=cmap(class_i), alpha=0.5)
                if (column_i == 1):
                    plt.scatter([column_i, column_i, column_i], [X_1sigma_minus_tmp[column_i], X_mean_tmp[column_i], X_1sigma_plus_tmp[column_i]], color=cmap(class_i), alpha=0.5, label=('cluster %d' % idx_k_tmp))
                else:
                    plt.scatter([column_i, column_i, column_i], [X_1sigma_minus_tmp[column_i], X_mean_tmp[column_i], X_1sigma_plus_tmp[column_i]], color=cmap(class_i), alpha=0.5)
        # 
        plt.legend(loc='best')
        plt.xticks(np.arange(len(column_name)))
        ax.set_xticklabels(column_name, rotation=90)
        plt.rcParams["font.size"] = font_size
        plt.title('distribtion of cluster (prior class data amount)')
        if (X_normalize):
            plt.ylabel('normalized value (mean, ±1σ)')
        else:
            plt.ylabel('value (mean, ±1σ)')
        plt.xlabel('column index')
        plt.rcParams["font.size"] = font_size
        plt.grid(True)
        # 
        plt.show()
        ####################################################

    return (kmeans_result, X_, idx_k_sort)

#
def knn_classification(X, 
                       y, 
                       k              = 20, 
                       p              = 2, 
                       X_normalize    = True,
                       X_outlier_care = False, 
                       font_size      = 12):
    
    #
    X_ = copy(X)

    # normalize 
    if (X_normalize):
        X_, X_mean, X_std = X_normalization(X=X_) 
        if (X_outlier_care):
            # suppress outlier
            for column_i in range(np.shape(X_)[1]):
                X_norm_tmp = X_[:, column_i]
                X_[(X_norm_tmp >  3), column_i] =  3
                X_[(X_norm_tmp < -3), column_i] = -3

    ###################################################################
    fig  = plt.figure(figsize=(12, (6 * len(np.unique(y)))), dpi=100)
    cmap = plt.get_cmap("tab10")
    # 
    process_num   = len(y)
    process_break = np.round(np.linspace(1, process_num, 50))
    process_i     = 0  
    str_now       = datetime.now()  
    print('k-nn searching on positive data [start time is %s]' % (str_now)) 
    print('--------------------------------------------------')
    print('START                                          END') 
    print('----+----1----+----2----+----3----+----4----+----5') 
    ###################################################################

    # 
    for (class_i, class_value_tmp) in enumerate(np.unique(y)):

        # acquire peripheral data centered on positive data
        neighbor_num = np.zeros([np.sum(y == class_value_tmp), len(np.unique(y))])
        class_idx    = np.array(np.where(y == class_value_tmp)[0])
        class_idx    = np.random.permutation(class_idx)

        # 
        for class_data_i in range(len(class_idx)):
            
            ###########################################################
            process_i = process_i + 1   
            if (sum(process_break == process_i) > 0):
                for print_i in range(sum(process_break == process_i)): 
                    print('*', end='', flush=True)                              
            ###########################################################
            
            X_diff_LX_tmp = np.sqrt( ((X_[class_idx[class_data_i], :] - X_)**p).sum(axis=1) ) # calc minkowski distance
            neighbor_idx  = np.argsort(X_diff_LX_tmp)
            # 
            for (class_j, class_value_tmp_) in enumerate(np.unique(y)):
                neighbor_num[class_data_i, class_j] = np.sum(y[neighbor_idx[0:(k + 1)]] == class_value_tmp_) - int(class_i == class_j) # subtract 1 for the number of data of itself -> number of positive data in close distance

        ###############################################################
        neighbor_num_tmp = []
        for class_j in np.arange(len(np.unique(y)))[np.argsort(np.abs(np.arange(len(np.unique(y))) - class_i))]:
            neighbor_num_tmp.append(neighbor_num[:, class_j])
        sort_idx = np.lexsort(neighbor_num_tmp)    
        #  
        plt.subplot(len(np.unique(y)), 1, (class_i + 1))
        # 
        neighbor_num_sum = np.zeros(len(neighbor_num))
        for class_j in np.arange(len(np.unique(y)))[np.argsort(np.abs(np.arange(len(np.unique(y))) - class_i))]:
            plt.bar(np.arange(len(neighbor_num)), neighbor_num[sort_idx, class_j], bottom=neighbor_num_sum[sort_idx], color=cmap(class_j), label=('class %s' % np.unique(y)[class_j]))
            neighbor_num_sum = neighbor_num_sum + neighbor_num[:, class_j]
        plt.xlabel('class %d data index' % class_i)
        plt.ylabel('ratio of sevral class')
        plt.title('K[%d]-nearest neighbor (class %d data based)' % (k, class_i))
        plt.rcParams["font.size"] = font_size
        plt.legend(loc='lower left')
        plt.grid(True)
    # 
    str_now = datetime.now()  
    print('\n                                [end time is %s]' % (str_now)) 
    print('') 
    # 
    plt.show()
    ###################################################################

    return X_

# 
def knn_mistake_search(X, 
                       y, 
                       y_hat, 
                       column_name       = None,       
                       base_idx_num      = 5, 
                       k                 = 10, 
                       p                 = 2, 
                       X_normalize       = True, 
                       X_outlier_care    = False):
    
    # get info
    class_num  = int(np.max(y) + 1) # y = 0, 1, 2, 3, ...
    sample_num = len(y)

    # adjust y_hat
    y_hat_tmp = y_hat.copy()
    if (y_hat_tmp.ndim == 1):
        y_hat_tmp = y_hat_tmp[:, np.newaxis]
    if (np.shape(y_hat_tmp)[1] == 1):
        y_hat_tmp = np.concatenate([y_hat_tmp, (1 - y_hat_tmp)], axis=1)

    # calc mistake
    mistake = np.zeros(sample_num)
    for sample_i in range(sample_num):
        mistake[sample_i] = 1 - y_hat_tmp[sample_i, int(y[sample_i])]
    mistake_rank = np.argsort(-mistake)
    mistake_rank = mistake_rank[:base_idx_num]

    # summary y_hat
    y_hat_summary = np.empty(sample_num, dtype='object')
    for sample_i in range(sample_num):
        y_hat_summary_tmp = ''
        for class_i in range(class_num):
            y_hat_summary_tmp = ('%s%d:%.3f, ' % (y_hat_summary_tmp, class_i, y_hat_tmp[sample_i, class_i]))
        y_hat_summary[sample_i] = y_hat_summary_tmp[:-1]

    #
    X_           = copy(X)
    column_name_ = copy(column_name)

    # get basic info
    X_num = np.shape(X_)[0]
    X_dim = np.shape(X_)[1]

    # normalize 
    if (X_normalize):
        X_, X_mean, X_std = X_normalization(X=X_) 
        if (X_outlier_care):
            # suppress outlier
            for column_i in range(np.shape(X_)[1]):
                X_norm_tmp = X_[:, column_i]
                X_[(X_norm_tmp >  3), column_i] =  3
                X_[(X_norm_tmp < -3), column_i] = -3
    
    # adjust
    if (column_name is None):
        if (str(type(X)).find('DataFrame') > -1):
            column_name_ = X.columns
        else:
            column_name_ = [('column_%d' % i) for i in range(X_dim)]
    if (type(column_name_) == str):
        column_name_ = [column_name_]
    if (type(column_name_) == list):
        column_name_ = np.array(column_name_)
    if (k > X_num):
        k = X_num
    
    # prepare work
    distance     = np.zeros([base_idx_num, len(X_)], dtype='float')
    nearest_rank = np.zeros([base_idx_num, len(X_)], dtype='int')
    nearest_info = []

    # 
    base_i = 0
    for row_i in mistake_rank:
        # 
        distance_tmp     = np.abs(X_ - X_[row_i, :])
        distance_tmp     = distance_tmp ** p
        distance_tmp     = np.sum(distance_tmp, axis=1)
        # 
        nearest_rank_tmp = np.argsort(distance_tmp)
        distance_tmp     = distance_tmp[nearest_rank_tmp]
        #
        distance_tmp     = distance_tmp[nearest_rank_tmp != row_i]
        nearest_rank_tmp = nearest_rank_tmp[nearest_rank_tmp != row_i]
        #
        distance[base_i, 0]      = 0
        nearest_rank[base_i, 0]  = row_i
        distance[base_i, 1:]     = distance_tmp
        nearest_rank[base_i, 1:] = nearest_rank_tmp
        # 
        df_tmp = pd.DataFrame(data=np.concatenate([nearest_rank[[base_i], :], 
                                                   distance[[base_i], :],
                                                   y[np.newaxis, nearest_rank[base_i, :]], 
                                                   y_hat_summary[np.newaxis, :], 
                                                   mistake[np.newaxis, nearest_rank[base_i, :]], 
                                                   X.T[:, nearest_rank[base_i, :]]], axis=0)[:, :(k + 1)], 
                              columns=['base'] + [('neighbor%d' % (i+1)) for i in range(k)], 
                              index=['data index', 'distance', 'correct label', 'prediction', 'mistake'] + list(column_name))
        nearest_info.append(df_tmp)
        # 
        base_i += 1
    #
    return (nearest_info)

# importance 1dim -> (column index)
# importance 2dim -> (column index, cross validation index)
# importance 3dim -> (column index, cross validation index, model index)
def draw_importance(importance, 
                    column_name, 
                    watch_rank = 30, 
                    draw_mode  = 0, 
                    font_size  = 12):
    
    # not create figure on this method

    # 
    if (len(np.shape(importance)) == 1):
        importance = importance[:, np.newaxis]
    if (len(np.shape(importance)) == 3):
        importance = importance.reshape(-1, (np.shape(importance)[1] * np.shape(importance)[2])) # dim 1:(column), dim 2:(model * cv)
    if (watch_rank > np.shape(importance)[0]):
        watch_rank = np.shape(importance)[0]
    # 
    importance_mean  = np.mean(importance, axis=1)
    importance_std   = np.std(importance, axis=1)
    importance_p1sig = importance_mean + importance_std
    importance_m1sig = importance_mean - importance_std
    importance_max   = np.max(importance, axis=1)
    importance_min   = np.min(importance, axis=1)

    ############################################################
    cmap = plt.get_cmap("tab10")
    # 
    if (draw_mode == 0):
        sort_idx = np.argsort(-np.abs(importance_mean))
        plt.bar(np.arange(watch_rank), importance_mean[sort_idx[:watch_rank]], color=cmap(0), alpha=0.5)
        for watch_rank_i in range(watch_rank):
            plt.plot([watch_rank_i, watch_rank_i, watch_rank_i, watch_rank_i], 
                     [importance_max[sort_idx[watch_rank_i]], importance_p1sig[sort_idx[watch_rank_i]], importance_m1sig[sort_idx[watch_rank_i]], importance_min[sort_idx[watch_rank_i]]], color=cmap(0), alpha=0.9)
            plt.scatter([watch_rank_i, watch_rank_i, watch_rank_i, watch_rank_i], 
                        [importance_max[sort_idx[watch_rank_i]], importance_p1sig[sort_idx[watch_rank_i]], importance_m1sig[sort_idx[watch_rank_i]], importance_min[sort_idx[watch_rank_i]]], color=cmap(0), alpha=0.9)
        plt.xticks(np.arange(watch_rank))
        ax = plt.gca()
        ax.set_xticklabels(column_name[sort_idx[:watch_rank]], rotation=90)
        plt.ylabel('importance')
        plt.xlabel('column index')
        plt.rcParams["font.size"] = font_size
        plt.grid(True)
    elif (draw_mode == 1):
        plt.bar(np.arange(len(importance_mean)), importance_mean, color=cmap(0), alpha=0.5)
        plt.ylabel('importance')
        plt.xlabel('column index')
        plt.rcParams["font.size"] = font_size
        plt.grid(True)
    elif (draw_mode == 2):
        plt.imshow(importance, interpolation='nearest', aspect='auto', vmin=np.min(importance), vmax=np.max(importance))
        plt.colorbar()
        plt.title('importance of several learning')
        plt.ylabel('model x cv index')
        plt.xlabel('column index')
        plt.rcParams["font.size"] = font_size
        plt.grid(True)
    ############################################################


def dump_value(csv_filename   = None,
               df_in          = None,
               head_num       = 30, 
               log_filename   = './memo.md', 
               not_dig_column = []):

    if ((csv_filename is None) & (df_in is None)):
        # 
        print('Data not found... Please set csv_filename or df_in')
        return False

    if (df_in is None):
        # data load from csv file
        df_in = pd.read_csv(csv_filename)

    # get value and column
    value_tmp  = df_in.values
    column_tmp = np.array(df_in.columns)

    # adjust
    not_dig_column = np.array(not_dig_column)

    # delete log file
    os.system('rm %s' % log_filename)

    ###############################################################
    with open(log_filename, mode='a') as f:
        f.write('data contents is ...\n\n')
        for column_i in range(len(column_tmp)):
            f.write(('|%s' % column_tmp[column_i]))
        f.write('|\n')
        for column_i in range(len(column_tmp)):
            f.write('|:-----')
        f.write('|\n')
        for row_i in range(np.min([head_num, len(value_tmp)])):
            for column_i in range(np.shape(value_tmp)[1]):
                f.write('|%s' % value_tmp[row_i, column_i])
            f.write('|\n')
        f.write('\n\n\n')

        for column_i in range(len(column_tmp)):
            if (column_tmp[column_i] in not_dig_column):
                f.write('%s column\'s value pattern do not dig\n\n\n' % (column_tmp[column_i]))
            else:
                value_count_tmp  = df_in.iloc[:, column_i].value_counts(dropna=False)

                f.write('%s column\'s value pattern\n\n' % (column_tmp[column_i]))
                f.write('|(%s)' % column_tmp[column_i])
                for idx_i in range(len(value_count_tmp.index)):
                    f.write('|%s' % value_count_tmp.index[idx_i])
                f.write('|\n')
                f.write('|:-----')
                for idx_i in range(len(value_count_tmp.index)):
                    f.write('|:-----')
                f.write('|\n')
                f.write('|(counts)')
                for idx_i in range(len(value_count_tmp.index)):
                    f.write('|%s' % value_count_tmp.values[idx_i])
                f.write('|\n')
                f.write('\n')
        f.write('\n\n\n')
    ###########################################################

    return True

# coding: utf-8

"""
PPAP : Pre-Processing And Prediction
"""

# Author: Taketo Kimura <mucun.wuxian@gmail.com>
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
from scipy.ndimage                   import gaussian_filter
from scipy.ndimage.interpolation     import map_coordinates

STR_NAN = '(nan)' 

MCLASS_DEFINE = 50

# 
# import os
# import pickle
# import cloudpickle
# import sys
import cv2
# import base64
# import json
# import requests
# import gc
# import Levenshtein

import numpy               as np
# import matplotlib.pyplot   as plt
# import japanize_matplotlib

# from datetime              import datetime
# from datetime              import timedelta

# from sklearn.linear_model  import LogisticRegression
# from sklearn.ensemble      import RandomForestClassifier
# from lightgbm              import LGBMClassifier
# from catboost              import CatBoostClassifier

# from sklearn.metrics       import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
# from sklearn.datasets      import fetch_mldata

# np.random.seed(0)

# import warnings
# warnings.filterwarnings('ignore')


# 
def imread_gray(filename):
    # 
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 
    return img


# 
def imread_RGB(filename):
    # 
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 
    return img


# [仕様rough]
#   - scaleとpixelの同時入力は、scaleが優先される 
#   - heightと、widthの何れかに値が無い場合、aspect比を保持しながらのresizeを行う
def imresize(img, height_scale=None, width_scale=None, height_pixel=None, width_pixel=None, clip_0_255=True):
    # 
    if (((height_scale is not None) | (width_scale is not None)) &  
        ((height_pixel is not None) | (width_pixel is not None))):
        # 
        print('adopted scale (reject pixel)')
    # 
    if ((height_scale is not None) | (width_scale is not None)): 
        # 
        if ((height_scale is not None) & (width_scale is None)):
            # 
            width_scale = height_scale
        # 
        if ((height_scale is None) & (width_scale is not None)):
            # 
            height_scale = width_scale
        # 
        height_pixel = int(np.shape(img)[0] * height_scale)
        width_pixel  = int(np.shape(img)[1] * width_scale)
    else:
        # 
        if ((height_pixel is not None) & (width_pixel is None)):
            # 
            width_pixel = int(np.round(height_pixel * (np.shape(img)[1] / np.shape(img)[0])))
        # 
        if ((height_pixel is None) & (width_pixel is not None)):
            # 
            height_pixel = int(np.round(width_pixel * (np.shape(img)[0] / np.shape(img)[1])))
    # 
    img = cv2.resize(img, (width_pixel, height_pixel), interpolation=cv2.INTER_CUBIC)
    # 
    if (clip_0_255):
        img = np.clip(img, 0, 255)
    #     
    return img


# 
def impadding_mirror(img, padding_vert, padding_horz=None, decay=None):
    # 
    if (decay is None):
        # 
        decay = 0.0
    # 
    if (padding_horz is None):
        padding_horz = padding_vert
    # 
    for padding_i in range(np.max([padding_vert, padding_horz])):
        # 
        if (padding_i < padding_vert):
            # 
            img_padding_top    = (img[0, :][np.newaxis, :]  + ((255 - img[0, :][np.newaxis, :])  * decay)).astype(np.uint8)
            img_padding_bottom = (img[-1, :][np.newaxis, :] + ((255 - img[-1, :][np.newaxis, :]) * decay)).astype(np.uint8)
            # 
            img = np.concatenate([img_padding_top, 
                                  img, 
                                  img_padding_bottom], axis=0)
        # 
        if (padding_i < padding_horz):
            # 
            img_padding_left   = (img[:, 0][:, np.newaxis]  + ((255 - img[:, 0][:, np.newaxis])  * decay)).astype(np.uint8)
            img_padding_right  = (img[:, -1][:, np.newaxis] + ((255 - img[:, -1][:, np.newaxis]) * decay)).astype(np.uint8)
            # 
            img = np.concatenate([img_padding_left, 
                                  img, 
                                  img_padding_right], axis=1)
    # 
    return img.astype(np.uint8)


# 
def imtrimming(img, threshold=150, img_size_min=5): # for gray
    # 
    img_height   = np.shape(img)[0]
    img_width    = np.shape(img)[1]
    # 
    adjust_ratio = np.zeros([4], dtype=np.float)
    #     
    while (True):
        # 
        process_flg = False
        # 1st, top
        if (np.min(img[0, :])  > threshold) & (np.shape(img)[0] > img_size_min):
            # 
            img              = img[1:, :]
            adjust_ratio[0] += 1 / img_height
            process_flg      = True
        # 2nd, left
        if (np.min(img[:, 0])  > threshold) & (np.shape(img)[1] > img_size_min):
            # 
            img              = img[:, 1:]
            adjust_ratio[1] += 1 / img_width
            process_flg      = True
        # 3rd, bottom
        if (np.min(img[-1, :]) > threshold) & (np.shape(img)[0] > img_size_min):
            # 
            img              = img[:-1, :]
            adjust_ratio[2] += 1 / img_height
            process_flg      = True
        # 4th, right
        if (np.min(img[:, -1]) > threshold) & (np.shape(img)[1] > img_size_min):
            # 
            img              = img[:, :-1]
            adjust_ratio[3] += 1 / img_width
            process_flg      = True
        # 
        if (process_flg == False):
            break
    # 
    return (img, adjust_ratio)


# 
def add_gaussian_noise(img, mean=0, sigma=10, roughness=1):
    # 
    img_height = np.shape(img)[0]
    img_width  = np.shape(img)[1]
    noise      = np.random.normal(mean, sigma, (int(img_height/roughness), int(img_width/roughness)))
    noise      = imresize(noise, height_pixel=img_height, width_pixel=img_width, clip_0_255=False)
    img_tmp    = img + noise
    img_tmp    = np.clip(img_tmp, 0, 255).astype(np.uint8)
    # 
    return img_tmp


# 
def add_gaussian_noise_and_blur(img, 
                                noise_mean=0, noise_sigma=10, noise_roughness=1, 
                                blur_sigma=1.0):
    # 
    img_tmp = img
    #
    window_size_vert = np.shape(img_tmp)[0]
    # 
    roulette = np.random.rand(1)
    if (roulette < 0.1):
        roulette = int((np.random.rand(1) * 3) + 1)
        img_tmp  = cv2.blur(img_tmp, (roulette, roulette)) 
    elif (roulette < 0.9):
        roulette = int(np.round(np.random.rand(1) * 3))
        for resize_i in range(roulette):
            img_tmp  = add_gaussian_noise(img_tmp, mean=noise_mean, 
                                                   sigma=noise_sigma, 
                                                   roughness=noise_roughness)
            roulette = np.random.rand(1)
            if (roulette < 0.5):
                img_tmp = gaussian_filter(img_tmp, sigma=blur_sigma)
    # 
    return img_tmp

# image augmantation (1⇔255対応要)
def imaugment_for_char(img, window_size_vert, 
                       color_adjust=True, 
                       add_noise=True, 
                       elastic_dist=True, 
                       eps=1e-8):
    # 
    img_aug = img.copy()
    # 
    for aug_i in range(len(img_aug)):
        # 
        img_aug_tmp = img_aug[aug_i]
        
        # ----- augmeantation start -----
        # 
        
        # (1) color adjust
        if (color_adjust == True):
            roulette = np.random.rand(1)
            if (roulette < 0.8):
                roulette    = np.random.rand(1)
                bottom_lumi = roulette * 50
                roulette    = np.random.rand(1)
                top_lumi    = 200 + (roulette * 55)
                img_aug_tmp = img_aug_tmp / (np.max(img_aug_tmp) + eps)
                img_aug_tmp = img_aug_tmp * (top_lumi - bottom_lumi)
                img_aug_tmp = img_aug_tmp + bottom_lumi
                img_aug_tmp = np.clip(img_aug_tmp, 0, 255)
        
        # (2) resize
        # print(np.shape(img_aug_tmp))
        img_aug_tmp = imresize(img_aug_tmp, height_pixel=window_size_vert)
        
        # (3) add noise and blur
        if (add_noise == True):
            roulette = np.random.rand(1)
            if (roulette < 0.1):
                roulette    = int((np.random.rand(1) * 3) + 1)
                img_aug_tmp = cv2.blur(img_aug_tmp, (roulette, roulette)) 
            elif (roulette < 0.9):
                roulette = 1 + int(np.round(np.random.rand(1) * 2))
                for resize_i in range(roulette):
                    roulette    = int((window_size_vert / 4) + (np.random.rand(1) * (window_size_vert / 4)))
                    img_aug_tmp = imresize(img_aug_tmp, height_pixel=roulette)
                    img_aug_tmp = add_gaussian_noise(img_aug_tmp, mean=0, sigma=(1 + (3 * np.random.rand(1))))
                    img_aug_tmp = imresize(img_aug_tmp, height_pixel=window_size_vert)

        # (5) elastic distortion
        if (elastic_dist == True):
            roulette = np.random.rand(1)
            if (roulette < 0.3):
                img_aug_tmp = img_aug_tmp[:, :, np.newaxis]
                img_aug_tmp = elastic_transform(img_aug_tmp, alpha=60, sigma=3)
                img_aug_tmp = img_aug_tmp[:, :, 0]
            if (roulette < 0.5):
                img_aug_tmp = img_aug_tmp[:, :, np.newaxis]
                img_aug_tmp = elastic_transform(img_aug_tmp, alpha=100, sigma=3)
                img_aug_tmp = img_aug_tmp[:, :, 0]

        # 
        # ----- augmeantation  end  -----
        
        # 
        img_aug_tmp    = np.clip(img_aug_tmp, 0, 255)
        img_aug[aug_i] = img_aug_tmp
    
    # 
    return img_aug


# refer from [https://gist.github.com/erniejunior/601cdf56d2b424757de5]
# from scipy.ndimage.filters import gaussian_filter

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), 
                         sigma, 
                         mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), 
                         sigma, 
                         mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)


# 
def split_image_from_grid(img_gray, 
                          threshold_white_and_black=150, 
                          sampling_y=5, 
                          sampling_x=5, 
                          margin=10, 
                          output_img_size=64, 
                          eps=1e-8, 
                          show_flg=True):
    # 
    if (show_flg == True):
        # 
        print('org image size = (%d, %d)' % np.shape(img_gray))
        # 
        plt.figure(figsize=(10,6),dpi=100)
        plt.imshow(img_gray)
        plt.show()

        # 
        plt.figure(figsize=(10,6),dpi=100)
        # 
        plt.subplot(2, 1, 1)
        plt.imshow(img_gray[sampling_y:(sampling_y + 1), :], clim=[0, 255])
        plt.axis('auto')
        plt.ylim([-0.5, 0.5])
        # 
        plt.subplot(2, 1, 2)
        plt.plot(img_gray[sampling_y, :])
        plt.xlim([0, np.shape(img_gray)[1]])
        plt.grid()
        plt.show()

    # 
    line_coord_x      = (np.where(img_gray[sampling_y, :] < threshold_white_and_black)[0])
    line_coord_x_diff = line_coord_x[1:] - line_coord_x[:-1]
    line_coord_x      = line_coord_x[np.where(line_coord_x_diff > margin)[0]]
    line_coord_x      = np.concatenate([line_coord_x, np.array([np.shape(img_gray)[1]])])

    # 
    if (show_flg == True):
        # 
        print(line_coord_x)
        print()
        # 
        plt.figure(figsize=(10,6),dpi=100)
        # 
        plt.subplot(1, 2, 1)
        plt.imshow(img_gray[:, sampling_x:(sampling_x + 1)], clim=[0, 255])
        plt.axis('auto')
        plt.xlim([-0.5, 0.5])
        # 
        plt.subplot(1, 2, 2)
        plt.plot(img_gray[:, sampling_x], np.arange(np.shape(img_gray)[0]))
        plt.ylim([0, np.shape(img_gray)[0]])
        plt.grid()
        plt.show()

    # 
    line_coord_y      = (np.where(img_gray[:, sampling_x] < threshold_white_and_black)[0])
    line_coord_y_diff = line_coord_y[1:] - line_coord_y[:-1]
    line_coord_y      = line_coord_y[np.where(line_coord_y_diff > margin)[0]]
    line_coord_y      = np.concatenate([line_coord_y, np.array([np.shape(img_gray)[0]])])

    # 
    if (show_flg == True):
        # 
        print(line_coord_y)
        print()
    # 
    img_stock          = [([np.nan] * (len(line_coord_x) - 1)) for i in range(len(line_coord_y) - 1)]
    aspect_ratio_stock = [([np.nan] * (len(line_coord_x) - 1)) for i in range(len(line_coord_y) - 1)]

    ########################################################
    # display processing progress 
    process_num   = len(line_coord_y) - 1 # set number of process
    process_break = np.round(np.linspace(1, process_num, 50)) 
    process_i     = 0  
    str_now       = datetime.now()
    print('pick up character [start time is %s]' % str_now) 
    print('--------------------------------------------------')
    print('START                                          END') 
    print('----+----1----+----2----+----3----+----4----+----5') 
    ########################################################

    # loop of vert grid
    for line_coord_y_i in range(len(line_coord_y) - 1):

        ####################################################
        # update processing progress
        process_i = process_i + 1   
        if (sum(process_break == process_i) > 0):
            for print_i in range(sum(process_break == process_i)): 
                print('*', end='', flush=True)                              
        ####################################################

        # 
        img_height_max = 0
        
        # loop of horz grid
        for line_coord_x_i in range(len(line_coord_x) - 1):

            # 
            coord_top    = line_coord_y[line_coord_y_i] + margin
            coord_left   = line_coord_x[line_coord_x_i] + margin
            coord_bottom = line_coord_y[line_coord_y_i + 1] - margin
            coord_right  = line_coord_x[line_coord_x_i + 1] - margin
            # 
            coord_top    = int(np.round(coord_top))
            coord_left   = int(np.round(coord_left))
            coord_bottom = int(np.round(coord_bottom))
            coord_right  = int(np.round(coord_right))
            # 
            img_crop     = img_gray[coord_top:coord_bottom, coord_left:coord_right]
            img_crop     = (img_crop - np.min(img_crop)) / (np.max(img_crop) - np.min(img_crop)) * 255
            # 
            (img_crop, adjust_ratio_tmp) = imtrimming(img_crop)
            # 
            img_stock[line_coord_y_i][line_coord_x_i] = img_crop
            # 
            if (img_height_max < np.shape(img_crop)[0]):
                # 
                img_height_max = np.shape(img_crop)[0]
        
        # loop of horz grid
        for line_coord_x_i in range(len(line_coord_x) - 1):

            # 
            img_crop = img_stock[line_coord_y_i][line_coord_x_i]
            img_crop = impadding_mirror(img_crop, padding_vert=int(np.round((img_height_max - np.shape(img_crop)[0]) / 2)), 
                                                  padding_horz=0, decay=0.9) # メチャメチャ悩ましい… 一旦、ZEROで… augmentationの中に、横paddingを入れるか…
            # get aspect ratio
            aspect_ratio_tmp  = np.shape(img_crop)[1] / np.shape(img_crop)[0]
            bins_tmp          = np.arange(0, 1.5+eps, 0.1)
            bins_tmp[-1]      = np.inf
            aspect_ratio_tmp_ = np.zeros([len(bins_tmp) - 1])
            for bins_i in range(len(bins_tmp) - 1):
                aspect_ratio_tmp_[bins_i] = float((bins_tmp[bins_i] <= aspect_ratio_tmp) & 
                                                  (aspect_ratio_tmp <  bins_tmp[bins_i + 1]))
            # 
            img_crop = imresize(img_crop, height_pixel=output_img_size, 
                                          width_pixel=output_img_size)
            # 
            img_stock[line_coord_y_i][line_coord_x_i]          = img_crop
            aspect_ratio_stock[line_coord_y_i][line_coord_x_i] = np.concatenate([np.array([aspect_ratio_tmp]), 
                                                                                 aspect_ratio_tmp_])

    ########################################################
    str_now = datetime.now()  
    print('')
    print('[end time is %s]' % (str_now)) 
    print('') 
    plt.show()
    ########################################################

    return (img_stock, aspect_ratio_stock)


# bad performance ...
def pick_image_randomly(img_gray, 
                         pick_num, window_size, 
                         resize_scale=np.arange(0.5, (3.0 + 0.00001), 0.1)):
    # 
    pick_i = 0

    # 
    img_pick = np.zeros([pick_num, window_size, window_size])

    # 
    while (True):
        # 
        for resize_i in np.random.permutation(np.arange(len(resize_scale))):
            for resize_j in np.random.permutation(np.arange(len(resize_scale))):

                # 
                img_resize = imresize(img=img_gray, height_scale=resize_scale[resize_i], 
                                                    width_scale=resize_scale[resize_j])
                # 
                y_idx = np.arange(0, (np.shape(img_resize)[0] - window_size), 1)
                x_idx = np.arange(0, (np.shape(img_resize)[1] - window_size), 1)
                # 
                y_i   = np.random.permutation(y_idx)[0]
                x_i   = np.random.permutation(x_idx)[0]
                # 
                img_tmp = img_resize[y_i:(y_i + window_size), 
                                     x_i:(x_i + window_size)]
                # 
                if (pick_i < pick_num):
                    img_pick[pick_i, :, :] = img_tmp
                    pick_i += 1
                else:
                    break

            # 
            if (pick_i >= pick_num):
                break
        # 
        if (pick_i >= pick_num):
            break
    # 
    return img_pick


# 
def reject_gap(img_gray, 
               search_range=[0.1, 0.9], threshold_gap=150, 
               gap_width=1):
    
    # 
    img_height = np.shape(img_gray)[0]
    img_width  = np.shape(img_gray)[1]

    # 
    img_stock  = []
    
    # 
    gap_search_from = int(np.shape(img_gray)[1] * search_range[0])
    gap_search_to   = int(np.shape(img_gray)[1] * search_range[1])
    
    # 
    from_the_begginning_flg = True

    # 
    while (True):
        
        # 
        flg_find_gap = False
        # 
        if (from_the_begginning_flg == True):
            # 
            gap_search_idx = np.arange(gap_search_from, gap_search_to, gap_width)
        # 
        else:
            # 
            gap_search_idx = np.arange(gap_search_to, gap_search_from, -gap_width)
        
        # 
        for gap_search_i in gap_search_idx:
            # 
            if (np.min(img_gray[:, gap_search_i:(gap_search_i + gap_width)]) > threshold_gap):
                # 
                flg_find_gap  = True
                # 
                img_gray      = np.concatenate([img_gray[:, :gap_search_i], 
                                                img_gray[:, (gap_search_i + gap_width):]], axis=1)
                # plt.imshow(img_gray)
                # plt.show()
                img_gray      = imresize(img_gray, height_pixel=img_height, 
                                                   width_pixel=img_width)
                (img_gray, _) = imtrimming(img_gray)
                img_gray      = imresize(img_gray, height_pixel=img_height, 
                                                   width_pixel=img_width)
        
                # 
                img_stock.append(img_gray)

                #
                from_the_begginning_flg = ~from_the_begginning_flg

                # 
                break
                
        # 
        if (flg_find_gap == False):
            # 
            if (gap_width == 1):
                # 
                break
            # 
            else:
                # 
                gap_width -= 1
    
    # 
    return img_stock


# 
def imcrop(img, crop_height, crop_width, 
                crop_top=None, crop_left=None):
    # 
    img_height = np.shape(img)[0]
    img_width  = np.shape(img)[1]

    # 
    if (crop_top is None):
        # random crop at vertical
        crop_top = np.random.randint(0, (img_height - crop_height))
    # 
    if (crop_left is None):
        # random crop at horizontal
        crop_left = np.random.randint(0, (img_width - crop_width))

    # 
    crop_bottom = crop_top  + crop_height
    crop_right  = crop_left + crop_width
    # 
    img_crop    = img[crop_top:crop_bottom, 
                      crop_left:crop_right]
    # 
    return (img_crop, crop_top, crop_left)

                
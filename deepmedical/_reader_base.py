######################################################################
## Deep Larning for Medical Image analysis
## - Medical image readers & patch generator
##
## -- Base Reader
## ---- ImageReader
## ------ SegmentationReader
## ---- ImageSimpleReader
## 
## Aug. 26, 2019
## Youngwon (youngwon08@gmail.com)
######################################################################
import os
import sys
import json
import timeit
import copy
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import scale, minmax_scale, robust_scale
import skimage.measure
from skimage.util.shape import view_as_windows

import multiprocessing as mp
from multiprocessing import Pool, Manager, Process
from functools import partial

from . import logging_daily
from .utils import file_size, convert_bytes, print_sysinfo
# from keras.utils import to_categorical

def normalize_none(img): return img
def normalize_scale(img): return (img - np.mean(img)) / np.std(img)
def normalize_minmax(img): return (img - np.min(img))/ (np.max(img) - np.min(img) + 1e-8)
def normalize_robustScale(img): return robust_scale(img)
def normalize_scale_2d(img): return scale(img)
def normalize_minmax_2d(img): return minmax_scale(img)
def normalize_robustScale_2d(img): return robust_scale(img)
def get_gt_mask(gt, stride=8):
    if stride: return skimage.measure.block_reduce(gt, (1,stride,stride), np.max)
def normalize_ct_safe(img): 
    img[img<-1024] = -1024
    return (img - (-1024))/ (1024 - (-1024))

########################################################################################################
# Base reader
########################################################################################################
class BaseReader(object):
    """Inherit from this class when implementing new readers."""
    def __init__(self, log, path_info, network_info, verbose=True):
        self.log = log
        self.path_info = path_info
        self.network_info = network_info
        self.verbose = verbose
        self.x_list = None
        self.data_info = None
        self.x_path = None
        self.y_path = None
        self.img_shape = None
        self.normalizer = globals()[self.network_info['model_info']['normalizer'].strip()]
        self.mapIndex = [mapname.strip() for mapname in self.network_info['model_info']['map_index'].split(',')]    
    
        self.read_dataset(self.path_info['data_info']['data_path'])
    
    def read_dataset(self, data_path):
        raise NotImplementedError()
    
    def get_cv_index(self, nfold=5, random_state = 12):
        kf = KFold(n_splits=nfold, shuffle=True, random_state=random_state)
        return kf.split(range(self.x_list.shape[0]))
        
    def get_training_validation_index(self, idx, validation_size=0.2):
        return train_test_split(idx, test_size = validation_size)
    
    def get_x_list(self, train_index):
        return self.x_list[train_index]
    
    #########################################################################
    # funtions for augmentation
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
    #########################################################################
    def _perspective(im, max_padsize=24):
        """
        Warning: assume 2D multi channel images
        """
        # Perspective tranformation
        im = im.astype(np.float64)
        rows,cols, ch = im.shape
        padsize=np.random.choice(np.arange(max_padsize//4,max_padsize,max_padsize//4), 1)[0]
        pts1 = np.float32([[padsize,padsize],[rows-padsize,padsize],[padsize,cols-padsize],[rows-padsize,cols-padsize]])
        pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        im = cv2.warpPerspective(im,M,(rows,cols))
        return im.astype(np.float32)
    
    def get_augment(self, x):
        return x
    
########################################################################################################
# Image reader
########################################################################################################
### Image simple reader
########################################################################################################
class ImageSimpleReader(BaseReader):
    ### From preprocessed small data (small memory size
    def __init__(self, log, path_info, network_info, verbose=True):
        super(ImageSimpleReader,self).__init__(log, path_info, network_info, verbose)
        
    def read_dataset(self, data_path):
        self.log.info('-----------------------------------------------------------------------')
        self.log.info('Construct Dataset')
        self.log.info('-----------------------------------------------------------------------')
        self.log.info('Load patch data list')
        self.x_list = np.load('%s/x_list.npy' % data_path)
        self.data_info = self.x_list[:,0]
        self.x_path = '%s/x.npy' % (data_path)
        self.y_path = '%s/y.npy' % (data_path)
        self.log.info("    : Total %d person's medical images are available" % (len(self.data_info)))
        
        self.x = np.load(self.x_path) 
        ##### TODO: fix ################################
#         self.x = (self.x-127.5) / 127.5 # range between -1, 1
        self.x = self.x / 255.0 # range between 0, 1
        ################################################
        self.y = np.load(self.y_path) # range between 0, 1, should use sigmoid
        self.y = np.expand_dims(self.y, axis=-1)
        self.log.info("    : Total %d images are available" % (len(self.data_info)))
        
    def load_training_image_list(self, idxs):
        return self.x[idxs], self.y[idxs]
    
    def load_test_image_list(self, idxs):
        return self.x[idxs]
    
########################################################################################################
## Image basic reader
########################################################################################################
class ImageReader(BaseReader):
    ### From raw data
    def __init__(self, log, path_info, network_info, verbose=True):
        super(ImageReader,self).__init__(log, path_info, network_info, verbose)    
        
    def read_dataset(self, data_path):
        self.log.info('-----------------------------------------------------------------------')
        self.log.info('Construct Dataset')
        self.log.info('-----------------------------------------------------------------------')
        self.log.info('Load medical image data list')
        self.x_list = np.load('%s/x_list.npy' % data_path)   
#         try: self.x_list = self.x_list[np.random.choice(np.arange(self.x_list.shape[0]), int(self.network_info['model_info']['tot_patients'].strip()))]
        try: self.x_list = self.x_list[:int(self.network_info['model_info']['tot_patients'].strip())]
        except: pass
        self.data_info = self.x_list[:,0]
        self.x_path = self.x_list[:,1]
        self.y = np.expand_dims(self.x_list[:,2].astype(np.int32), axis=-1)
        self.log.info("    : Total %d person's medical images are available" % (len(self.data_info)))
        
    ########################################################################################################
    # Load images
    ########################################################################################################
    def load_training_image_list(self, idxs, parallel=0, verbose=1):
        # load images
        if parallel:
            pool = Pool(processes=parallel)
            medical_images = pool.map_async(np.load, self.x_path[idxs]).get()
            medical_images = np.array(medical_images).astype(np.float32)
            pool.close()
            pool.join()
        else:
            medical_images = []
            for i, file_info in enumerate(self.x_path[idxs]):
                medical_images.append(np.load(file_info).astype(np.float32))
                if (len(medical_images[-1].shape) == 3): medical_images[-1] = np.expand_dims(medical_images[-1], axis=-1)
            medical_images = np.array(medical_images).astype(np.float32)
        for i in range(len(idxs)):
            for m in range(len(self.mapIndex)):
                medical_images[i,:,:,:,m] = self.normalizer(medical_images[i,:,:,:,m])
                
        if verbose==2:
            self.log.info('Load images with time %fs' % (timeit.default_timer() - tic))
            self.log.info('    Image shape  : {}'.format(medical_images.shape[1:]))
            self.log.info('    Image number : %s' % medical_images.shape[0])
            self.log.info('    Y ratio : %s' % np.mean(self.y[idxs], axis=0))
            self.log.info('--------------------------')
        return medical_images, self.y[idxs]
    
    def load_y_list(self, idxs):
        return self.y[idxs]
    
    def load_test_image_list(self, idxs, parallel=0, verbose=1):
        # load images
        if parallel:
            pool = Pool(processes=parallel)
            medical_images = pool.map_async(np.load, self.x_path[idxs]).get()
            medical_images = np.array(medical_images).astype(np.float32)
            pool.close()
            pool.join()
        else:
            medical_images = []
            for i, file_info in enumerate(self.x_path[idxs]):
                medical_images.append(np.load(file_info).astype(np.float32))
                if (len(medical_images[-1].shape) == 3): medical_images[-1] = np.expand_dims(medical_images[-1], axis=-1)
            medical_images = np.array(medical_images).astype(np.float32)
        for i in range(len(idxs)):
            for m in range(len(self.mapIndex)):
                medical_images[i,:,:,:,m] = self.normalizer(medical_images[i,:,:,:,m])
                
        if verbose==2:
            self.log.info('Load images with time %fs' % (timeit.default_timer() - tic))
            self.log.info('    Image shape  : {}'.format(medical_images.shape[1:]))
            self.log.info('    Image number : %s' % medical_images.shape[0])
            self.log.info('--------------------------')
        return medical_images
    
    def get_augment(self, x):
        """
        Warning: assume 3D multi channel images
        """
        for i in range(x.shape[0]):
            if np.random.choice([0,1], size=1, p=[0.5,0.5])[0]: x[i] = np.array([self._perspective(zslice) for zslice in x[i]])
        return x
    
########################################################################################################
# Segmentation reader
########################################################################################################
### Segmentation reader for sampling patches
########################################################################################################
class SegmentationReader(ImageReader):
    ### From raw data
    ### TODO: think about the sampler.. it only works with the basePatchSampler
    def __init__(self, log, path_info, network_info, verbose=True):
        super(SegmentationReader,self).__init__(log, path_info, network_info, verbose)
        self.gtIndex = self.network_info['model_info']['gt_index'].strip()
        img_shape = [int(ishape.strip()) for ishape in self.network_info['model_info']['img_shape'].split(',')]
        self.img_shape = img_shape + [len(self.mapIndex)]
        self.gt_shape = img_shape
        
    def read_dataset(self, data_path):
        self.log.info('-----------------------------------------------------------------------')
        self.log.info('Construct Dataset')
        self.log.info('-----------------------------------------------------------------------')
        self.log.info('Load medical image data list')
        self.x_list = np.load('%s/x_list.npy' % data_path)   
#         try: self.x_list = self.x_list[np.random.choice(np.arange(self.x_list.shape[0]), int(self.network_info['model_info']['tot_patients'].strip()))]
        try: self.x_list = self.x_list[:int(self.network_info['model_info']['tot_patients'].strip())]
        except: pass
        self.data_info = self.x_list[:,0]
        self.x_path = self.x_list[:,1]
        self.y_path = self.x_list[:,2]
        self.log.info("    : Total %d person's medical images are available" % (len(self.data_info)))
        
    ########################################################################################################
    # Load images
    ########################################################################################################
    def get_probability_mask(self, gt_images, patch_size=(4,64,64), stride=8):
        gt_images = np.squeeze(gt_images, axis=-1)
        gt_shape = np.array(gt_images.shape[1:]) // (1,stride,stride)
        patch_size = np.array(patch_size) // (1,stride,stride)
         ##
        check_full = gt_shape == patch_size
        ##
        p_mask = np.zeros(gt_shape)
        mask_range = np.vstack((patch_size//2, np.array(gt_shape) - (patch_size// 2 + patch_size % 2))).transpose()
        # p_mask[mask_range[0][0]:mask_range[0][1], mask_range[1][0]:mask_range[1][1], mask_range[2][0]:mask_range[2][1]] = 1
        p_mask[mask_range[0][0]:mask_range[0][1]+check_full[0], mask_range[1][0]:mask_range[1][1]+check_full[1], mask_range[2][0]:mask_range[2][1]+check_full[2]] = 1
        p_masks = np.array([p_mask] * gt_images.shape[0])
        gt_masks = np.array([get_gt_mask(gt,stride=stride) for gt in gt_images])
        return p_masks, gt_masks
    
    def load_training_image_list(self, idxs, parallel=0, verbose=1):
        if parallel:
            pool = Pool(processes=parallel)
            medical_images = pool.map_async(np.load, self.x_path[idxs]).get()
            medical_images = np.array(medical_images).astype(np.float32)
            gt_images = pool.map_async(np.load, self.y_path[idxs]).get()
            gt_images = np.array(gt_images).astype(np.int32)
            pool.close()
            pool.join()
        else:
            medical_images = []
            gt_images = []
            for i, file_info in enumerate(zip(self.x_path[idxs], self.y_path[idxs])):
                medical_images.append(np.load(file_info[0]).astype(np.float32))
                gt_images.append(np.load(file_info[1]))
            medical_images = np.array(medical_images).astype(np.float32)
            gt_images = np.array(gt_images).astype(np.int32)
        for i in range(len(idxs)):
            for m in range(len(self.mapIndex)):
                medical_images[i,:,:,:,m] = self.normalizer(medical_images[i,:,:,:,m])
                
        gt_images = np.expand_dims(gt_images, axis=-1)
        if verbose==2:
            positive_ratio = pd.DataFrame(np.array(list(map(lambda gt: np.sum(gt)*100./np.prod(gt.shape), gt_images))),
                                          index= self.data_info[idxs], columns=['positive (%)'])
            self.log.info('Load images with time %fs' % (timeit.default_timer() - tic))
            self.log.info('    Image shape  : {}'.format(medical_images.shape[1:]))
            self.log.info('    Ground Truth  shape  : {}'.format(gt_images.shape))
            self.log.info('    Image number : %s' % medical_images.shape[0])
            self.log.info('--------------------------')
            self.log.info(positive_ratio.describe())
        return medical_images, gt_images
    
    def load_test_image_list(self, idxs, parallel=0, verbose=1):
        if parallel:
            pool = Pool(processes=parallel)
            medical_images = pool.map_async(np.load, self.x_path[idxs]).get()
            medical_images = np.array(medical_images).astype(np.float32)
            pool.close()
            pool.join()
        else:
            medical_images = []
            for i, file_info in enumerate(self.x_path[idxs]):
                medical_images.append(np.load(file_info).astype(np.float32))
            medical_images = np.array(medical_images).astype(np.float32)
        for i in range(len(idxs)):
            for m in range(len(self.mapIndex)):
                medical_images[i,:,:,:,m] = self.normalizer(medical_images[i,:,:,:,m])
                
        if verbose==2:
            self.log.info('Load images with time %fs' % (timeit.default_timer() - tic))
            self.log.info('    Image shape  : {}'.format(medical_images.shape[1:]))
            self.log.info('    Image number : %s' % medical_images.shape[0])
            self.log.info('--------------------------')
        return medical_images
    
    ########################################################################################################
    # Patch generator
    ########################################################################################################
    def get_loc(self, flat_pt, shape=(4,64,64)):
        return (flat_pt // np.prod(shape[1:]),
                (flat_pt % np.prod(shape[1:])) // np.prod(shape[2:]),
                (flat_pt % np.prod(shape[2:])))

    def get_img_loc(self, flat_pt, shape=(5,4,64,64), stride=8):
        ## TODO : Check
        value = np.array([flat_pt // np.prod(shape[1:]),
                (flat_pt % np.prod(shape[1:])) // np.prod(shape[2:]), #* stride + stride // 2,
                (flat_pt % np.prod(shape[2:])) // np.prod([shape[3:]]) * stride + stride // 2,
                flat_pt % np.prod(shape[3:]) * stride + stride // 2])
        return value.astype(np.int32)
    
    ## Generate Training patch
    def get_patch_x(self, pt_c, img, patch_size=(4,64,64)):
        patch_size = np.array(patch_size)
        pt_range = np.vstack((pt_c - patch_size // 2, pt_c + patch_size// 2 + patch_size % 2)).transpose()
        main_img = img[pt_range[0][0]:pt_range[0][1],
                       pt_range[1][0]:pt_range[1][1],
                       pt_range[2][0]:pt_range[2][1]]
        return main_img

    def get_patch_y(self, pt_c, img, patch_size=(4,64,64)):
        return self.get_patch_x(pt_c, img, patch_size)
    
#     ## Generate Validation & Test patch
#     def get_evauate_patch_x(self, img, patch_size=(4,64,64), step=(2,32,32)):
#         patch_size = np.array(patch_size)
#         main_img = view_as_windows(img, window_shape=np.append(patch_size, img.shape[-1]), step=np.append(step,[1]))
#         batch_idx_shape = main_img.shape[:3]
#         main_img_stack = main_img.reshape(np.append(np.prod(batch_idx_shape),np.append(patch_size, img.shape[-1])))
#         return main_img_stack, batch_idx_shape

#     def get_evauate_patch_y(self, img, patch_size=(4,64,64), step=(2,32,32)):
#         patch_size = np.array(patch_size)
#         main_img = view_as_windows(img, window_shape=patch_size, step=step)
#         batch_idx_shape = main_img.shape[:3]
#         main_img_stack = main_img.reshape(np.append(np.prod(batch_idx_shape),list(patch_size)+[1]))
#         return main_img_stack

#     def stack_patch_y(self, img, batch_idx, patch_size=(4,64,64)):
#         ## TODO : correct when patch step is not patch_size // 2
#         stacked_img = np.zeros(self.img_shape[:-1])
#         for idx, patch in enumerate(img):
#             i,j,k = batch_idx[idx]
#             stacked_img[i*(patch_size[0]//2):i*(patch_size[0]//2)+patch_size[0],
#                         j*(patch_size[1]//2):j*(patch_size[1]//2)+patch_size[1],
#                         k*(patch_size[2]//2):k*(patch_size[2]//2)+patch_size[2]] += patch
#         stacked_img[patch_size[0]//2:-(patch_size[0]//2+patch_size[0]%2),:,:] *= 0.5
#         stacked_img[:,patch_size[1]//2:-(patch_size[1]//2+patch_size[1]%2),:] *= 0.5
#         stacked_img[:,:,patch_size[2]//2:-(patch_size[2]//2+patch_size[2]%2)] *= 0.5
#         return stacked_img

    ## generate batch data
    def generate_training_batch_data(self, idxs, image, gt, patch_size=(4,64,64), stride=8, augment=False):
        # x = np.empty(shape=[len(idxs)] + list(patch_size) + [image.shape[-1]], dtype=np.float32)
        # y = np.empty(shape=[len(idxs)] + list(patch_size) + [1], dtype=np.int16)
        x = []
        y = []
        position = np.zeros(3, dtype=np.int)
        bd = np.array([np.array(patch_size)//2, np.array(image.shape[1:-1])-np.array(patch_size)//2])
        for i, pt in enumerate(idxs):
            if augment: # Zitter
                position = np.clip(pt[1:]+np.append([0],np.random.randint(-stride//4,stride//4,2)), bd[0], bd[1])
            else:
                position = pt[1:]
            # x[i] = self.get_patch_x(position, image[pt[0]], patch_size = patch_size)
            # y[i] = self.get_patch_y(position, gt[pt[0]], patch_size = patch_size)
            x.append(self.get_patch_x(position, image[pt[0]], patch_size = patch_size))
            y.append(self.get_patch_y(position, gt[pt[0]], patch_size = patch_size))
        #y = to_categorical(y, num_classes=2)
        x = np.array(x).astype(np.float32)
        y = np.array(y).astype(np.int32)
        return x, y
    
    def generate_evaluate_batch_data(self, image, gt, patch_size=(4,64,64)):
        # generate patches from one person per batch
        patch_size = np.array(patch_size)
        x, batch_idx_shape = self.get_evauate_patch_x(image, patch_size, step=patch_size//2)
        y = self.get_evauate_patch_y(gt, patch_size, step=patch_size)
        #y = to_categorical(y, num_classes=2)
        return x, y
    
    def generate_predict_batch_data(self, image, patch_size=(4,64,64)):
        # generate patches from one person per batchpatch_size = np.array(patch_size)
        patch_size = np.array(patch_size)
        return self.get_evauate_patch_x(image, patch_size, step=patch_size//2)
    
    def get_augment(self, x):
        """
        Warning: assume 3D multi channel images
        """
        for i in range(x.shape[0]):
            ## TODO: check, segital axis flipping
            if np.random.choice([0,1], size=1, p=[0.5,0.5])[0]: x[i] = x[i,:,:,::-1]
        return x
        
########################################################################################################
# Test : TODO fix
########################################################################################################
# if __name__ == "__main__":
    
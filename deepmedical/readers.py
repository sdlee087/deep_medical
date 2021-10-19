######################################################################
## Deep Larning for Medical Image analysis
## - Medical image readers & patch generator
##
## -- Base Reader
## ---- ImageReader
## ------ SegmentationReader
## ---- ImageSimpleReader
## ------ SegmentationSimpleReader
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

from scipy.ndimage import zoom

import multiprocessing as mp
from multiprocessing import Pool, Manager, Process
from functools import partial

from . import logging_daily
from .utils import file_size, convert_bytes, print_sysinfo
# from keras.utils import to_categorical

from . import _reader_base
from ._reader_base import ImageSimpleReader, ImageReader, SegmentationReader

########################################################################################################    
# IPF segmentation reader
########################################################################################################    
class IPFSegmentationReader(ImageSimpleReader):
    ### From preprocessed data
    def __init__(self, log, path_info, network_info, verbose=True):
        super(IPFSegmentationReader,self).__init__(log, path_info, network_info, verbose)
        
    def read_dataset(self, data_path):
        self.log.info('-----------------------------------------------------------------------')
        self.log.info('Construct Dataset')
        self.log.info('-----------------------------------------------------------------------')
        self.log.info('Load patch data list')
        self.x_list = np.load('%s/x_list.npy' % '/'.join(data_path.split('/')[:-1]))
        self.x_path = '%s/x.npy' % (data_path)
        self.y_path = '%s/y.npy' % (data_path)
        self.log.info("    : Total %d person's medical images are available" % (len(self.x_list)))
        
        self.x = np.load(self.x_path) 
        ##### TODO: fix ################################
#         self.x = (self.x-127.5) / 127.5 # range between -1, 1
        self.x = self.x / 255.0 # range between 0, 1
        ################################################
        self.y = np.load(self.y_path) # range between 0, 1, should use sigmoid
        self.y = np.expand_dims(self.y, axis=-1)
        self.patch_info = np.load('%s/patches_info.npy' % (data_path))
        self.x_list = np.arange(self.y.shape[0])
        self.log.info("    : Total %d patches are available" % (len(self.x_list)))
        
    def get_patch_info(self):
        return self.patch_info

########################################################################################################    
# IPF prediction reader
######################################################################################################## 
IPFPredictionReader = ImageReader
######################################################################################################## 
### IPF prediction reader with prior knowledge
######################################################################################################## 
class IPFPredictionGuidedAttentionReader(ImageReader):
    ### From preprocessed data
    def __init__(self, log, path_info, network_info, verbose=True):
        super(IPFPredictionGuidedAttentionReader,self).__init__(log, path_info, network_info, verbose)
        
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
        
        info_fibro = np.load('%s/info_fibro.npy' % data_path).astype(np.float64)
        info_otherlf = np.load('%s/info_otherlf.npy' % data_path).astype(np.float64)

        qt_fibro = info_fibro / np.max(info_fibro)
        qt_otherlf = info_otherlf / np.max(info_otherlf)

    #         prior_info = (qt_fibro + qt_otherlf) * 0.5
        prior_info = np.max(np.stack([qt_fibro,qt_otherlf]), axis=0)
#         prior_info = (np.max(np.stack([qt_fibro,qt_otherlf]), axis=0)> 0).astype(np.float32)

        ## TODO: Fix to find resize scale! automatically!!!!!
        self.prior_info = np.array([zoom(prior_info, zoom=(1/4.,1/4.,1/4.), order=1).astype(np.float32)])
        self.log.info("    : Total %d person's medical images are available" % (len(self.data_info)))

    def get_prior_info(self):
        return self.prior_info
    
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
        return [medical_images, np.repeat(np.expand_dims(self.prior_info, axis=-1), len(idxs), axis=0)], [self.y[idxs], np.zeros_like(self.y[idxs])]
        
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
        return [medical_images, np.repeat(np.expand_dims(self.prior_info, axis=-1), len(idxs), axis=0)]
    
    def get_augment(self, x):
        """
        Warning: assume 3D multi channel images
        """
        for i in range(x[0].shape[0]):
            if np.random.choice([0,1], size=1, p=[0.5,0.5])[0]: x[0][i] = np.array([self._perspective(zslice) for zslice in x[0][i]])
        return x
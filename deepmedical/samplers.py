######################################################################
## Deep Larning for Medical Image analysis
## - train_generator
## - BaseSampler
## ---- UniformSampler
## ---- OverSamplingSampler
## ---- BasePatchSampler (?)
## ------ UniformPatchSampler
## ------ OverSamplingPatchSampler
##
## Aug. 26, 2019
## Youngwon (youngwon08@gmail.com)
######################################################################
import os
import sys
import timeit
import numpy as np
import pandas as pd
import copy

import multiprocessing as mp
import itertools
from multiprocessing import Pool, Manager
from functools import partial

from keras.utils import Sequence

from . import logging_daily
from . import readers
from .utils import convert_bytes

#########################################################################################################################
# Generator
#########################################################################################################################
class train_generator(Sequence):
    """
    keras.utils.Sequence class was used to avoid duplicating data to multiple workers
    Sampler : instanse of BaseSampler class or BaseSampler inheritted Sampler class
    """
    def __init__(self, sampler):
        self.sampler = sampler
        self.batch_size = sampler.get_batch_size()
        self.steps = sampler.get_steps()
        self.sampler.on_training_start()
            
    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        return self.sampler.load_batches(idx)
    
    def on_epoch_end(self):
        self.sampler.on_epoch_end()

#########################################################################################################################
# Sampler
#########################################################################################################################
class BaseSampler():
    """
    Probability based Sampler for fit_generator and data generator which inherited keras.utils.Sequence
    Inherit from this class when implementing new probability based sampling.
    
    Example
    =================
    TODO
    """ 
    def __init__(self, log, train_idx, reader, training_info, parallel=0, verbose=True):
        self.log = log
        self.train_idx = train_idx
        self.sample_id = None
        self.reader = reader
        self.parallel = parallel
        self.verbose = verbose
        self.batch_size = int(training_info['batch_size'])
        self.log.info("    : Total %d images are available" % self.train_idx.shape[0])
        
        if 'augment' in training_info: 
            if training_info['augment'] == 'True': self.augment = True
            else: self.augment = False
        else: self.augment = False
        
        try:
            if training_info['replace'] == 'True': self.replace = True
            else: self.replace = False
        except: self.replace = False
            
        try:
            if training_info['sequential'] == 'True': self.sequential = True
            else: self.sequential = True
        except: self.sequential = False
            
        try: self.subsets_per_epoch = int(training_info['subsets_per_epoch'])
        except: self.subsets_per_epoch = None
        
        if training_info['steps_per_epoch'] == 'None': 
            if self.subsets_per_epoch == None: self.steps_per_epoch = np.ceil(self.train_idx.shape[0]/self.batch_size).astype(np.int)
            else: self.steps_per_epoch = np.ceil(self.subsets_per_epoch/self.batch_size).astype(np.int)
        else: self.steps_per_epoch = int(training_info['steps_per_epoch'])       
        
        try:
            if training_info['fix_sets'].strip() == 'True': self.fix_sets = True
            else: self.fix_sets = False
        except: self.fix_sets = False
        
        self.probability = None
        self.epoch_idxs = None
        self.epoch_probability_idxs = None
        
        if training_info['mode'] == 'training':
            self.load_batches = self._load_training_batches
        elif training_info['mode'] == 'evaluation':
            self.load_batches = self._load_training_batches
        elif training_info['mode'] == 'test':
            self.load_batches = self._load_test_batches
        else: raise ValueError('Mode error : no functions for mode %s' % training_info['mode']) 
    
    def set_epoch_idxs(self):
        if self.subsets_per_epoch != None: 
            self.epoch_idxs = np.random.choice(self.train_idx, self.subsets_per_epoch, replace=False)
        else: self.epoch_idxs = self.train_idx
        self.epoch_probability_idxs = np.arange(self.epoch_idxs.shape[0])
            
    def set_probability_vector(self, y):
        """
        Probability Vector for Probability sampling
        """
        raise NotImplementedError()
        
    def probability_sampling(self, idxs, size, replace=True):
        """
        Probability based Index Sampling Functions
        probability : The probabilities associated with each entry in a. 
                      If not given the sample assumes a uniform distribution over all entries in a.
        """
        try:
            # np.random.seed(seed=seed)
            return np.random.choice(idxs, size=size, replace=replace, p=self.probability)
        except Exception as e:
            self.log.error(e)
            raise SystemExit
    
    def get_proability_vector(self):
        return self.probability
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_steps(self):
        return self.steps_per_epoch

    def get_epoch_idxs(self):
        return self.epoch_idxs
    
    def get_physical_id(self, idxs):
        return self.reader.get_x_list(idxs)[:,0]
    
    def get_current_physical_id(self):
        return self.current_physical_id
    
    #######################################################################################
    def on_training_start(self):
        self.set_epoch_idxs()
        self.x, self.y = self.reader.load_training_image_list(self.epoch_idxs, min(self.parallel, len(self.epoch_idxs)), verbose=self.verbose)
        if self.sequential:
            np.random.shuffle(self.epoch_probability_idxs)
        else: 
            self.set_probability_vector(self.y[self.epoch_probability_idxs])
        
    def on_epoch_end(self):
        if self.sequential:
            np.random.shuffle(self.epoch_probability_idxs)
        else:
            if self.fix_sets == False:
                self.set_epoch_idxs()
                self.x, self.y = self.reader.load_training_image_list(self.epoch_idxs, 
                                                                      min(self.parallel, len(self.epoch_idxs)), verbose=self.verbose)
                self.set_probability_vector(self.y[self.epoch_probability_idxs])
            
    def _load_training_batches(self, i):
        if self.sequential: idxs = self.epoch_probability_idxs[i*self.batch_size:min((i+1)*self.batch_size, self.epoch_probability_idxs.shape[0])]
        else: idxs = self.probability_sampling(self.epoch_probability_idxs, self.batch_size, replace=self.replace)
        self.current_physical_id = self.get_physical_id(self.epoch_idxs[idxs])
        x, y = self.x[idxs], self.y[idxs]
        if self.augment: x = self.reader.get_augment(x)
        return x, y
    
    def _load_test_batches(self, i):
        if self.sequential: idxs = self.epoch_probability_idxs[i*self.batch_size:min((i+1)*self.batch_size, self.epoch_probability_idxs.shape[0])]
        else: idxs = self.probability_sampling(self.epoch_probability_idxs, self.batch_size, replace=self.replace)
        self.current_physical_id = self.get_physical_id(self.epoch_idxs[idxs])
        x = self.x[idxs]
        return x
    #######################################################################################
    
#########################################################################################################################
# Large Image Sampler
#########################################################################################################################
class LargeImageSampler(BaseSampler):
    """
    Probability based Sampler for fit_generator and data generator which inherited keras.utils.Sequence
    Inherit from this class when implementing new probability based sampling.
    
    Example
    =================
    TODO
    """ 
    def __init__(self, log, train_idx, reader, training_info, parallel=0, verbose=True):
        super(LargeImageSampler,self).__init__(log, train_idx, reader, training_info, parallel, verbose)
        
    def set_epoch_idxs(self):
        self.epoch_idxs = self.train_idx
        self.epoch_probability_idxs = np.arange(self.epoch_idxs.shape[0])
            
    #######################################################################################
    def on_training_start(self):
        self.set_epoch_idxs()
        self.y = self.reader.load_y_list(self.epoch_idxs)
        if self.sequential:
            np.random.shuffle(self.epoch_probability_idxs)
        else: 
            self.set_probability_vector(self.y[self.epoch_probability_idxs])
        
    def on_epoch_end(self):
        if self.sequential:
            np.random.shuffle(self.epoch_probability_idxs)
        else:
            self.set_probability_vector(self.y[self.epoch_probability_idxs])
            
    def _load_training_batches(self, i):
        if self.sequential: idxs = self.epoch_probability_idxs[i*self.batch_size:min((i+1)*self.batch_size, self.epoch_probability_idxs.shape[0])]
        else: idxs = self.probability_sampling(self.epoch_probability_idxs, self.batch_size, replace=self.replace)
        self.current_physical_id = self.get_physical_id(self.epoch_idxs[idxs])
        return self.reader.load_test_image_list(idxs, min(self.parallel, len(idxs)), verbose=self.verbose)
    
    def _load_test_batches(self, i):
        if self.sequential: idxs = self.epoch_probability_idxs[i*self.batch_size:min((i+1)*self.batch_size, self.epoch_probability_idxs.shape[0])]
        else: idxs = self.probability_sampling(self.epoch_probability_idxs, self.batch_size, replace=self.replace)
        self.current_physical_id = self.get_physical_id(self.epoch_idxs[idxs])
        return self.reader.load_test_image_list(idxs, min(self.parallel, len(idxs)), verbose=self.verbose)
    
    #######################################################################################
    
#########################################################################################################################
# Patch sampler for segmentation
#########################################################################################################################
class BasePatchSampler(BaseSampler):
    """
    Probability based Patch Sampler for fit_generator and data generator which inherited keras.utils.Sequence
    Inherit from this class when implementing new probability based sampling.
    
    Example
    =================
    TODO
    """
    def __init__(self, log, train_idx, reader, training_info, parallel=0, verbose=True):
        super(BasePatchSampler,self).__init__(log, train_idx, reader, training_info, parallel, verbose)
        
        if 'patch_size' in training_info: self.patch_size = np.array(training_info['patch_size'].split(',')).astype(int)
        if 'stride' in training_info: self.stride = int(training_info['stride'])
        
        if training_info['mode'] == 'training':
            self.load_batches = self._load_training_batches
            self.on_training_start = self._on_training_start
            self.on_epoch_end = self._on_training_epoch_end
        elif training_info['mode'] == 'evaluation':
            self.load_batches = self._load_evaluation_batches
            self.on_training_start = self._on_evaluation_start
            self.on_epoch_end = self._on_evaluation_epoch_end
        elif training_info['mode'] == 'test':
            self.load_batches = self._load_test_batches
            self.on_training_start = self._on_test_start
            self.on_epoch_end = self._on_evaluation_epoch_end
        else: raise ValueError('Mode error : no functions for mode %s' % training_info['mode']) 
            
#         try:
#             if training_info['filtered'] == 'True':
#                 self.log.debug('input_idx : %s', self.train_idx)
#                 self.train_idx_before_filtering = copy.deepcopy(self.train_idx)
#                 filter_thrd = float(training_info['filter_thrd'])
#                 fltr = np.max(self.reader.whole_gt_ratio[self.train_idx], axis=1) < filter_thrd
#                 self.log.info('total %d of %d images are filtered. (cut thrd : %d%%)', np.sum(~fltr), len(self.train_idx), filter_thrd)
#                 self.train_idx = self.train_idx[fltr]
#                 self.filtered_idx = self.train_idx_before_filtering[~fltr]
#                 self.log.debug('filtered input_idx : %s', self.train_idx)
#                 self.log.info('    filtered idx : %s', self.filtered_idx)
#         except: pass

    def set_probability_vector(self, p_masks, gt_masks):
        """
        Probability Vector for Probability sampling
        """
        raise NotImplementedError()
        
    #######################################################################################
    def _on_training_start(self):
        self.set_epoch_idxs()
        self.medical_images, self.gt_images = self.reader.load_training_image_list(self.epoch_idxs, min(self.parallel, len(self.epoch_idxs)), verbose=self.verbose)
        p_masks, gt_masks = self.reader.get_probability_mask(self.gt_images, self.patch_size, self.stride)
        self.gt_masks_shape = gt_masks.shape
        self.set_probability_vector(p_masks, gt_masks)
        
    def _on_training_epoch_end(self):
        self.set_epoch_idxs()
        self.medical_images, self.gt_images = self.reader.load_training_image_list(self.epoch_idxs, min(self.parallel, len(self.epoch_idxs)), verbose=self.verbose)
        p_masks, gt_masks = self.reader.get_probability_mask(self.gt_images, self.patch_size, self.stride)
        self.gt_masks_shape = gt_masks.shape
        self.set_probability_vector(p_masks, gt_masks)
        self.current_physical_id = self.get_physical_id(self.epoch_idxs)
    
    def _on_evaluation_epoch_end(self):
        self.set_epoch_idxs()
        
    def _on_evaluation_start(self):
        self.set_epoch_idxs()
        # self.medical_images, self.gt_images = self.reader.load_training_image_list(self.epoch_idxs, min(self.parallel, len(self.epoch_idxs)), verbose=self.verbose)
        
    def _on_test_start(self):
        self.set_epoch_idxs()
        # self.medical_images = self.reader.load_test_image_list(self.epoch_idxs, min(self.parallel, len(self.epoch_idxs)), verbose=self.verbose)
   
    def _load_training_batches(self, i):
        idxs = self.probability_sampling(range(self.probability.shape[0]), self.batch_size, replace=self.replace)
        idxs = np.array([self.reader.get_img_loc(idx, self.gt_masks_shape, stride=self.stride) for idx in idxs])
        x, y = self.reader.generate_training_batch_data(idxs, self.medical_images, self.gt_images, patch_size=self.patch_size, stride=self.stride, augment=self.augment)
        if self.augment: x = self.reader.get_augment(x)
        return x, y
    
    def _load_evaluation_batches(self, i):
        # to make evaluation per person's whole images
        if self.sequential: idxs = self.epoch_idxs[i*self.batch_size:min((i+1)*self.batch_size, self.epoch_idxs.shape[0])]
        else: idxs = self.probability_sampling(self.epoch_idxs, self.batch_size, replace=self.replace)
        self.current_physical_id = self.get_physical_id(idxs)
        return self.reader.load_training_image_list(idxs, 0, verbose=self.verbose)
        # return self.medical_images[idxs], self.gt_images[idxs]
    
    def _load_test_batches(self, i):
        # to make prediction per person's whole images
        if self.sequential: idxs = self.epoch_idxs[i*self.batch_size:min((i+1)*self.batch_size, self.epoch_idxs.shape[0])]
        else: idxs = self.probability_sampling(self.epoch_idxs, self.batch_size, replace=self.replace)
        self.current_physical_id = self.get_physical_id(idxs)
        return self.reader.load_test_image_list(idxs, 0, verbose=self.verbose)
        # return self.medical_images[idxs]
    
#     def _load_evaluation_batches(self, i):
#         self.set_epoch_idxs()
#         x_valid = []
#         y_valid = []
#         for i in range(self.mri_images.shape[0]):
#             x, y = self.reader.generate_evaluate_batch_data(self.mri_images[i], self.gt_images[i], patch_size=self.patch_size)
#             x_valid.append(x)
#             y_valid.append(y)
#         return np.vstack(x_valid), np.vstack(y_valid)
    
#     def _load_test_batches(self):
#         self.set_epoch_idxs()
#         x_test = []
#         for i in range(self.mri_images.shape[0]):
#             x, batch_idx_shape = self.reader.generate_predict_batch_data(self.mri_images[i], patch_size=self.patch_size)
#             x_test.append(x)
#         self.batch_idx = [self.reader.get_loc(pt, batch_idx_shape) for pt in range(len(x_test[-1]))] # TODO: fix. for stack_y
#         self.n_person = len(x_test)
#         return np.vstack(x_test)
    #######################################################################################
    
#########################################################################################################################
## Basic sampler: Uniform sampling / Oversampling / Labelsampling
#########################################################################################################################
class UniformSampler(BaseSampler):
    """
    Sampling from uniform distribution
    """
    def __init__(self, log, train_idx, reader, training_info, parallel=0, verbose=True):
        super(UniformSampler, self).__init__(log, train_idx, reader, training_info, parallel, verbose)
                    
    def set_probability_vector(self, y):
        """
        Probability Vector for Probability sampling
        sampling_distribution : 'uniform'
        """
        self.probability = np.ones(y.shape[0])
        self.probability = self.probability/np.sum(self.probability)
        return self.probability
    
class OverSamplingSampler(BaseSampler):
    """
    Oversampling from minarity class to deal with imbalance
    """
    def __init__(self, log, train_idx, reader, training_info, parallel=0, verbose=True):
        super(OverSamplingSampler, self).__init__(log, train_idx, reader, training_info, parallel, verbose)   
        
    def set_probability_vector(self, y):
        """
        Probability Vector for Probability sampling
        sampling_distribution : 'equiprobability'
        """
        self.probability = y * np.sum(1-y) + (1-y) * np.sum(y) # TODO: fix for multiclass y (np.sum(., axis=0)?)
        self.probability = self.probability / np.sum(self.probability)
        return self.probability
    
#########################################################################################################################
## LargeImageSampler: Uniform sampling / Oversampling / Labelsampling
#########################################################################################################################
class UniformLargeSampler(LargeImageSampler):
    """
    Sampling from uniform distribution
    """
    def __init__(self, log, train_idx, reader, training_info, parallel=0, verbose=True):
        super(UniformLargeSampler, self).__init__(log, train_idx, reader, training_info, parallel, verbose)
                    
    def set_probability_vector(self, y):
        """
        Probability Vector for Probability sampling
        sampling_distribution : 'uniform'
        """
        self.probability = np.ones(y.shape[0])
        self.probability = self.probability/np.sum(self.probability)
        return self.probability
    
class OverSamplingLargeSampler(LargeImageSampler):
    """
    Oversampling from minarity class to deal with imbalance
    """
    def __init__(self, log, train_idx, reader, training_info, parallel=0, verbose=True):
        super(OverSamplingLargeSampler, self).__init__(log, train_idx, reader, training_info, parallel, verbose)   
        
    def set_probability_vector(self, y):
        """
        Probability Vector for Probability sampling
        sampling_distribution : 'equiprobability'
        """
        self.probability = y * np.sum(1-y) + (1-y) * np.sum(y) # TODO: fix for multiclass y (np.sum(., axis=0)?)
        self.probability = self.probability / np.sum(self.probability)
        return self.probability
    
#########################################################################################################################
## Patch sampler: Uniform sampling / Oversampling / Labelsampling
#########################################################################################################################
class UniformPatchSampler(BasePatchSampler):
    """
    Sampling from uniform distribution
    """
    def __init__(self, log, train_idx, reader, training_info, parallel=0, verbose=True):
        super(UniformPatchSampler, self).__init__(log, train_idx, reader, training_info, parallel, verbose)
                    
    def set_probability_vector(self, p_masks, gt_masks):
        """
        Probability Vector for Probability sampling
        sampling_distribution : 'uniform'
        """
        self.probability = np.copy(p_masks)
        self.probability = self.probability.flatten()
        self.probability = self.probability/np.sum(self.probability)
        return self.probability
    
class OverSamplingPatchSampler(BasePatchSampler):
    """
    Oversampling from minarity class to deal with imbalance
    """
    def __init__(self, log, train_idx, reader, training_info, parallel=0, verbose=True):
        super(OverSamplingPatchSampler, self).__init__(log, train_idx, reader, training_info, parallel, verbose)   
        
    def set_probability_vector(self, p_masks, gt_masks):
        """
        Probability Vector for Probability sampling
        sampling_distribution : 'equiprobability'
        """
        self.probability = gt_masks * np.sum(1-gt_masks) + (1-gt_masks) * np.sum(gt_masks)
        self.probability = self.probability * np.copy(p_masks)
        self.probability = self.probability.flatten()
        self.probability = self.probability / np.sum(self.probability)
        return self.probability


#########################################################################################################################
# if __name__ == "__main__":
#     # logger
#     logger = logging_daily.logging_daily('base_model/config/log_info.yaml')
#     logger.reset_logging()
#     log = logger.get_logging()
    
#     reader = readers.ISLESReader(log)
#     cv_index = reader.get_cv_index(nfold=5)
#     cv_idx, test_idx = next(cv_index)
#     train_idx, validation_idx = reader.get_training_validation_index(cv_idx, validation_size = 0.2)
    
#     # uniform sampler
#     log.info('######################################################')
#     log.info('Uniform sampler test')
#     log.info('######################################################')
#     sampler = UniformSampler(log, train_idx, batch_size=16, steps_per_epoch=50, reader=reader,
#                              patch_size=(4,64,64), stride=8, subsets_per_epoch=5, augment=False, parallel=10, verbose=True)   
#     gen = train_generator(sampler)
#     x_train_batch, y_train_batch = gen[0]
#     gen.on_epoch_end()
#     log.info('x_train_batch shape : %s', x_train_batch.shape)
#     log.info('y_train_batch shape : %s', y_train_batch.shape)
    
#     # oversampling sampler
#     log.info('######################################################')
#     log.info('Oversampling sampler test')
#     log.info('######################################################')
#     sampler = OverSamplingSampler(log, train_idx, batch_size=16, steps_per_epoch=50, reader=reader,
#                                   patch_size=(4,64,64), stride=8, subsets_per_epoch=5, augment=False, parallel=10, verbose=True)   
#     gen = train_generator(sampler)
#     x_train_batch, y_train_batch = gen[0]
#     gen.on_epoch_end()
#     log.info('x_train_batch shape : %s', x_train_batch.shape)
#     log.info('y_train_batch shape : %s', y_train_batch.shape)
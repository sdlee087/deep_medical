######################################################################
## Deep Larning for Medical Image analysis
## - Build network for medical images
##
## Aug. 26, 2019
## Youngwon (youngwon08@gmail.com)
######################################################################

import time
import json
import sys
import abc

import numpy as np
import pandas as pd
import keras
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Average
import keras.callbacks
from keras.models import model_from_yaml
from keras.utils import multi_gpu_model
import keras.backend as k
import tensorflow as tf

from . import loss_and_metric
from . import tensorboard_utils
from .ops import *

#####################################################################################################################
# Base Network
#####################################################################################################################
class Base_Network(abc.ABC):
    """Inherit from this class when implementing new networks."""
    def __init__(self, network_info, log, fold=None):
        # Build Network
        
        if tf.__version__.startswith('2'):
            gpus = tf.config.experimental.get_visible_devices(device_type='GPU')
            self.number_of_gpu = len(gpus)
        else:
            self.number_of_gpu = len(k.tensorflow_backend._get_available_gpus())
        self.network_info = network_info
        self.log = log
        self.best_model_save = False
        self.TB = None
        self.model = None
        
        if fold != None: self.fold = fold
        else: self.fold = ''

    def load_model(self, model_yaml_path,  custom_objects={}, verbose=0):
        # load YAML and create model
        yaml_file = open(model_yaml_path, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        model = model_from_yaml(loaded_model_yaml, custom_objects=custom_objects)
        if verbose: self.log.info(model.summary())
        return model
    
    def build_network(self, model_yaml_path, verbose=0):
        self.model = self.load_model(model_yaml_path, verbose=verbose)
        
    def get_tensorboard_callbacks(self, validation_data=None):
        histogram_freq=int(self.network_info['tensorboard_info']['histogram_freq'])
        try: zcut = [int(zc.strip()) for zc in self.network_info['tensorboard_info']['zcut'].split(',')]
        except: zcut= [0,0]
        try: downsampling_scale = float(self.network_info['tensorboard_info']['downsampling_scale'].strip())
        except: downsampling_scale = 1.
        callback = self.TB(log_dir='%s/%s' % (self.network_info['tensorboard_info']['tensorboard_dir'],self.fold),
                           validation_data = validation_data,
                           histogram_freq=histogram_freq,
                           batch_size=int(self.network_info['validation_info']['batch_size']),
                           write_graph=self.network_info['tensorboard_info']['write_graph']=='True',
                           write_grads=self.network_info['tensorboard_info']['write_grads']=='True',
                           write_images=self.network_info['tensorboard_info']['write_images']=='True',
                           write_weights_histogram=self.network_info['tensorboard_info']['write_weights_histogram']=='True', 
                           write_weights_images=self.network_info['tensorboard_info']['write_weights_images']=='True',
                           embeddings_freq=int(self.network_info['tensorboard_info']['embeddings_freq']),
                           embeddings_metadata='metadata.tsv',
                           tb_data_steps=1,
                           zcut=zcut, downsampling_scale=downsampling_scale)
        return callback
    
    def get_callbacks(self, validation_data=None, model_path=None):
        # Callback
        if 'callbacks' in self.network_info['training_info']:
            callbacks = [cb.strip() for cb in self.network_info['training_info']['callbacks'].split(',')]
            for idx, callback in enumerate(callbacks):
                if 'EarlyStopping' in callback:
                    callbacks[idx] = getattr(keras.callbacks, callback)(monitor=self.network_info['training_info']['monitor'],
                                                                        mode=self.network_info['training_info']['callback_mode'],
                                                                        patience=int(self.network_info['training_info']['patience']),
                                                                        min_delta=float(self.network_info['training_info']['min_delta']),
                                                                        verbose=1)
                elif 'ModelCheckpoint' in callback:
                    self.best_model_save = True
                    callbacks[idx] = getattr(keras.callbacks, callback)(filepath=model_path,
                                                                        monitor=self.network_info['training_info']['monitor'],
                                                                        mode=self.network_info['training_info']['callback_mode'],
                                                                        save_best_only=True, save_weights_only=False,
                                                                        verbose=0)
                    self.log.info('Save base mode only')
                else: callbacks[idx] = getattr(keras.callbacks, callback)()
        else: callbacks = []
        if 'tensorboard_dir' in self.network_info['tensorboard_info'] and 'None' not in self.network_info['tensorboard_info']['tensorboard_dir']:
            callbacks.append(self.get_tensorboard_callbacks(validation_data))
        return callbacks

    def model_compile(self):
        try:
            self.parallel_model = multi_gpu_model(self.model, gpus=self.number_of_gpu)
            self.log.info("Training using multiple GPUs")
        except ValueError:
            self.parallel_model = self.model
            self.log.info("Training using single GPU or CPU")
    
        self.parallel_model.compile(loss=getattr(loss_and_metric, self.network_info['model_info']['loss']),
                           optimizer=getattr(keras.optimizers, 
                                             self.network_info['model_info']['optimizer'])(lr=float(self.network_info['model_info']['lr']),
                                                                                           decay=float(self.network_info['model_info']['decay'])),
                           metrics=[getattr(loss_and_metric, metric.strip()) for metric in self.network_info['model_info']['metrics'].split(',')])
        self.log.info('Build Network')
        self.log.info('Optimizer = {}'.format(self.network_info['model_info']['optimizer']))
        self.log.info('Loss = {}'.format(self.network_info['model_info']['loss']))
        self.log.info('Metrics = {}'.format(self.network_info['model_info']['metrics']))
        
    def save_model(self, model_yaml_path):
        model_yaml = self.model.to_yaml()
        with open(model_yaml_path, "w") as yaml_file:
            yaml_file.write(model_yaml)
    
    def save_weights(self, model_path):
        if self.best_model_save:
            self.log.info('Load best model')
            self.model.load_weights(model_path)
        self.model.save(model_path)
        self.log.info('Saved trained model weight at {} '.format(model_path))
        
    def load_weights(self, model_path, verbose=0):
        if verbose: self.log.info(self.model.summary())
        self.model.load_weights(model_path)
        self.log.info('Load trained model weight at {} '.format(model_path))

    def save_evaluation(self, eval_path, evaluation):
        np.save(eval_path, evaluation)
        
    def save_prediction(self, pred_path, prediction):
        np.save(pred_path, prediction)
            
    def save_history(self, hist_path, history):
        try:
            with open(hist_path, 'w+') as f:
                json.dump(history, f)
        except:
            with open(hist_path, 'w+') as f:
                hist = dict([(ky, np.array(val).astype(np.float).tolist()) for (ky, val) in history.items()])
                json.dump(hist, f)
            
    def fit(self, x, y, epochs=1, batch_size=1,
            warm_start=False, warm_start_model=None, hist_path = None,
            initial_epoch=0,
            max_queue_size=50, workers=1, use_multiprocessing=False,
            model_path = None):
        batch_size = int(self.network_info['training_info']['batch_size'])
        callbacks = self.get_callbacks(None)
        
        if warm_start:
            with open('./%s/%s' % (warm_start_model, hist_path), 'r') as f:
                history = json.load(f)
            try:
                trained_epoch = int(history['epochs'][-1])
                if np.isnan(trained_epoch):
                    trained_epoch = int(history['epochs'][-2])
            except:
                trained_epoch = len(list(history.values())[0])
            epochs += trained_epoch
            epoch = initial_epoch+trained_epoch
            self.load_weights('%s/%s'% (warm_start_model, model_path))
            self.log.info('Load %d epoch trained weights from %s' % (trained_epoch, warm_start_model))
        else:
            epoch = initial_epoch
        
        self.log.info('Training start!')
        trainingtime = time.time()
        
        hist = self.parallel_model.fit(x, y, batch_size=batch_size,
                              epochs=epochs, 
                              verbose=1, 
                              callbacks=callbacks, 
                              validation_split=self.network_info['training_info']['validation_size'],
                              initial_epoch=epoch,                                        
                              max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)
        
        if self.best_model_save:
            self.log.info('Load best model')
            self.load_weights(model_path)
        self.log.info('Training end with time {}!'.format(time.time()-trainingtime))
        return hist
    
    def fit_generator(self, train_sampler, epochs=1, validation_sampler=None, 
                      warm_start=False, warm_start_model=None, hist_path = None,
                      initial_epoch=0,
                      max_queue_size=50, workers=1, use_multiprocessing=False,
                      model_path = None):
        callbacks = self.get_callbacks(validation_sampler, model_path)
        
        if warm_start:
            with open('./%s/%s' % (warm_start_model, hist_path), 'r') as f:
                history = json.load(f)
            try:
                trained_epoch = int(history['epochs'][-1])
                if np.isnan(trained_epoch):
                    trained_epoch = int(history['epochs'][-2])
            except:
                trained_epoch = len(list(history.values())[0])
            epochs += trained_epoch
            epoch = initial_epoch+trained_epoch
            self.load_weights('%s/%s'% (warm_start_model, model_path))
            self.log.info('Load %d epoch trained weights from %s' % (trained_epoch, warm_start_model))
        else:
            epoch = initial_epoch
        
        self.log.info('Training start!')
        trainingtime = time.time()
        
        hist = self.parallel_model.fit_generator(train_sampler,
                                        steps_per_epoch=len(train_sampler),
                                        epochs=epochs, 
                                        verbose=1, 
                                        initial_epoch=epoch,
                                        callbacks=callbacks, 
                                        validation_data=validation_sampler,
                                        validation_steps=len(validation_sampler),
                                        max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)
        
        if self.best_model_save:
            self.log.info('Load best model')
            self.load_weights(model_path)
        self.log.info('Training end with time {}!'.format(time.time()-trainingtime))
        return hist
    
    def evaluate(self, x, y, batch_size):
        self.log.info('Evaluation start!')
        trainingtime = time.time()
        evaluation = self.parallel_model.evaluate(x, y, batch_size = batch_size, verbose=1)
        self.log.info('Evaluation end with time {}!'.format(time.time()-trainingtime))
        evaluation_measure = [self.network_info['model_info']['loss'].strip()] + [metric.strip() for metric in self.network_info['model_info']['metrics'].split(',')]
        self.log.info('Evaluation : %s' % ''.join(['%12s' % k for k in evaluation_measure]))
        self.log.info('              : %s', ''.join(['%10.4f' % k for k in evaluation]))
        return evaluation
    
    def evaluate_generator(self, test_sampler, max_queue_size=50, workers=1, use_multiprocessing=False):
        self.log.info('Evaluation start!')
        trainingtime = time.time()
        evaluation = self.parallel_model.evaluate_generator(test_sampler, steps=len(test_sampler), 
                                                            max_queue_size=max_queue_size, workers=1, use_multiprocessing=False)
        self.log.info('Evaluation end with time {}!'.format(time.time()-trainingtime))
        evaluation_measure = [self.network_info['model_info']['loss'].strip()] + [metric.strip() for metric in self.network_info['model_info']['metrics'].split(',')]
        self.log.info('Evaluation : %s' % ''.join(['%12s' % k for k in evaluation_measure]))
        self.log.info('              : %s', ''.join(['%10.4f' % k for k in evaluation]))
        return evaluation
    
    def predict(self, x, batch_size):
        self.log.info('Prediction start!')
        trainingtime = time.time()
        prediction = self.parallel_model.predict(x, batch_size = batch_size, verbose=1)
        self.log.info('Prediction end with time {}!'.format(time.time()-trainingtime))
        return prediction
    
    def predict_generator(self, test_sampler, max_queue_size=50, workers=1, use_multiprocessing=False):
        self.log.info('Prediction start!')
        trainingtime = time.time()
        pred = self.parallel_model.predict_generator(test_sampler, steps=len(test_sampler), 
                                            max_queue_size=max_queue_size, workers=1, use_multiprocessing=False,
                                            verbose=1)
        self.log.info('Prediction end with time {}!'.format(time.time()-trainingtime))
        return prediction
    
#####################################################################################################################
# Classification Network
#####################################################################################################################
class ClassificationNetwork(Base_Network):
    def __init__(self, network_info, log, fold=None):
        super(ClassificationNetwork,self).__init__(network_info, log, fold)
        self.TB = tensorboard_utils.TensorBoardClassificationWrapper
        
#####################################################################################################################
# Classification Network with attention
#####################################################################################################################
class ClassificationAttentionNetwork(Base_Network):
    def __init__(self, network_info, log, fold=None):
        super(ClassificationAttentionNetwork,self).__init__(network_info, log, fold)
        self.TB = tensorboard_utils.TensorBoardClassificationAttentionWrapper
        
    def build_network(self, model_yaml_path, verbose=0):
        self.model = self.load_model(model_yaml_path, custom_objects={'Attention3D':Attention3D}, verbose=verbose)
        
#####################################################################################################################
# Classification Network with guided attention
### Assume that info is mean..
#####################################################################################################################
class ClassificationGuidedAttentionNetwork(Base_Network):
    def __init__(self, network_info, log, fold=None):
        super(ClassificationGuidedAttentionNetwork,self).__init__(network_info, log, fold)
        self.TB = tensorboard_utils.TensorBoardClassificationGuidedAttentionWrapper
        self.penalty_lambda = float(self.network_info['model_info']['penalty_lambda'].strip())
        
    def build_network(self, model_yaml_path, verbose=0):
        def get_mean(x):
            import keras.backend as k
            return k.mean(x, axis=0, keepdims=True)
        def get_mse(x):
            import keras.backend as k
            return k.mean((x[0]-x[1])**2.) + k.zeros_like(x[2])
       

        self.task_model = self.load_model(model_yaml_path, custom_objects={'Attention3D':Attention3D}, verbose=verbose)
        attention_layer = [l for l in self.task_model.layers if 'attention' in l.name][0]
        _, attention = attention_layer.output
        p_hat = self.task_model.output
        mean_attention = Lambda(get_mean, name='marginal_attention')(attention)
        prior_shape = k.int_shape(attention)
        prior = Input(shape=prior_shape[1:], name='input_information', dtype='float32')
        mean_prior = Lambda(get_mean, name='prior')(prior)
        penalty = Lambda(get_mse, name='penalty')([mean_attention,mean_prior,p_hat])
        self.model = Model(inputs= [self.task_model.input, prior] , outputs=[p_hat, penalty])
        if verbose: self.model.summary(line_length=180)
    
    def model_compile(self):
        try:
            self.parallel_model = multi_gpu_model(self.model, gpus=self.number_of_gpu)
            self.log.info("Training using multiple GPUs")
        except ValueError:
            self.parallel_model = self.model
            self.log.info("Training using single GPU or CPU")
    
        self.parallel_model.compile(loss={'p_hat': getattr(loss_and_metric, self.network_info['model_info']['loss']), 
                                 'penalty': getattr(loss_and_metric, self.network_info['model_info']['penalty_loss'])},
                           optimizer=getattr(keras.optimizers, 
                                             self.network_info['model_info']['optimizer'])(lr=float(self.network_info['model_info']['lr']),
                                                                                           decay=float(self.network_info['model_info']['decay'])),
                           metrics={'p_hat':[getattr(loss_and_metric, metric.strip()) for metric in self.network_info['model_info']['metrics'].split(',')], 
                                    'penalty':[]},
                           loss_weights={'p_hat':1., 'penalty':self.penalty_lambda})
        self.log.info('Build Network')
        self.log.info('Optimizer = {}'.format(self.network_info['model_info']['optimizer']))
        self.log.info('Loss = {}'.format(self.network_info['model_info']['loss']))
        self.log.info('Metrics = {}'.format(self.network_info['model_info']['metrics']))

#####################################################################################################################
# Classification Network with guided attention
#####################################################################################################################
# class ClassificationGANGuidedAttentionNetwork(Base_Network):
#     def __init__(self, network_info, log, fold=None):
#         super(ClassificationGANGuidedAttentionNetwork,self).__init__(network_info, log)
#         self.TB = tensorboard_utils.TensorBoardClassificationGANGuidedAttentionWrapper
        
#     def build_network(self, model_yaml_path, verbose=0):
#         self.task_model = self.load_model(model_yaml_path, custom_objects={'Attention3D':Attention3D}, verbose=verbose)
#         self.discriminator_model = self.load_model(model_yaml_dir+self.path_info['model_info']['model_discriminator'], verbose=verbose==2)
        
#     def fit_generator(self):

#####################################################################################################################
# Segmentation Network
#####################################################################################################################
class SegmentationNetwork(Base_Network):
    def __init__(self, network_info, log, fold=None):
        super(SegmentationNetwork,self).__init__(network_info, log, fold)
        self.TB = tensorboard_utils.TensorBoardSegmentationWrapper
    
    # def predict_generator(self, test_sampler, max_queue_size=50, workers=1, use_multiprocessing=False):
    #     self.log.info('Prediction start!')
    #     trainingtime = time.time()
    #     pred = self.parallel_model.predict_generator(test_sampler, steps=len(test_sampler), 
    #                                         max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing,
    #                                         verbose=1)
    #     prediction = []
    #     n_person = test_sampler.get_n_person()
    #     batch_idx = test_sampler.get_batch_idx()
    #     for i in range(n_person):
    #         prediction.append(test_sampler.sampler.reader.stack_patch_y(pred[i*len(batch_idx):(i+1)*len(batch_idx),:,:,:,0], batch_idx,
    #                                                                     patch_size=test_sampler.sampler.patch_size))
    #     prediction = np.array(prediction)
    #     self.log.info('Prediction end with time {}!'.format(time.time()-trainingtime))
    #     return prediction
    
#####################################################################################################################
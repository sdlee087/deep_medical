######################################################################
## Deep Larning for Medical Image analysis
## - Main code
##
## Aug. 26, 2019
## Youngwon (youngwon08@gmail.com)
######################################################################

import os
import sys
import json
import time
import numpy as np
import random
import gc
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf

seed_value = 123
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set `tensorflow` pseudo-random generator at a fixed value
if tf.__version__.startswith('2'): tf.random.set_seed(seed_value)
else: tf.set_random_seed(seed_value)

########### TODO: fix after install package #######################################################
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
from deepmedical import logging_daily
from deepmedical import configuration
from deepmedical import loss_and_metric
from deepmedical import readers
from deepmedical import samplers
from deepmedical import build_network
from deepmedical.utils import file_path_fold, argv_parse
#####################################################################################################

import keras.backend as k
argdict = argv_parse(sys.argv)

if tf.__version__.startswith('2'):
    gpus = tf.config.experimental.get_visible_devices(device_type='GPU')
    try: tf.config.experimental.set_memory_growth(gpus, True)
    except: pass
else:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

# nfold=int(argdict['kfold'][0])
max_queue_size=int(argdict['max_queue_size'][0])
workers=int(argdict['workers'][0])
use_multiprocessing=argdict['use_multiprocessing'][0]=='True'      

### Logger ############################################################################################
logger = logging_daily.logging_daily(argdict['log_info'][0])
logger.reset_logging()
log = logger.get_logging()
log.setLevel(logging_daily.logging.INFO)

log.info('Argument input')
for argname, arg in argdict.items():
    log.info('    {}:{}'.format(argname,arg))
    
### Configuration #####################################################################################
config_data = configuration.Configurator(argdict['path_info'][0], log)
config_data.set_config_map(config_data.get_section_map())
config_data.print_config_map()

config_network = configuration.Configurator(argdict['network_info'][0], log)
config_network.set_config_map(config_network.get_section_map())
config_network.print_config_map()

path_info = config_data.get_config_map()
network_info = config_network.get_config_map()

### Training hyperparameter ##########################################################################
model_save_dir = path_info['model_info']['model_dir']
warm_start= network_info['training_info']['warm_start'] == 'True'
warm_start_model = network_info['training_info']['warm_start_model']

model_path = os.path.join(model_save_dir, path_info['model_info']['weight'])
model_yaml_path = os.path.join(model_save_dir, path_info['model_info']['model'].strip())
hist_path = os.path.join(model_save_dir, path_info['model_info']['history'])

### Reader ###########################################################################################
log.info('-----------------------------------------------------------------')
reader_class = getattr(readers, network_info['model_info']['reader_class'].strip())
reader = reader_class(log, path_info, network_info, verbose=True)

### Index reader #################################################################################
train_tot_idxs_path = path_info['model_info']['train_tot_idxs']
validation_tot_idxs_path = path_info['model_info']['validation_tot_idxs']
test_tot_idxs_path = path_info['model_info']['test_tot_idxs']
train_tot_idxs = np.load(train_tot_idxs_path, allow_pickle=True)
validation_tot_idxs = np.load(validation_tot_idxs_path, allow_pickle=True)
test_tot_idxs = np.load(test_tot_idxs_path, allow_pickle=True)

# cv_index = reader.get_cv_index(nfold=nfold)
# train_tot_idxs = []
# validation_tot_idxs = []
# test_tot_idxs = []
# for fold, (cv_idx, test_idx) in enumerate(cv_index):
#     train_idx, validation_idx = reader.get_training_validation_index(cv_idx, 
#                                                                         validation_size = float(network_info['training_info']['validation_size']))
#     train_tot_idxs.append(np.array(train_idx))
#     validation_tot_idxs.append(np.array(validation_idx))
#     test_tot_idxs.append(np.array(test_idx))
# train_tot_idxs = np.array(train_tot_idxs)
# validation_tot_idxs = np.array(validation_tot_idxs)
# test_tot_idxs = np.array(test_tot_idxs)
## save
# train_tot_idxs_path = os.path.join(model_save_dir, path_info['model_info']['train_tot_idxs'])
# validation_tot_idxs_path = os.path.join(model_save_dir, path_info['model_info']['validation_tot_idxs'])
# test_tot_idxs_path = os.path.join(model_save_dir, path_info['model_info']['test_tot_idxs'])
# np.save(train_tot_idxs_path, train_tot_idxs)
# np.save(validation_tot_idxs_path, validation_tot_idxs)
# np.save(test_tot_idxs_path, test_tot_idxs)

history = []
evaluation = []
# exp_weight = []
# exp_probability = []
# exp_expert = []
starttime = time.time()
log.info('-------Optimization start!----------------------------------')
foldstarttime = time.time()

### Sampler ##########################################################################################
log.info('-----------------------------------------------------------------')
log.info('Construct training data sampler')
train_sampler_class = getattr(samplers, network_info['training_info']['sampler_class'].strip())  
train_sampler = train_sampler_class(log, train_tot_idxs, reader, network_info['training_info'], parallel=workers, verbose=False)
train_generator = samplers.train_generator(train_sampler)

# Validation data sampler
log.info('Construct validation data sampler')
validation_sampler_class = getattr(samplers, network_info['validation_info']['sampler_class'].strip())
validation_sampler = validation_sampler_class(log, validation_tot_idxs, reader, network_info['validation_info'], parallel=workers, verbose=False)
validation_generator = samplers.train_generator(validation_sampler)

# Test data sampler
log.info('Construct test data sampler')
test_sampler_class = getattr(samplers, network_info['test_info']['sampler_class'].strip())
test_sampler = test_sampler_class(log, test_tot_idxs, reader, network_info['test_info'], parallel=workers, verbose=False)
test_generator = samplers.train_generator(test_sampler)
# log.info('test_index: %s' % str(test_idx))
sys.stdout.flush()

### Bulid network ####################################################################################
if not tf.__version__.startswith('2'): k.set_session(tf.Session(config=config))
log.info('-----------------------------------------------------------------')
log.info('Building network.')
network_class = getattr(build_network, network_info['model_info']['network_class'].strip())  
network = network_class(network_info, log, 0)
network.build_network(model_yaml_path, verbose=0)
network.model_compile() ## TODO : weight clear only (no recompile)
sys.stdout.flush()

### Training #########################################################################################
log.info('-----------------------------------------------------------------')
log.info('computing start!----------------------------------')
hist = network.fit_generator(train_generator, 
                                epochs=int(network_info['training_info']['epochs']),
                                validation_sampler=validation_generator,
                                warm_start=warm_start,
                                warm_start_model=warm_start_model,
                                hist_path = hist_path,
                                max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing,
                                model_path=model_path)
sys.stdout.flush()
network.save_weights(model_path)
log.info('Save weight at {}'.format(model_path))
network.save_history(hist_path, hist.history)
log.info('Save history at {}'.format(hist_path))
sys.stdout.flush()

# Evaluation
eval_res = network.evaluate_generator(test_generator)
evaluation.append(eval_res)

if not tf.__version__.startswith('2'): k.clear_session()
log.info('Compute time : {}'.format(time.time()-foldstarttime))
# log.info('%d fold computing end!---------------------------------------------' % (fold+1))

### Save #########################################################################################
np.save(os.path.join(model_save_dir, path_info['model_info']['evaluation']),evaluation)
### Summary #########################################################################################
evaluation = np.vstack(evaluation)
evaluation_measure = [network_info['model_info']['loss'].strip()] + [metric.strip() for metric in network_info['model_info']['metrics'].split(',')]
log.info('Evaluation : %s' % ''.join(['%12s' % k for k in evaluation_measure]))
mean = np.mean(evaluation, axis=0)
std = np.std(evaluation, axis=0)
log.info('      mean : %s', ''.join(['%10.4f' % k for k in mean]))
log.info('       std : %s', ''.join(['%10.4f' % k for k in std]))

### Exit #########################################################################################
log.info('Total Computing Ended')
log.info('-----------------------------------------------------------------')
gc.collect()
sys.exit(0)

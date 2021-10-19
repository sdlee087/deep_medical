###################################################################################################################
# Preprocessing
#    Construct patients' list
#    Resizing
#    Normalization
#    nii to npy
#    Save as np.uint8 for MRI & GT
# Youngwon Choi (youngwon08@gmail.com)
# Jul. 2019
###################################################################################################################
import os
import sys
import numpy as np

import timeit
from scipy import ndimage
from scipy.ndimage import interpolation, zoom

import medpy.io as medio

import multiprocessing as mp
from multiprocessing import Pool, Manager, Process
from functools import partial

############################################################################################################################
# Arguments
############################################################################################################################
# ISLES 15 SISS
#########################################################################################
# data_path = "/DATA/data/isles15_SISS/Training"
# save_path = "/DATA/data/isles15_SISS/npy_training"
# training = True
# mapIndex = ['MR_DWI','MR_T1','MR_T2','MR_Flair']
# gtIndex = 'OT'

# data_path = "/DATA/data/isles15_SISS/Testing"
# save_path = "/DATA/data/isles15_SISS/npy_testing"
# training = False
# mapIndex = ['MR_DWI','MR_T1','MR_T2','MR_Flair']

# img_shape = [154,224,224,len(mapIndex)] #[%%2, %%8, %%8] should be 0
#########################################################################################
# ISLES 15 SPES
#########################################################################################
# data_path = "/DATA/data/isles15_SPES/Training"
# save_path = "/DATA/data/isles15_SPES/npy_training"
# training = True
# mapIndex = ['DWI','CBF','CBV','T1c','T2','Tmax','TTP']
# gtIndex = 'OT'

# data_path = "/DATA/data/isles15_SPES/Testing"
# save_path = "/DATA/data/isles15_SPES/npy_testing"
# training = False
# mapIndex = ['DWI','CBF','CBV','T1c','T2','Tmax','TTP']

# img_shape = [72,112,112,len(mapIndex)] #[%%2, %%8, %%8] should be 0
#########################################################################################
# ISLES 16
#########################################################################################
# data_path = "/DATA/data/isles16/Training"
# save_path = "/DATA/data/isles16/npy_training"
# training = True
# mapIndex = ['ADC', 'MTT', 'rBF', 'rBV', 'Tmax', 'TTP']
# gtIndex = 'GT'

# data_path = "/DATA/data/isles16/Testing"
# save_path = "/DATA/data/isles16/npy_testing"
# training = False
# mapIndex = ['ADC', 'MTT', 'rBF', 'rBV', 'Tmax', 'TTP']

# img_shape = [32,512,512,len(mapIndex)] #[%%2, %%8, %%8] should be 0
#########################################################################################
# ISLES 17
#########################################################################################
# data_path = "/home/muha/data/isles17/Training"
# save_path = "/home/muha/data/isles17/npy_training"
# training = True
# mapIndex = ['ADC', 'MTT', 'rCBF', 'rCBV', 'Tmax', 'TTP']
# gtIndex = 'OT'

# data_path = "/home/muha/data/isles17/Testing"
# save_path = "/home/muha/data/isles17/npy_testing"
# training = False
# mapIndex = ['ADC', 'MTT', 'rCBF', 'rCBV', 'Tmax', 'TTP']

# img_shape = [32,512,512,len(mapIndex)]
#########################################################################################
# ISLES 17 with perfusion
#########################################################################################
data_path = "/home/reddragon/data/isles17/Training"
save_path = "/home/reddragon/data/isles17/npy_training20_withperf"
training = True
mapIndex = ['ADC', 'Tmax', 'TTP']
gtIndex = 'OT'

# data_path = "/home/muha/data/isles17/Testing"
# save_path = "/home/muha/data/isles17/npy_testing_perfonly"
# training = False
# mapIndex = ['Tmax', 'TTP']

img_shape = [32,512,512,len(mapIndex)]
#########################################################################################
# ISLES 17 diff only
#########################################################################################
# data_path = "/home/muha/data/isles17/Training"
# save_path = "/home/muha/data/isles17/npy_training_diffonly"
# training = True
# mapIndex = ['ADC']
# gtIndex = 'OT'

# data_path = "/home/muha/data/isles17/Testing"
# save_path = "/home/muha/data/isles17/npy_testing_diffonly"
# training = False
# mapIndex = ['ADC']

# img_shape = [32,512,512,len(mapIndex)]
#########################################################################################
order = 1
thrd = 0.5
verbose = 2
parallel = 10
gt_shape = img_shape[:-1]

############################################################################################################################
# Helper Functions
############################################################################################################################
# def resize(patient, img, targetsize, order=1, type_of_image='None', verbose=0):
#     # fix with same ratio & z padding like CT
    
#     tic = timeit.default_timer()
    
#     # x-y axis rescale
#     scale = targetsize[1+np.argmax(img.shape[1:])] / np.max(img.shape[1:])
#     if np.abs(scale-1) < 1e-10:
#         resized_img = img
#     else:
#         resized_img = zoom(img, zoom=[1,scale,scale], order=order)
    
#     # z-axis rescale
#     #scale_z = np.floor(targetsize[0]/resized_img.shape[0])
#     scale_z = targetsize[0]/resized_img.shape[0]
#     if np.abs(scale_z-1) > 1e-10:
#         resized_img = zoom(resized_img, zoom=[scale_z,1,1], order=order)
    
#     # padding
#     diff = np.array(targetsize) - np.array(resized_img.shape)
#     pad = np.vstack(np.array([diff //2, diff //2 + diff % 2]).transpose())
#     resized_img = np.pad(resized_img, pad, mode='constant').astype(np.float32)
#     if verbose:
#         print('%14s-%2s: zoom %s with z scale = %.3f, x-y scale = %.3f, pad = %s with time %.5fs' % (type_of_image, patient, img.shape,
#                                                                                                      np.array(scale_z).round(3),
#                                                                                                      np.array(scale).round(3),
#                                                                                                      pad.tolist(),
#                                                                                                      timeit.default_timer() - tic))
#         sys.stdout.flush()
#     return resized_img

def resize(patient, img, targetsize, order=1, type_of_image='None', verbose=0):
    tic = timeit.default_timer()
    
    # x-y axis rescale
    # scale = targetsize[1+np.argmax(img.shape[1:])] / np.max(img.shape[1:])
    scale = np.array(targetsize) / np.array(img.shape)
    resized_img = zoom(img, zoom=scale, order=order)
    
    # padding
    diff = np.array(targetsize) - np.array(resized_img.shape)
    pad = np.vstack(np.array([diff //2, diff //2 + diff % 2]).transpose())
    resized_img = np.pad(resized_img, pad, mode='constant').astype(np.float32)
    if verbose:
        print('%14s-%s: zoom %s with scale = %s, pad = %s with time %.5fs' % (type_of_image, patient, img.shape,
                                                                              np.array(scale).round(3), pad.tolist(),
                                                                              timeit.default_timer() - tic))
    sys.stdout.flush()
    return resized_img

def normalize_minmax(img): return (img - np.min(img))/ (np.max(img) - np.min(img))*225

def save_image(file_info, target_shape=(32,512,512), order=0, thrd=0.5, verbose=0, mapIndex=['ADC', 'MTT', 'rCBF', 'rCBV', 'Tmax', 'TTP']):
    map_img = np.zeros(list(target_shape) + [len(mapIndex)])
    for i, map_name in enumerate(mapIndex):
        img = medio.load(file_info['MRI'][map_name])[0].transpose([2,0,1])
        img = resize(file_info['case'], img, target_shape, order, type_of_image="MRI(%4s)" % map_name, verbose=verbose)
        map_img[:,:,:,i] = normalize_minmax(img)
    sys.stdout.flush()
    
    np.save('%s/MRI/%s.npy' % (save_path,file_info['case']), map_img.astype(np.uint8))
    if verbose: print('Save %s/MRI/%s.npy' % (save_path,file_info['case']))
    sys.stdout.flush()
    
    if training: 
        gt = medio.load(file_info['GT'])[0].transpose([2,0,1])
        gt = resize(file_info['case'], gt, target_shape, order, type_of_image="GT", verbose=verbose)
        gt[gt >= thrd] = 1
        gt[gt < thrd] = 0
        np.save('%s/GT/%s.npy' % (save_path,file_info['case']), gt.astype(np.uint8))
        if verbose: print('Save %s/GT/%s.npy' % (save_path,file_info['case']))
    sys.stdout.flush()
    
############################################################################################################################
# Construct patients' list
############################################################################################################################
x_list = []
case_list = [f for f in os.listdir("%s"% data_path) if not (f.startswith('.') or f.startswith('__'))]
for case in case_list:
    map_list = {}
    ispatient = False
    for dirName, subdirList, fileList in os.walk("%s/%s" % (data_path, case)):
        for filename in fileList:
            if ".nii" in filename.lower():
                ispatient = True
                for map_name in mapIndex:
                    if map_name in filename: map_list[map_name] = os.path.join(dirName,filename)
                # if training and gtIndex in filename: gt = os.path.join(dirName,filename)
                # For SPES
                if training and gtIndex in filename and "penumbralabel": gt = os.path.join(dirName,filename)
    if ispatient:
        if training: x_list.append({'case':case, 'MRI': map_list, 'GT':gt})
        else : x_list.append({'case':case, 'MRI': map_list})
x_list = np.array(x_list)
for info in x_list:
    print(info['case'])
    print(list(info['MRI'].values()))
    if training: print(info['GT']) 

############################################################################################################################
# Preprocessing
############################################################################################################################

tic = timeit.default_timer()

try: os.stat(save_path)
except: os.mkdir(save_path)
try: os.stat('%s/MRI'%save_path)
except: os.mkdir('%s/MRI'%save_path)
if training:
    try: os.stat('%s/GT'%save_path)
    except: os.mkdir('%s/GT'%save_path)

if parallel:
    pool = Pool(processes=parallel)
    file_info_list = x_list
    pool.map_async(partial(save_image, target_shape=img_shape[:-1], order=order, thrd=thrd, verbose=verbose, mapIndex=mapIndex), file_info_list).get()
    pool.close()
    pool.join()
else:
    for i, file_info in enumerate(x_list):
        save_image(file_info, target_shape=img_shape[:-1], order=order, thrd=thrd, verbose=verbose, mapIndex=mapIndex)
        
if training: x_list = [[info['case'], '%s/MRI/%s.npy'%(save_path,info['case']), '%s/GT/%s.npy'%(save_path,info['case'])] for info in x_list]
else: x_list = [[info['case'], '%s/MRI/%s.npy'%(save_path,info['case'])] for info in x_list]
print(x_list)
np.save('%s/x_list.npy' % save_path, x_list)
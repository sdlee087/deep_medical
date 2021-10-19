######################################################################
## Medical segmentation using CNN
## - Miscellaneous
##
## Nov 16. 2018
## Youngwon (youngwon08@gmail.com)
##
## Reference
## - Keras (https://github.com/keras-team/keras)
######################################################################

import os
import platform
import timeit
import glob
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm
import cv2

def argv_parse(argvs):
    arglist = [arg.strip() for args in argvs[1:] for arg in args.split('=') if not arg.strip()=='']
    arglist.reverse()
    argdict = dict()
    argname = arglist.pop()
    while len(arglist) > 0:
        if '--' not in argname:
            raise Exception('python argument error')
        argv = []
        while len(arglist) > 0:
            arg = arglist.pop()
            if '--' in arg:
                argdict[argname.split('--')[-1]] = argv
                argname = arg
                break
            argv.append(arg)
    argdict[argname.split('--')[-1]] = argv
    return argdict

def file_path_fold(path, fold):
    path = path.split('.')
    return '.'+''.join(path[:-1])+'_'+str(fold)+'.'+path[-1]

def convert_bytes(num, x='MB'):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0

def file_size(file_path, scale='MB'):
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size, scale)
    
def print_sysinfo():
    print('\nPython version  : {}'.format(platform.python_version()))
    print('compiler        : {}'.format(platform.python_compiler()))
    print('\nsystem     : {}'.format(platform.system()))
    print('release    : {}'.format(platform.release()))
    print('machine    : {}'.format(platform.machine()))
    print('processor  : {}'.format(platform.processor()))
    print('CPU count  : {}'.format(mp.cpu_count()))
    print('interpreter: {}'.format(platform.architecture()[0]))
    print('\n\n')
    
def _delation(im, kernal_size=(4,4)):
    # Dilation (thickness)
    kernel = np.ones(kernal_size,np.uint8)
    im = cv2.dilate(im,kernel,iterations = 1)
    im[im>=0.5] = 1.
    im[im<0.5] = 0.
    return np.array(im)

def set_ax_plot(ax, models=['base_model'], models_aka=['base_model'], base_path='.', 
                y_name = 'val_dice', monitor='max',
                x_name = 'epochs', mode='summary', niters=None):
    x_title = x_name.title()
    if 'val' in y_name:
        y_title = y_name.split('_')[-1].title()
        title = 'Validation History - %s (per %s)' % (y_title, x_title)
    else:
        y_title = y_name.title()
        title = 'History - %s (per %s)' % (y_title, x_title)
        
    ax.set_title(title, fontdict={'fontsize':15})
    for col, model in enumerate(models):
        hist_path = './%s/hist.json' % (model)
        kfold = len(glob.glob('%s/%s'%(base_path,file_path_fold(hist_path,'*'))))
        history = []
        for fold in range(kfold):
            with open('%s/%s'%(base_path,file_path_fold(hist_path,fold)), 'r') as f:
                history.append(json.load(f))
        max_epoch = np.max([len(hist['loss']) for hist in history])
        
        if not niters == None: niter = niters[col]
        else: niter = None
        if 'epoch' in x_name: index = np.arange(1, max_epoch+1)
        elif 'iter' in x_name: index = np.arange(1, max_epoch*niter+1, niter)
        else: raise ValueError
            
        value = np.zeros((len(history),max_epoch))
        for i, hist in enumerate(history):
            value[i,:len(hist[y_name])] = hist[y_name]
            
        if mode == 'summary':
            ax.plot(index, np.mean(value, axis=0), 'C%s.-' % (col+1), label='%s-%s'% (models_aka[col], y_title))
            ax.fill_between(index, np.mean(value, axis=0)-np.std(value, axis=0), np.mean(value, axis=0)+np.std(value, axis=0),
                            color='C%s' % (col+1), alpha=0.2)
            if monitor=='max' : # total epoch max per each fold
                ax.plot(index[np.argmax(value, axis=1)], np.max(value, axis=1), 'C%s*' % (col+1), markersize=12)
                for j, (x,y) in enumerate(zip(index[np.argmax(value, axis=1)], np.max(value, axis=1))):
                    ax.annotate(j+1, (x,y))
            if monitor=='min' : # total epoch min per each fold
                ax.plot(index[np.argmin(value, axis=1)], np.min(value, axis=1), 'C%s*' % (col+1), markersize=12)
                for j, (x,y) in enumerate(zip(index[np.argmin(value, axis=1)], np.min(value, axis=1))):
                    ax.annotate(j+1, (x,y)) 
            ax.plot(index, np.max(value, axis=0), 'C%s.' % (col+1), alpha=0.3) # total fold max per epoch
            ax.plot(index, np.min(value, axis=0), 'C%s.' % (col+1), alpha=0.3) # total fold min per epoch
        elif mode == 'all':
            for j, v in enumerate(value):
                if j == 0: ax.plot(index, v, 'C%s.-' % (col+1), label='%s-%s'% (models_aka[col], y_title), alpha=0.4)
                else: ax.plot(index, v, 'C%s.-' % (col+1), alpha=0.4)
            
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.legend(loc='lower right', fontsize='small')

def plot_history(models, models_aka, base_path='.', history_types = ['validation','train'],
                 y_names = ['dice','precision','recall'],
                 y_monitor=['max','max','max'],
                 x_name='epochs',
                 mode = 'summary', figsize=(20,20), niters=None):
    fig, axes = plt.subplots(len(y_names), len(history_types), figsize=figsize)
    if len(y_names) == 1: axes = np.expand_dims(axes, 0)
    if len(history_types) == 1: axes = np.expand_dims(axes, -1)
    for j in range(len(history_types)):
        for i, y_name in enumerate(y_names):
            if 'val' in history_types[j]: set_ax_plot(axes[i,j], models, models_aka, base_path, 'val_%s' % y_name, y_monitor[i], x_name, mode, niters)
            else: set_ax_plot(axes[i,j], models, models_aka, base_path, y_name, y_monitor[i], x_name, mode, niters)
    fig.tight_layout()
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
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
import keras.backend as k
import tensorflow as tf

import matplotlib
import matplotlib.cm

######################################################################
# For segmentation
######################################################################
class TensorBoardSegmentationWrapper(TensorBoard):
    '''
    Sets the self.validation_data property for use with TensorBoard callback.
    
    Image Summary with multi-modal medical 3D volumes:  
        Thumbnail of nrow x ncol 2D images (of one person) 
            nrow: number of slice (z-axis)
            ncol: 
                   input images: number of modals
                   bottleneck images : number of filters
                   output images: 2 (GT, predict)
        TODO: fix one person as reference..
    '''

    def __init__(self, validation_data, write_weights_histogram = True, write_weights_images=False, 
                 tb_data_steps=1, zcut=[0,0], downsampling_scale = 1,
                 **kwargs):
        super(TensorBoardSegmentationWrapper, self).__init__(**kwargs)
        self.write_weights_histogram = write_weights_histogram
        self.write_weights_images = write_weights_images
        self.tb_data_steps = tb_data_steps
        self.validation_data = validation_data
        self.img_shape = validation_data[0][0].shape[1:]
        self.zcut = zcut
        self.downsampling_scale = downsampling_scale
        # print('initial image_shape: %s' % list(self.img_shape))
        
        if self.embeddings_data is None and self.validation_data:
            self.embeddings_data = self.validation_data
    
    def normalize(self, value):
        vmin = tf.reduce_min(value)
        vmax = tf.reduce_max(value)
        value = (value - vmin) / (vmax - vmin)
        return value
        
    def colorize(self, value, cmap=None):
        """
        ref: https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b
        A utility function for TensorFlow that maps a grayscale image to a matplotlib
        colormap for use with TensorBoard image summaries.
        By default it will normalize the input value to the range 0..1 before mapping
        to a grayscale colormap.
        Arguments:
          - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
            [height, width, 1].
          - vmin: the minimum value of the range used for normalization.
            (Default: value minimum)
          - vmax: the maximum value of the range used for normalization.
            (Default: value maximum)
          - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
            (Default: 'gray')
        Example usage:
        ```
        output = tf.random_uniform(shape=[256, 256, 1])
        output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
        tf.summary.image('output', output_color)
        ```

        Returns a 3D tensor of shape [height, width, 3].
        """

        # squeeze last dim if it exists
        value = tf.squeeze(value)

        # quantize
        indices = tf.to_int32(tf.round(value * 255))

        # gather
        cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
        colors = tf.constant(cm.colors, dtype=tf.float32)
        value = tf.gather(colors, indices)
        return value

    def resize2D(self, x, target_shape):
        # input x = (batch, x, y, filter)
        x = tf.image.resize_images(x, size=target_shape[0:2])
        return x
    
    def resize3D(self, x, target_shape, zcut):
        # input x = (z, x, y, filter)
        x = k.stack([tf.image.resize_images(x[i], size=target_shape[1:3]) for i in range(target_shape[0]+np.sum(zcut))])  # x = (z, x, y, filter)
        x = x[zcut[0]:]
        if zcut[1] > 0: x = x[:-zcut[1]]
        return x
    
    def tile_patches_medical(self, x, shape):
        """
        Should use only one person's image as input
        """
        if len(shape) == 4:
        # For 3D
            # input x = (z, x, y, filter)
            x = tf.transpose(x, [0,3,1,2]) # x = (z, filter, x, y)
            x = k.reshape(x,[shape[0],shape[3],shape[1]*shape[2],1]) # (batch, z, filter, x*y, 1)
            x = tf.transpose(x, perm=[2,0,1,3])
            tiled_x = tf.batch_to_space_nd(x, shape[1:3], [[0,0],[0,0]])
        elif len(shape) == 3:
            # For 2D
            # input x = (batch, x, y, filter)
            nbatch = self.validation_data.batch_size
            x = tf.transpose(x, [0,3,1,2]) # x = (batch, filter, x, y)
            x = k.reshape(x,[nbatch,shape[2],shape[0]*shape[1],1]) # (batch, filter, x*y, 1)
            x = tf.transpose(x, perm=[2,0,1,3])
            tiled_x = tf.batch_to_space_nd(x, shape[0:2], [[0,0],[0,0]])
        else:
            raise ValueError('image must be 2D or 3D')
        return tiled_x
    
    def set_model(self, model):
        self.model = model
        if k.backend() == 'tensorflow':
            self.sess = k.get_session()
            
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:
                for weight in layer.weights:
                    mapped_weight_name = 'weight_%s' % weight.name.replace(':', '_')
                    # histogram
                    if self.write_weights_histogram: tf.summary.histogram(mapped_weight_name, weight)
                    # gradient histogram
                    if self.write_grads:
                        grads = model.optimizer.get_gradients(model.total_loss,
                                                              weight)

                        def is_indexed_slices(grad):
                            return type(grad).__name__ == 'IndexedSlices'
                        grads = [
                            grad.values if is_indexed_slices(grad) else grad
                            for grad in grads]
                        tf.summary.histogram('{}_grad'.format(mapped_weight_name), grads)
                        
                    if self.write_weights_images:
                        w_img = tf.squeeze(weight)
                        shape = k.int_shape(w_img)
                        if len(shape) == 2:  # dense layer kernel case
                            if shape[0] > shape[1]:
                                w_img = tf.transpose(w_img)
                                shape = k.int_shape(w_img)
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       shape[1],
                                                       1])
                        elif len(shape) == 3:  # 1d convnet case
                            if k.image_data_format() == 'channels_last':
                                # switch to channels_first to display
                                # every kernel as a separate image
                                w_img = tf.transpose(w_img, perm=[2, 0, 1])
                                shape = k.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0],
                                                       shape[1],
                                                       shape[2],
                                                       1])
                        elif len(shape) == 4: # conv2D
                            # input_dim * output_dim, width, hieght
                            w_img = tf.transpose(w_img, perm=[2, 3, 0, 1])
                            shape = k.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0]*shape[1],
                                                       shape[2],
                                                       shape[3],
                                                       1])
                        elif len(shape) == 5: # conv3D
                            # input_dim * output_dim*depth, width, hieght
                            w_img = tf.transpose(w_img, perm=[3, 4, 0, 1, 2])
                            shape = k.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0]*shape[1]*shape[2],
                                                       shape[3],
                                                       shape[4],
                                                       1])
                        elif len(shape) == 1:  # bias case
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       1,
                                                       1])
                        tf.summary.image(mapped_weight_name, w_img)

                if hasattr(layer, 'output'):
                    if isinstance(layer.output, list):
                        for i, output in enumerate(layer.output):
                            tf.summary.histogram('{}_out_{}'.format(layer.name, i), output)
                    else:
                        tf.summary.histogram('{}_out'.format(layer.name),
                                             layer.output)
            #################################################################################
            # image summary
            if self.write_images:
                if len(self.img_shape) == 4:
                    # for 3D images
                    input_shape = []
                    input_shape[:] = self.img_shape[:]
                    input_shape = np.array(input_shape)
                    input_shape[-1] += 2
                    input_shape[0] = input_shape[0] - np.sum(self.zcut)
                    input_shape[1:3] = input_shape[1:3] * self.downsampling_scale
                        
                    tot_pred_image = []
                    for i in range(self.batch_size):
                        # input images, GT, prediction
                        input_img = self.resize3D(model.inputs[0][i], target_shape= input_shape, zcut=self.zcut)
                        gt = self.resize3D(model.targets[0][i], target_shape= input_shape, zcut=self.zcut)
                        pred = self.resize3D(model.outputs[0][i], target_shape= input_shape, zcut=self.zcut)
                        pred_image = self.tile_patches_medical(k.concatenate([input_img, gt, pred], axis=-1), shape=input_shape) # output : [1,x*nrow, y*ncol, 1]
                        pred_image = self.colorize(pred_image, cmap='inferno') # output : [x*nrow, y*ncol, 3]
                        tot_pred_image.append(pred_image)
                    tot_pred_image = k.stack(tot_pred_image) # output : [batch, x*nrow, y*ncol, 3]
                elif len(self.img_shape) == 3:
                    # for 2D images
                    input_shape = []
                    input_shape[:] = self.img_shape[:]
                    input_shape = np.array(input_shape)
                    input_shape[-1] += 2
                    input_shape[0:2] = input_shape[0:2] * self.downsampling_scale
                    
                    input_img = self.resize2D(model.inputs[0], target_shape= input_shape)
                    gt = self.resize2D(model.targets[0], target_shape= input_shape)
                    pred = self.resize2D(model.outputs[0], target_shape= input_shape)
                    # input images, GT, prediction
                    tot_pred_image = self.tile_patches_medical(k.concatenate([input_img, gt, pred], axis=-1), shape=input_shape) # output : [batch_size, x*nrow, y*ncol, 1]
                    # TODO: fix
#                     tot_pred_image = self.colorize(tot_pred_image, cmap='inferno') # output : [batch_size, x*nrow, y*ncol, 3]
                shape = k.int_shape(tot_pred_image)
                assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                tf.summary.image('prediction', tot_pred_image, max_outputs=self.batch_size)
                                
            # TODO : center image
            #################################################################################
                
        self.merged = tf.summary.merge_all()
        #################################################################################
        # TODO: fix
        self.tf_physical_ids = tf.placeholder(tf.string, shape=(None,))
        self.summary_physical_ids = tf.summary.text('patient_ids', self.tf_physical_ids)
        #################################################################################
        
        # tensor graph & file write
        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

        #################################################################################
        # embedding : TODO
        if self.embeddings_freq:
            embeddings_layer_names = self.embeddings_layer_names

            if not embeddings_layer_names:
                embeddings_layer_names = [layer.name for layer in self.model.layers
                                          if type(layer).__name__ == 'Embedding']
            self.assign_embeddings = []
            embeddings_vars = {}

            self.batch_id = batch_id = tf.placeholder(tf.int32)
            self.step = step = tf.placeholder(tf.int32)

            for layer in self.model.layers:
                if layer.name in embeddings_layer_names:
                    embedding_input = self.model.get_layer(layer.name).output
                    embedding_size = int(np.prod(embedding_input.shape[1:]))
                    embedding_input = tf.reshape(embedding_input,
                                                 (step, embedding_size))
                    shape = (self.embeddings_data[0].shape[0], embedding_size)
                    embedding = tf.Variable(tf.zeros(shape),
                                            name=layer.name + '_embedding')
                    embeddings_vars[layer.name] = embedding
                    batch = tf.assign(embedding[batch_id:batch_id + step],
                                      embedding_input)
                    self.assign_embeddings.append(batch)

            self.saver = tf.train.Saver(list(embeddings_vars.values()))

            embeddings_metadata = {}

            if not isinstance(self.embeddings_metadata, str):
                embeddings_metadata = self.embeddings_metadata
            else:
                embeddings_metadata = {layer_name: self.embeddings_metadata
                                       for layer_name in embeddings_vars.keys()}

            config = projector.ProjectorConfig()

            for layer_name, tensor in embeddings_vars.items():
                embedding = config.embeddings.add()
                embedding.tensor_name = tensor.name

                if layer_name in embeddings_metadata:
                    embedding.metadata_path = embeddings_metadata[layer_name]

            projector.visualize_embeddings(self.writer, config)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if not self.validation_data and self.histogram_freq:
            raise ValueError("If printing histograms, validation_data must be "
                             "provided, and cannot be a generator.")
        if self.embeddings_data is None and self.embeddings_freq:
            raise ValueError("To visualize embeddings, embeddings_data must "
                             "be provided.")
            
        if self.validation_data and self.histogram_freq:
            if epoch == 0 or (epoch+1) % self.histogram_freq == 0:

                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [k.learning_phase()]

                for i in range(self.tb_data_steps):
                    x, y = val_data[i]
                    physical_ids = val_data.sampler.get_current_physical_id()
                    if type(x) != list:
                        x = [x]
                    if type(y) != list:
                        y = [y]
                    if self.model.uses_learning_phase:
                        batch_val = x + y + [np.ones(self.batch_size, dtype=np.float32) for tmp in range(len(self.model.sample_weights))] + [0.0]
                    else:
                        batch_val = x + y + [np.ones(self.batch_size, dtype=np.float32) for tmp in range(len(self.model.sample_weights))]
                    
                    assert len(batch_val) == len(tensors)
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, epoch)
                    
                    summary_patient_id = self.sess.run([self.summary_physical_ids], feed_dict={self.tf_physical_ids: physical_ids})
                    self.writer.add_summary(summary_patient_id[0], epoch)
                    
        if self.embeddings_freq and self.embeddings_data is not None:
            if epoch == 0 or epoch % self.embeddings_freq == 0:
                embeddings_data = self.embeddings_data
                for i in range(self.tb_data_steps):
                    if type(self.model.input) == list:
                        feed_dict = {model_input: embeddings_data[i][idx]
                                     for idx, model_input in enumerate(self.model.input)}
                    else:
                        feed_dict = {self.model.input: embeddings_data[i]}

                    feed_dict.update({self.batch_id: i, self.step: self.batch_size})

                    if self.model.uses_learning_phase:
                        feed_dict[k.learning_phase()] = False

                    self.sess.run(self.assign_embeddings, feed_dict=feed_dict)
                    self.saver.save(self.sess,
                                    os.path.join(self.log_dir, 'keras_embedding.ckpt'),
                                    epoch)
                    
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            if isinstance(value, np.ndarray):
                summary_value.simple_value = value.item()
            else:
                summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)        
        
        self.writer.flush()
        

######################################################################
# For classification
######################################################################
class TensorBoardClassificationWrapper(TensorBoardSegmentationWrapper):
    '''
    Sets the self.validation_data property for use with TensorBoard callback.
    
    Image Summary with multi-modal medical 3D volumes:  
        Thumbnail of nrow x ncol 2D images (of one person) 
            nrow: number of slice (z-axis)
            ncol: 
                   input images: number of modals
                   bottleneck images : number of filters
                   output images: 2 (GT, predict)
        TODO: fix one person as reference
    '''

    def __init__(self, validation_data, write_weights_histogram = True, write_weights_images=False, 
                 tb_data_steps=1, zcut=[0,0], downsampling_scale=1.,
                 **kwargs):
        super(TensorBoardClassificationWrapper, self).__init__(validation_data, write_weights_histogram, write_weights_images, 
                                                               tb_data_steps, zcut, downsampling_scale, **kwargs)
        
    def set_model(self, model):
        self.model = model
        if k.backend() == 'tensorflow':
            self.sess = k.get_session()
            
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:
                for weight in layer.weights:
                    mapped_weight_name = 'weight_%s' % weight.name.replace(':', '_')
                    # histogram
                    if self.write_weights_histogram: tf.summary.histogram(mapped_weight_name, weight)
                    # gradient histogram
                    if self.write_grads:
                        grads = model.optimizer.get_gradients(model.total_loss,
                                                              weight)

                        def is_indexed_slices(grad):
                            return type(grad).__name__ == 'IndexedSlices'
                        grads = [
                            grad.values if is_indexed_slices(grad) else grad
                            for grad in grads]
                        tf.summary.histogram('{}_grad'.format(mapped_weight_name), grads)
                        
                    if self.write_weights_images:
                        w_img = tf.squeeze(weight)
                        shape = k.int_shape(w_img)
                        if len(shape) == 2:  # dense layer kernel case
                            if shape[0] > shape[1]:
                                w_img = tf.transpose(w_img)
                                shape = k.int_shape(w_img)
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       shape[1],
                                                       1])
                        elif len(shape) == 3:  # 1d convnet case
                            if k.image_data_format() == 'channels_last':
                                # switch to channels_first to display
                                # every kernel as a separate image
                                w_img = tf.transpose(w_img, perm=[2, 0, 1])
                                shape = k.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0],
                                                       shape[1],
                                                       shape[2],
                                                       1])
                        elif len(shape) == 4: # conv2D
                            # input_dim * output_dim, width, hieght
                            w_img = tf.transpose(w_img, perm=[2, 3, 0, 1])
                            shape = k.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0]*shape[1],
                                                       shape[2],
                                                       shape[3],
                                                       1])
                        elif len(shape) == 5: # conv3D
                            # input_dim * output_dim*depth, width, hieght
                            w_img = tf.transpose(w_img, perm=[3, 4, 0, 1, 2])
                            shape = k.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0]*shape[1]*shape[2],
                                                       shape[3],
                                                       shape[4],
                                                       1])
                        elif len(shape) == 1:  # bias case
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       1,
                                                       1])
                        tf.summary.image(mapped_weight_name, w_img)

                if hasattr(layer, 'output'):
                    if isinstance(layer.output, list):
                        for i, output in enumerate(layer.output):
                            tf.summary.histogram('{}_out_{}'.format(layer.name, i), output)
                    else:
                        tf.summary.histogram('{}_out'.format(layer.name),
                                             layer.output)
            #################################################################################
            # image summary
            # TODO: progress vs stable
            if self.write_images:
                if len(self.img_shape) == 4:
                    # for 3D images
                    input_shape = []
                    input_shape[:] = self.img_shape[:]
                    input_shape = np.array(input_shape)
                    input_shape[0] = input_shape[0] - np.sum(self.zcut)
                    input_shape[1:3] = input_shape[1:3] * self.downsampling_scale
                    
                    tot_pred_image = []
                    for i in range(self.batch_size):
#                         title = tf.strings.format("predicted: {}, label: {}", model.outputs[0][i], model.targets[0][i])
                        input_img = self.resize3D(model.inputs[0][i], target_shape= input_shape, zcut=self.zcut)
                        pred_image = self.tile_patches_medical(input_img, shape=input_shape) # output : [1,x*nrow, y*ncol, 1]
#                         pred_image = pred_image[0]
                        pred_image = self.colorize(pred_image, cmap='inferno') # output : [x*nrow, y*ncol, 3]
                        tot_pred_image.append(pred_image)
                    tot_pred_image = k.stack(tot_pred_image) # output : [batch, x*nrow, y*ncol, 3]
                elif len(self.img_shape) == 3:
                    # for 2D images
                    input_shape = []
                    input_shape[:] = self.img_shape[:]
                    input_shape = np.array(input_shape)
                    input_shape[0:2] = input_shape[0:2] * self.downsampling_scale
                    
#                     tot_title = tf.strings.format("predicted: {}, label: {}", model.outputs[0], model.targets[0])
                    input_img = self.resize2D(model.inputs[0], target_shape= input_shape)
                    tot_pred_image = self.tile_patches_medical(input_img, shape=input_shape) # output : [batch_size, x*nrow, y*ncol, 1]
#                     tot_pred_image = tot_pred_image[0]
                    tot_pred_image = self.colorize(tot_pred_image, cmap='inferno') # output : [x*nrow, y*ncol, 3]
                shape = k.int_shape(tot_pred_image)
                assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                tf.summary.image('prediction', tot_pred_image, max_outputs=self.batch_size)
            
            # TODO : center image
            #################################################################################
            
        self.merged = tf.summary.merge_all()
        #################################################################################
        # tensor graph & file write
        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

        #################################################################################
        # embedding : TODO
        if self.embeddings_freq:
            embeddings_layer_names = self.embeddings_layer_names

            if not embeddings_layer_names:
                embeddings_layer_names = [layer.name for layer in self.model.layers
                                          if type(layer).__name__ == 'Embedding']
            self.assign_embeddings = []
            embeddings_vars = {}

            self.batch_id = batch_id = tf.placeholder(tf.int32)
            self.step = step = tf.placeholder(tf.int32)

            for layer in self.model.layers:
                if layer.name in embeddings_layer_names:
                    embedding_input = self.model.get_layer(layer.name).output
                    embedding_size = int(np.prod(embedding_input.shape[1:]))
                    embedding_input = tf.reshape(embedding_input,
                                                 (step, embedding_size))
                    shape = (self.embeddings_data[0].shape[0], embedding_size)
                    embedding = tf.Variable(tf.zeros(shape),
                                            name=layer.name + '_embedding')
                    embeddings_vars[layer.name] = embedding
                    batch = tf.assign(embedding[batch_id:batch_id + step],
                                      embedding_input)
                    self.assign_embeddings.append(batch)

            self.saver = tf.train.Saver(list(embeddings_vars.values()))

            embeddings_metadata = {}

            if not isinstance(self.embeddings_metadata, str):
                embeddings_metadata = self.embeddings_metadata
            else:
                embeddings_metadata = {layer_name: self.embeddings_metadata
                                       for layer_name in embeddings_vars.keys()}

            config = projector.ProjectorConfig()

            for layer_name, tensor in embeddings_vars.items():
                embedding = config.embeddings.add()
                embedding.tensor_name = tensor.name

                if layer_name in embeddings_metadata:
                    embedding.metadata_path = embeddings_metadata[layer_name]

            projector.visualize_embeddings(self.writer, config)
            
            
######################################################################
class TensorBoardClassificationAttentionWrapper(TensorBoardClassificationWrapper):
    '''
    Sets the self.validation_data property for use with TensorBoard callback.
    
    Image Summary with multi-modal medical 3D volumes:  
        Thumbnail of nrow x ncol 2D images (of one person) 
            nrow: number of slice (z-axis)
            ncol: 
                   input images: number of modals
                   bottleneck images : number of filters
                   output images: 2 (GT, predict)
        TODO: fix one person as reference
    '''

    def __init__(self, validation_data, write_weights_histogram = True, write_weights_images=False, 
                 tb_data_steps=1, zcut=[0,0], downsampling_scale=1.,
                 **kwargs):
        super(TensorBoardClassificationWrapper, self).__init__(validation_data, write_weights_histogram, write_weights_images, 
                                                               tb_data_steps, zcut, downsampling_scale, **kwargs)
    
    def set_model(self, model):
        self.model = model
        if k.backend() == 'tensorflow':
            self.sess = k.get_session()
            
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:
                for weight in layer.weights:
                    mapped_weight_name = 'weight_%s' % weight.name.replace(':', '_')
                    # histogram
                    if self.write_weights_histogram: tf.summary.histogram(mapped_weight_name, weight)
                    # gradient histogram
                    if self.write_grads:
                        grads = model.optimizer.get_gradients(model.total_loss,
                                                              weight)

                        def is_indexed_slices(grad):
                            return type(grad).__name__ == 'IndexedSlices'
                        grads = [
                            grad.values if is_indexed_slices(grad) else grad
                            for grad in grads]
                        tf.summary.histogram('{}_grad'.format(mapped_weight_name), grads)
                        
                    if self.write_weights_images:
                        w_img = tf.squeeze(weight)
                        shape = k.int_shape(w_img)
                        if len(shape) == 2:  # dense layer kernel case
                            if shape[0] > shape[1]:
                                w_img = tf.transpose(w_img)
                                shape = k.int_shape(w_img)
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       shape[1],
                                                       1])
                        elif len(shape) == 3:  # 1d convnet case
                            if k.image_data_format() == 'channels_last':
                                # switch to channels_first to display
                                # every kernel as a separate image
                                w_img = tf.transpose(w_img, perm=[2, 0, 1])
                                shape = k.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0],
                                                       shape[1],
                                                       shape[2],
                                                       1])
                        elif len(shape) == 4: # conv2D
                            # input_dim * output_dim, width, hieght
                            w_img = tf.transpose(w_img, perm=[2, 3, 0, 1])
                            shape = k.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0]*shape[1],
                                                       shape[2],
                                                       shape[3],
                                                       1])
                        elif len(shape) == 5: # conv3D
                            # input_dim * output_dim*depth, width, hieght
                            w_img = tf.transpose(w_img, perm=[3, 4, 0, 1, 2])
                            shape = k.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0]*shape[1]*shape[2],
                                                       shape[3],
                                                       shape[4],
                                                       1])
                        elif len(shape) == 1:  # bias case
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       1,
                                                       1])
                        tf.summary.image(mapped_weight_name, w_img)

                if hasattr(layer, 'output'):
                    if isinstance(layer.output, list):
                        for i, output in enumerate(layer.output):
                            tf.summary.histogram('{}_out_{}'.format(layer.name, i), output)
                    else:
                        tf.summary.histogram('{}_out'.format(layer.name),
                                             layer.output)
            #################################################################################
            # image summary
            # TODO: progress vs stable
            if self.write_images:
                if len(self.img_shape) == 4:
                    # for 3D images
                    input_shape = []
                    input_shape[:] = self.img_shape[:]
                    input_shape = np.array(input_shape)
                    input_shape[-1] += 1
                    input_shape[0] = input_shape[0] - np.sum(self.zcut)
                    input_shape[1:3] = input_shape[1:3] * self.downsampling_scale
                    
                    attention_layer = [l for l in model.layers if 'attention' in l.name][0]
                    tot_pred_image = []
                    for i in range(self.batch_size):
#                         title = tf.strings.format("predicted: {}, label: {}", model.outputs[0][i], model.targets[0][i])
                        input_img = self.resize3D(model.inputs[0][i], target_shape= input_shape, zcut=self.zcut)
                        attention = attention_layer.output[1][i]
                        ratio =  k.int_shape(model.inputs[0][i])[0] // k.int_shape(attention)[0]
                        attention = k.repeat_elements(attention, ratio, axis=0)
                        attention = self.resize3D(attention, target_shape= input_shape, zcut=self.zcut)
                        pred_image = self.tile_patches_medical(k.concatenate([self.normalize(input_img), self.normalize(attention)], axis=-1), 
                                                               shape=input_shape) # output : [1,x*nrow, y*ncol, 1]
#                         pred_image = pred_image[0]
                        pred_image = self.colorize(pred_image, cmap='inferno') # output : [x*nrow, y*ncol, 3]
                        tot_pred_image.append(pred_image)
                    tot_pred_image = k.stack(tot_pred_image) # output : [batch, x*nrow, y*ncol, 3]
                elif len(self.img_shape) == 3:
                    # for 2D images
                    input_shape = []
                    input_shape[:] = self.img_shape[:]
                    input_shape = np.array(input_shape)
                    input_shape[-1] += 1
                    input_shape[0:2] = input_shape[0:2] * self.downsampling_scale
                    
#                     tot_title = tf.strings.format("predicted: {}, label: {}", model.outputs[0], model.targets[0])
                    input_img = self.resize2D(model.inputs[0], target_shape= input_shape)
                    attention = self.resize2D(attention_layer.output[1], target_shape= input_shape)
                    tot_pred_image = self.tile_patches_medical(k.concatenate([self.normalize(input_img), self.normalize(attention)], axis=-1),
                                                               shape=input_shape) # output : [1,x*nrow, y*ncol, 1]
#                     tot_pred_image = tot_pred_image[0]
                    tot_pred_image = self.colorize(tot_pred_image, cmap='inferno') # output : [x*nrow, y*ncol, 3]
                shape = k.int_shape(tot_pred_image)
                assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                tf.summary.image('prediction', tot_pred_image, max_outputs=self.batch_size)
            
            # TODO : center image
            #################################################################################
            
        self.merged = tf.summary.merge_all()
        #################################################################################
        # tensor graph & file write
        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

        #################################################################################
        # embedding : TODO
        if self.embeddings_freq:
            embeddings_layer_names = self.embeddings_layer_names

            if not embeddings_layer_names:
                embeddings_layer_names = [layer.name for layer in self.model.layers
                                          if type(layer).__name__ == 'Embedding']
            self.assign_embeddings = []
            embeddings_vars = {}

            self.batch_id = batch_id = tf.placeholder(tf.int32)
            self.step = step = tf.placeholder(tf.int32)

            for layer in self.model.layers:
                if layer.name in embeddings_layer_names:
                    embedding_input = self.model.get_layer(layer.name).output
                    embedding_size = int(np.prod(embedding_input.shape[1:]))
                    embedding_input = tf.reshape(embedding_input,
                                                 (step, embedding_size))
                    shape = (self.embeddings_data[0].shape[0], embedding_size)
                    embedding = tf.Variable(tf.zeros(shape),
                                            name=layer.name + '_embedding')
                    embeddings_vars[layer.name] = embedding
                    batch = tf.assign(embedding[batch_id:batch_id + step],
                                      embedding_input)
                    self.assign_embeddings.append(batch)

            self.saver = tf.train.Saver(list(embeddings_vars.values()))

            embeddings_metadata = {}

            if not isinstance(self.embeddings_metadata, str):
                embeddings_metadata = self.embeddings_metadata
            else:
                embeddings_metadata = {layer_name: self.embeddings_metadata
                                       for layer_name in embeddings_vars.keys()}

            config = projector.ProjectorConfig()

            for layer_name, tensor in embeddings_vars.items():
                embedding = config.embeddings.add()
                embedding.tensor_name = tensor.name

                if layer_name in embeddings_metadata:
                    embedding.metadata_path = embeddings_metadata[layer_name]

            projector.visualize_embeddings(self.writer, config)
            
######################################################################   
class TensorBoardClassificationGuidedAttentionWrapper(TensorBoardClassificationAttentionWrapper):
    '''
    Sets the self.validation_data property for use with TensorBoard callback.
    
    Image Summary with multi-modal medical 3D volumes:  
        Thumbnail of nrow x ncol 2D images (of one person) 
            nrow: number of slice (z-axis)
            ncol: 
                   input images: number of modals
                   bottleneck images : number of filters
                   output images: 2 (GT, predict)
        TODO: fix one person as reference
    '''

    def __init__(self, validation_data, write_weights_histogram = True, write_weights_images=False, 
                 tb_data_steps=1, zcut=[0,0], downsampling_scale=1.,
                 **kwargs):
        super(TensorBoardSegmentationWrapper, self).__init__(**kwargs)
        self.write_weights_histogram = write_weights_histogram
        self.write_weights_images = write_weights_images
        self.tb_data_steps = tb_data_steps
        self.validation_data = validation_data
        self.img_shape = validation_data[0][0][0].shape[1:]
        self.zcut = zcut
        self.downsampling_scale = downsampling_scale
        
        if self.embeddings_data is None and self.validation_data:
            self.embeddings_data = self.validation_data
    
    def set_model(self, model):
        self.model = model
        if k.backend() == 'tensorflow':
            self.sess = k.get_session()
            
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:
                for weight in layer.weights:
                    mapped_weight_name = 'weight_%s' % weight.name.replace(':', '_')
                    # histogram
                    if self.write_weights_histogram: tf.summary.histogram(mapped_weight_name, weight)
                    # gradient histogram
                    if self.write_grads:
                        grads = model.optimizer.get_gradients(model.total_loss,
                                                              weight)

                        def is_indexed_slices(grad):
                            return type(grad).__name__ == 'IndexedSlices'
                        grads = [
                            grad.values if is_indexed_slices(grad) else grad
                            for grad in grads]
                        tf.summary.histogram('{}_grad'.format(mapped_weight_name), grads)
                        
                    if self.write_weights_images:
                        w_img = tf.squeeze(weight)
                        shape = k.int_shape(w_img)
                        if len(shape) == 2:  # dense layer kernel case
                            if shape[0] > shape[1]:
                                w_img = tf.transpose(w_img)
                                shape = k.int_shape(w_img)
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       shape[1],
                                                       1])
                        elif len(shape) == 3:  # 1d convnet case
                            if k.image_data_format() == 'channels_last':
                                # switch to channels_first to display
                                # every kernel as a separate image
                                w_img = tf.transpose(w_img, perm=[2, 0, 1])
                                shape = k.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0],
                                                       shape[1],
                                                       shape[2],
                                                       1])
                        elif len(shape) == 4: # conv2D
                            # input_dim * output_dim, width, hieght
                            w_img = tf.transpose(w_img, perm=[2, 3, 0, 1])
                            shape = k.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0]*shape[1],
                                                       shape[2],
                                                       shape[3],
                                                       1])
                        elif len(shape) == 5: # conv3D
                            # input_dim * output_dim*depth, width, hieght
                            w_img = tf.transpose(w_img, perm=[3, 4, 0, 1, 2])
                            shape = k.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0]*shape[1]*shape[2],
                                                       shape[3],
                                                       shape[4],
                                                       1])
                        elif len(shape) == 1:  # bias case
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       1,
                                                       1])
                        tf.summary.image(mapped_weight_name, w_img)

                if hasattr(layer, 'output'):
                    if isinstance(layer.output, list):
                        for i, output in enumerate(layer.output):
                            tf.summary.histogram('{}_out_{}'.format(layer.name, i), output)
                    else:
                        tf.summary.histogram('{}_out'.format(layer.name),
                                             layer.output)
            #################################################################################
            # image summary
            # TODO: progress vs stable
            if self.write_images:
                if len(self.img_shape) == 4:
                    # for 3D images
                    input_shape = []
                    input_shape[:] = self.img_shape[:]
                    input_shape = np.array(input_shape)
                    input_shape[-1] += 3
                    input_shape[0] = input_shape[0] - np.sum(self.zcut)
                    input_shape[1:3] = input_shape[1:3] * self.downsampling_scale
                    
                    attention_layer = [l for l in model.layers if 'attention' in l.name][0]
                    marginal_attention_layer = [l for l in model.layers if 'marginal_attention' in l.name][0]
                    prior_layer = [l for l in model.layers if 'prior' in l.name][0]
                    
                    tot_pred_image = []
                    prior = prior_layer.output[0]
                    prior = k.repeat_elements(prior, k.int_shape(model.inputs[0][0])[0] // k.int_shape(prior)[0], axis=0)
                    prior = self.resize3D(prior, target_shape= input_shape, zcut=self.zcut)
                    marginal_attention = marginal_attention_layer.output[0]
                    marginal_attention = k.repeat_elements(marginal_attention, k.int_shape(model.inputs[0][0])[0] // k.int_shape(marginal_attention)[0], axis=0)
                    marginal_attention = self.resize3D(marginal_attention, target_shape= input_shape, zcut=self.zcut)
                    for i in range(self.batch_size):
#                         title = tf.strings.format("predicted: {}, label: {}", model.outputs[0][i], model.targets[0][i])
                        input_img = self.resize3D(model.inputs[0][i], target_shape= input_shape, zcut=self.zcut)
                        attention = attention_layer.output[1][i]
                        attention = k.repeat_elements(attention, k.int_shape(model.inputs[0][i])[0] // k.int_shape(attention)[0], axis=0)
                        attention = self.resize3D(attention, target_shape= input_shape, zcut=self.zcut)
                        pred_image = self.tile_patches_medical(k.concatenate([self.normalize(input_img), self.normalize(attention),
                                                                              self.normalize(marginal_attention), self.normalize(prior)], axis=-1), 
                                                               shape=input_shape) # output : [1,x*nrow, y*ncol, 1]
#                         pred_image = pred_image[0]
                        pred_image = self.colorize(pred_image, cmap='inferno') # output : [x*nrow, y*ncol, 3]
                        tot_pred_image.append(pred_image)
                    tot_pred_image = k.stack(tot_pred_image) # output : [batch, x*nrow, y*ncol, 3]
                elif len(self.img_shape) == 3:
                    # for 2D images
                    input_shape = []
                    input_shape[:] = self.img_shape[:]
                    input_shape = np.array(input_shape)
                    input_shape[-1] += 3
                    input_shape[0:2] = input_shape[0:2] * self.downsampling_scale
                    
#                     tot_title = tf.strings.format("predicted: {}, label: {}", model.outputs[0], model.targets[0])
                    input_img = self.resize2D(model.inputs[0], target_shape= input_shape)
                    attention = self.resize2D(attention_layer.output[1], target_shape= input_shape)
                    prior = self.resize2D(prior_layer.output[0], target_shape= input_shape)
                    marginal_attention = self.resize2D(marginal_attention_layer.output[0], target_shape= input_shape)
                    tot_pred_image = self.tile_patches_medical(k.concatenate([self.normalize(input_img), self.normalize(attention), 
                                                                              self.normalize(marginal_attention), self.normalize(prior)], axis=-1), 
                                                               shape=input_shape) # output : [1,x*nrow, y*ncol, 1]
#                     tot_pred_image = tot_pred_image[0]
                    tot_pred_image = self.colorize(tot_pred_image, cmap='inferno') # output : [x*nrow, y*ncol, 3]
                shape = k.int_shape(tot_pred_image)
                assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                tf.summary.image('prediction', tot_pred_image, max_outputs=self.batch_size)
            
            # TODO : center image
            #################################################################################
            
        self.merged = tf.summary.merge_all()
        #################################################################################
        # tensor graph & file write
        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

        #################################################################################
        # embedding : TODO
        if self.embeddings_freq:
            embeddings_layer_names = self.embeddings_layer_names

            if not embeddings_layer_names:
                embeddings_layer_names = [layer.name for layer in self.model.layers
                                          if type(layer).__name__ == 'Embedding']
            self.assign_embeddings = []
            embeddings_vars = {}

            self.batch_id = batch_id = tf.placeholder(tf.int32)
            self.step = step = tf.placeholder(tf.int32)

            for layer in self.model.layers:
                if layer.name in embeddings_layer_names:
                    embedding_input = self.model.get_layer(layer.name).output
                    embedding_size = int(np.prod(embedding_input.shape[1:]))
                    embedding_input = tf.reshape(embedding_input,
                                                 (step, embedding_size))
                    shape = (self.embeddings_data[0].shape[0], embedding_size)
                    embedding = tf.Variable(tf.zeros(shape),
                                            name=layer.name + '_embedding')
                    embeddings_vars[layer.name] = embedding
                    batch = tf.assign(embedding[batch_id:batch_id + step],
                                      embedding_input)
                    self.assign_embeddings.append(batch)

            self.saver = tf.train.Saver(list(embeddings_vars.values()))

            embeddings_metadata = {}

            if not isinstance(self.embeddings_metadata, str):
                embeddings_metadata = self.embeddings_metadata
            else:
                embeddings_metadata = {layer_name: self.embeddings_metadata
                                       for layer_name in embeddings_vars.keys()}

            config = projector.ProjectorConfig()

            for layer_name, tensor in embeddings_vars.items():
                embedding = config.embeddings.add()
                embedding.tensor_name = tensor.name

                if layer_name in embeddings_metadata:
                    embedding.metadata_path = embeddings_metadata[layer_name]

            projector.visualize_embeddings(self.writer, config)
            
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
from keras.layers import Layer

#########################################################################################################################
# Attention Layers
###############################################################################################
### 3D Attention Layer
###############################################################################################
class Attention3D(Layer):
    import keras.backend as k
    def __init__(self, **kwargs):
        super(Attention3D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernelf = self.add_weight(name='convf', 
                                      shape=(1,1,1,input_shape[-1],1),
                                      initializer='glorot_uniform',
                                      trainable=True)
#         self.kernelg = self.add_weight(name='convg', 
#                                       shape=(1,1,1,input_shape[-1],input_shape[-1]//4),
#                                       initializer='glorot_uniform',
#                                       trainable=True)
        self.kernelh = self.add_weight(name='convh', 
                                      shape=(1,1,1,input_shape[-1],input_shape[-1]),
                                      initializer='glorot_uniform',
                                      trainable=True)
#         self.gamma_1 = self.add_weight(name='gamma_1', 
#                                       shape=(1,),
#                                       initializer='zeros',
#                                       trainable=True)
#         self.gamma_2 = self.add_weight(name='gamma_2', 
#                                       shape=(1,),
#                                       initializer='ones',
#                                       trainable=True)
        self.gamma = self.add_weight(name='gamma', 
                                      shape=(1,),
                                      initializer='zeros',
                                      trainable=True)
        super(Attention3D, self).build(input_shape)

    def call(self, x):
        import keras.backend as k
        import tensorflow as tf
        def hw_flatten(x): return tf.reshape(x,[-1, k.shape(x)[1]*k.shape(x)[2]*k.shape(x)[3], k.shape(x)[4]])
        
        f = k.conv3d(x, kernel=self.kernelf, padding='same') # [bs, t, h, w, c']
#         g = k.conv3d(x, kernel=self.kernelg, padding='same') # [bs, t, h, w, c']
        h = k.conv3d(x, kernel=self.kernelh, padding='same') # [bs, t, h, w, c]
#         s = k.sum(f*g, axis=4, keepdims=True)  # [bs, t , h , w] 
        s = f
        beta = k.sigmoid(s)  # attention map [bs, t , h , w]
        o = beta * h  # [bs, t, h, w, c]
#         gamma = (self.gamma_1)/(self.gamma_1 + self.gamma_2+k.epsilon())
        x = self.gamma  * o + (1-self.gamma) * x
        return [x, beta]

    def compute_output_shape(self, input_shape):
        return [input_shape, (input_shape[0], input_shape[1], input_shape[2], input_shape[3], 1)]
    
###############################################################################################
### 3D Attention Layer previous
###############################################################################################
class Attention3D_previous(Layer):
    import keras.backend as k
    def __init__(self, **kwargs):
        super(Attention3D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernelf = self.add_weight(name='convf', 
                                      shape=(1,1,1,input_shape[-1],input_shape[-1]//4),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.kernelg = self.add_weight(name='convg', 
                                      shape=(1,1,1,input_shape[-1],input_shape[-1]//4),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.kernelh = self.add_weight(name='convh', 
                                      shape=(1,1,1,input_shape[-1],input_shape[-1]),
                                      initializer='glorot_uniform',
                                      trainable=True)
#         self.gamma_1 = self.add_weight(name='gamma_1', 
#                                       shape=(1,),
#                                       initializer='zeros',
#                                       trainable=True)
#         self.gamma_2 = self.add_weight(name='gamma_2', 
#                                       shape=(1,),
#                                       initializer='ones',
#                                       trainable=True)
        self.gamma = self.add_weight(name='gamma', 
                                      shape=(1,),
                                      initializer='zeros',
                                      trainable=True)
        super(Attention3D, self).build(input_shape)

    def call(self, x):
        import keras.backend as k
        import tensorflow as tf
        def hw_flatten(x): return tf.reshape(x,[-1, k.shape(x)[1]*k.shape(x)[2]*k.shape(x)[3], k.shape(x)[4]])
        
        f = k.conv3d(x, kernel=self.kernelf, padding='same') # [bs, t, h, w, c']
        g = k.conv3d(x, kernel=self.kernelg, padding='same') # [bs, t, h, w, c']
        h = k.conv3d(x, kernel=self.kernelh, padding='same') # [bs, t, h, w, c]
        s = k.sum(f*g, axis=4, keepdims=True)  # [bs, t , h , w] 
        beta = k.sigmoid(s)  # attention map [bs, t , h , w]
        o = beta * h  # [bs, t, h, w, c]
#         gamma = (self.gamma_1)/(self.gamma_1 + self.gamma_2+k.epsilon())
        x = self.gamma  * o + (1-self.gamma) * x
        return [x, beta]

    def compute_output_shape(self, input_shape):
        return [input_shape, (input_shape[0], input_shape[1], input_shape[2], input_shape[3], 1)]
    
#########################################################################################################################
### 2D Self-Attention Layer
###############################################################################################
class Self_Attention2D(Layer):
    import keras.backend as k
    def __init__(self, **kwargs):
        super(Self_Attention2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernelf = self.add_weight(name='convf', 
                                      shape=(1,1,input_shape[-1],input_shape[-1]//4),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.kernelg = self.add_weight(name='convg', 
                                      shape=(1,1,input_shape[-1],input_shape[-1]//4),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.kernelh = self.add_weight(name='convh', 
                                      shape=(1,1,input_shape[-1],input_shape[-1]),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.gamma = self.add_weight(name='gamma', 
                                      shape=(1,),
                                      initializer='zeros',
                                      trainable=True)
        super(Self_Attention2D, self).build(input_shape)

    def call(self, x):
        import keras.backend as k
        import tensorflow as tf
        def hw_flatten(x): return tf.reshape(x,[-1, k.shape(x)[1]*k.shape(x)[2], k.shape(x)[3]])
        
        f = k.conv2d(x, kernel=self.kernelf, padding='same') # [bs, h, w, c']
        g = k.conv2d(x, kernel=self.kernelg, padding='same') # [bs, h, w, c']
        h = k.conv2d(x, kernel=self.kernelh, padding='same') # [bs, h, w, c]
        
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # [bs, N, N] where N = h * w
        beta = k.softmax(s)  # attention map [bs, N, N]
    
        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, c]
        o = tf.reshape(o, [-1, k.shape(x)[1], k.shape(x)[2], k.shape(x)[3]])  # [bs, h, w, c]
        x = self.gamma * o + x
        beta = tf.reshape(beta, [-1, k.shape(x)[1], k.shape(x)[2], 1])  # [bs, h, w, 1]
        return [x, beta]

    def compute_output_shape(self, input_shape):
        return [input_shape, (input_shape[0], input_shape[1], input_shape[2], 1)]

###############################################################################################
### 3D Self-Attention Layer
###############################################################################################
class Self_Attention3D(Layer):
    import keras.backend as k
    def __init__(self, **kwargs):
        super(Self_Attention3D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernelf = self.add_weight(name='convf', 
                                      shape=(1,1,1,input_shape[-1],input_shape[-1]//4),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.kernelg = self.add_weight(name='convg', 
                                      shape=(1,1,1,input_shape[-1],input_shape[-1]//4),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.kernelh = self.add_weight(name='convh', 
                                      shape=(1,1,1,input_shape[-1],input_shape[-1]),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.gamma = self.add_weight(name='gamma', 
                                      shape=(1,),
                                      initializer='zeros',
                                      trainable=True)
        super(Self_Attention3D, self).build(input_shape)

    def call(self, x):
        import keras.backend as k
        import tensorflow as tf
        def hw_flatten(x): return tf.reshape(x,[-1, k.shape(x)[1]*k.shape(x)[2]*k.shape(x)[3], k.shape(x)[4]])
        
        f = k.conv3d(x, kernel=self.kernelf, padding='same') # [bs, t, h, w, c']
        g = k.conv3d(x, kernel=self.kernelg, padding='same') # [bs, t, h, w, c']
        h = k.conv3d(x, kernel=self.kernelh, padding='same') # [bs, t, h, w, c]
        
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # [bs, N, N] where N = t * h * w
        beta = k.softmax(s)  # attention map [bs, N, N]
    
        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, c]
        o = tf.reshape(o, [-1, k.shape(x)[1], k.shape(x)[2], k.shape(x)[3], k.shape(x)[4]])  # [bs, t, h, w, c]
        x = self.gamma * o + x
        beta = tf.reshape(beta, [-1, k.shape(x)[1], k.shape(x)[2], k.shape(x)[3], 1])  # [bs, t, h, w, 1]
        return [x, beta]

    def compute_output_shape(self, input_shape):
        return [input_shape, (input_shape[0], input_shape[1], input_shape[2], input_shape[3], 1)]
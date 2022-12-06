import tensorflow as tf
import numpy as np
import math

c1 = 1000.0
c2 = 1e5

        
# =========== hard quantizers based on straight through estimator =============
def tf_binarize(x):
    # Reassign the sign gradient otherwise it will be almost everywhere equal to zero due to the sign function
    # using the straight through estimator
    with tf.compat.v1.get_default_graph().gradient_override_map({'Sign': 'Identity'}):
        return 0.5*(1.0+tf.sign(x))  #* tf.stop_gradient(stepsize/2)              #   <-- wrong sign doesn't return +1 for zero
        # return tf.sign(tf.sign(x)+1e-8) #   <-- this should be ok, ugly but okay

def tf_hard_quantizer(x, lsb, b):
    num_levels = 2**b
    L = int(num_levels/2)
    
    quantized_signal = 0.5*(tf_binarize(c1*x) - tf_binarize(-c1*x))
    
    for i in range(1,L):
        quantized_signal = quantized_signal + \
            tf_binarize(c1*(x-i*lsb)) - tf_binarize(c1*(-x-i*lsb))
            
    quantized_signal = tf.multiply(lsb,quantized_signal) 
    
    return quantized_signal

def tf_hard_quantizer_up(x, lsb, b):
    y = tf_sigmoid_soft_quantizer(x, lsb, b)
    L = 2**(b-1) - 1
    Q_up = y + 0.5*lsb + c2*tf_binarize(c1*(x-L*lsb))

    return Q_up

def tf_hard_quantizer_low(x, lsb, b):
    y = tf_sigmoid_soft_quantizer(x, lsb, b)
    L = 2**(b-1) - 1
    Q_low = y - 0.5*lsb - c2*tf_binarize(c1*(-x-L*lsb))

    return Q_low


# ==================== soft quantizers based on tanh =======================
def tf_tanh_soft_quantizer(x, lsb, b):
    num_levels = 2**b
    L = int(num_levels/2)
    
    quantized_signal = 0.5*((tf.tanh(c1*x)+1.0)/2 - (tf.tanh(-c1*x)+1.0)/2)
    
    for i in range(1,L):
        quantized_signal = quantized_signal + \
            (tf.tanh(c1*(x-i*lsb))+1.0)/2 - (tf.tanh(c1*(-x-i*lsb))+1.0)/2
            
    quantized_signal = tf.multiply(lsb,quantized_signal) 
    
    return quantized_signal

def tf_tanh_soft_quantizer_up(x, lsb, b):
    y = tf_sigmoid_soft_quantizer(x, lsb, b)
    L = 2**(b-1) - 1
    Q_up = y + 0.5*lsb + c2*(tf.tanh(c1*(x-L*lsb))+1.0)/2

    return Q_up

def tf_tanh_soft_quantizer_low(x, lsb, b):
    y = tf_sigmoid_soft_quantizer(x, lsb, b)
    L = 2**(b-1) - 1
    Q_low = y - 0.5*lsb - c2*(tf.tanh(c1*(-x-L*lsb))+1.0)/2

    return Q_low

# ==================== soft quantizers based on sigmoid =======================
def tf_sigmoid_soft_quantizer(x, lsb, b):
    num_levels = 2**b
    L = int(num_levels/2)
    
    quantized_signal = 0.5*(tf.sigmoid(c1*x) - tf.sigmoid(-c1*x))
    
    for i in range(1,L):
        quantized_signal = quantized_signal + \
            tf.sigmoid(c1*(x-i*lsb)) - tf.sigmoid(c1*(-x-i*lsb))
            
    quantized_signal = tf.multiply(lsb,quantized_signal) 
    
    return quantized_signal

def tf_sigmoid_soft_quantizer_up(x, lsb, b):
    y = tf_sigmoid_soft_quantizer(x, lsb, b)
    L = 2**(b-1) - 1
    Q_up = y + 0.5*lsb + c2*tf.sigmoid(c1*(x-L*lsb))

    return Q_up

def tf_sigmoid_soft_quantizer_low(x, lsb, b):
    y = tf_sigmoid_soft_quantizer(x, lsb, b)
    L = 2**(b-1) - 1
    Q_low = y - 0.5*lsb - c2*tf.sigmoid(c1*(-x-L*lsb))

    return Q_low

"""
# the following two functions are old versions:
def sigmoid_soft_quantizer2_up(x, lsb, b):
    num_levels = 2**b
    L = int(num_levels/2 - 1)
    Q_up = tf.zeros(tf.shape(x))
    for i in range(L):
        Q_up = Q_up + tf.sigmoid(c2*(x-i*lsb)) - tf.sigmoid(c2*(-x-(i+1)*lsb))
    Q_up = tf.multiply(lsb,Q_up) + infinity*tf.sigmoid(c3*(x-L*lsb))

    return Q_up

def sigmoid_soft_quantizer2_low(x, lsb, b):
    num_levels = 2**b
    L = int(num_levels/2 - 1)
    Q_low = tf.zeros(tf.shape(x))
    for i in range(L):
        Q_low = Q_low + tf.sigmoid(c2*(x-(i+1)*lsb)) - tf.sigmoid(c2*(-x-i*lsb))
    Q_low = tf.multiply(lsb,Q_low) - infinity*tf.sigmoid(c3*(-x-L*lsb))

    return Q_low
"""

# ==================== soft quantizers based on relu =======================
c1_relu = 0.01
c2_relu = 1e3

def tf_relu_soft_quantizer(x, lsb, b):
    quantized_signal = -(2**b - 1)*lsb*0.5
    B = 2**(b-1) - 1
    temp = 0.0
    
    for i in range(-B,B+1):
        temp = temp + tf.nn.relu(x + i*lsb + c1_relu) -\
                        tf.nn.relu(x + i*lsb - c1_relu)
    
    quantized_signal = quantized_signal + 0.5*(lsb/c1_relu)*temp
    
    return quantized_signal

    
def tf_relu_soft_quantizer_up(x, lsb, b):
    B = 2**(b-1) - 1
    q = tf_relu_soft_quantizer(x, lsb, b)
    q_up = q + 0.5*lsb + c2_relu*(tf.nn.relu(x - B*lsb + c1_relu) - \
                             tf.nn.relu(x - B*lsb - c1_relu))
    return q_up


def tf_relu_soft_quantizer_low(x, lsb, b):
    B = 2**(b-1) - 1
    q = tf_relu_soft_quantizer(x, lsb, b)
    q_low = q - 0.5*lsb - c2_relu*(tf.nn.relu(-x - B*lsb + c1_relu) - \
                              tf.nn.relu(-x - B*lsb - c1_relu))
    return q_low

# =================== Projector ====================
"""
for QPSK:  b = 1, lsb = tf.sqrt(2) 
for 16QAM: b = 2, lsb = 2.0/tf.sqrt(10)

Similar functions:
for QPSK:
    x_est = -1 + tf.nn.relu(temp2+t)/tf.abs(t) - tf.nn.relu(temp2-t)/tf.abs(t)
    output = (1.0/tf.sqrt(2.0))*x_est

for 16QAM:
    x_est = -3 + (tf.nn.relu(temp2+2/tf.sqrt(10.0)+t) - tf.nn.relu(temp2+2/tf.sqrt(10.0)-t) \
                  + tf.nn.relu(temp2+t) - tf.nn.relu(temp2-t) \
                  + tf.nn.relu(temp2-2/tf.sqrt(10.0)+t) - tf.nn.relu(temp2-2/tf.sqrt(10.0)-t))/tf.abs(t)
    output = (1.0/tf.sqrt(10.0))*x_est
"""
def tf_relu_projector(x, lsb, b, t):
    quantized_signal = -(2**b - 1)*lsb*0.5
    B = 2**(b-1) - 1
    temp = 0.0
    
    for i in range(-B,B+1):
        temp = temp + tf.nn.relu(x + i*lsb + t) -\
                        tf.nn.relu(x + i*lsb - t)
    
    quantized_signal = quantized_signal + 0.5*(lsb/t)*temp
    
    return quantized_signal

def tf_tanh_projector(x, lsb, b, t):
    quantized_signal = 0.5*lsb*(0.5*(tf.tanh(t*x)+1.0) - 0.5*(tf.tanh(-t*x)+1.0))
    B = 2**(b-1) - 1
    
    for i in range(1,B+1):
        quantized_signal = quantized_signal + \
            lsb*(0.5*(tf.tanh(t*(x-i*lsb))+1) - \
                   0.5*(tf.tanh(t*(-x-i*lsb))+1))

    return quantized_signal

def tf_sigmoid_projector(x, lsb, b, t):
    quantized_signal = 0.5*lsb*(tf.sigmoid(t*x) - tf.sigmoid(-t*x))
    B = 2**(b-1) - 1
    
    for i in range(1,B+1):
        quantized_signal = quantized_signal + \
            lsb*(tf.sigmoid(t*(x-i*lsb)) - \
                   tf.sigmoid(t*(-x-i*lsb)))

    return quantized_signal

# ============ quantizer and projector in numpy for testing ===================

def np_quantizer(input_signal,input_power,Delta,num_levels):
    lsb = Delta*np.sqrt(input_power) # least significant bit
    up_bounds = np.zeros((num_levels),float)
    low_bounds = np.zeros((num_levels),float)
    up_bounds[0] = (-0.5*num_levels+1)*lsb
    low_bounds[0] = -1e10
    up_bounds[num_levels-1] = 1e10
    low_bounds[num_levels-1] = (0.5*num_levels-1)*lsb
    for l in range(1,num_levels-1):
        up_bounds[l] =  (-0.5*num_levels+l+1)*lsb
        low_bounds[l] = (-0.5*num_levels+l)*lsb
    clip_lvl = lsb*(num_levels)/2 # clipping level
    clip_bound = clip_lvl-lsb/1e5
    clip_signal = np.clip(input_signal, -clip_bound, clip_bound)
    idx = np.floor(clip_signal/lsb)
    quantized_signal = lsb*idx + lsb/2.0
    Q_up = up_bounds[(idx+0.5*num_levels).astype(int)]
    Q_low = low_bounds[(idx+0.5*num_levels).astype(int)]
    return quantized_signal, Q_up, Q_low

def np_relu(x):
    return np.maximum(0,x)

def np_relu_projector(x, lsb, b, t):
    quantized_signal = -(2**b - 1)*lsb*0.5
    B = 2**(b-1) - 1
    temp = 0.0
    
    for i in range(-B,B+1):
        temp = temp + np_relu(x + i*lsb + t) -\
                        np_relu(x + i*lsb - t)
    
    quantized_signal = quantized_signal + 0.5*(lsb/t)*temp
    
    return quantized_signal

def np_tanh_projector(x, lsb, b, t):
    quantized_signal = 0.5*lsb*(0.5*(np.tanh(t*x)+1.0) - 0.5*(np.tanh(-t*x)+1.0))
    B = 2**(b-1) - 1
    
    for i in range(1,B+1):
        quantized_signal = quantized_signal + \
            lsb*(0.5*(np.tanh(t*(x-i*lsb))+1) - \
                   0.5*(np.tanh(t*(-x-i*lsb))+1))

    return quantized_signal

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def np_sigmoid_projector(x, lsb, b, t):
    quantized_signal = 0.5*lsb*(sigmoid(t*x) - sigmoid(-t*x))
    B = 2**(b-1) - 1
    
    for i in range(1,B+1):
        quantized_signal = quantized_signal + \
            lsb*(sigmoid(t*(x-i*lsb)) - \
                   sigmoid(t*(-x-i*lsb)))

    return quantized_signal

def np_QPSK_projector(x, t):
    x_est = -1 + np_relu(x+t)/t - np_relu(x-t)/t
    return (1.0/np.sqrt(2.0))*x_est

def np_16QAM_projector(x, t):
    x_est = -3 + (np_relu(x+2/np.sqrt(10.0)+t) - np_relu(x+2/np.sqrt(10.0)-t) + \
                  np_relu(x+t) - np_relu(x-t) + \
                  np_relu(x-2/np.sqrt(10.0)+t) - np_relu(x-2/np.sqrt(10.0)-t))/np.abs(t)
    return (1.0/np.sqrt(10.0))*x_est
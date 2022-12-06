import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from Quantizers import *
import random
from scipy.linalg import dft  # import DFT matrix

random.seed(20)


resolution_bit = 2   # number of bits in the uniform quantizer
num_levels = 2**resolution_bit
# step sizes that minimizes MSE for unit variance standard Gaussian random variables
#                      1-bit , 2-bit , 3-bit , 4-bit 
Delta_list = np.array([1.5958, 0.9957, 0.5860, 0.3352])
Delta = Delta_list[resolution_bit-1] # step size of resolution_bit

K = 4  # number of users, each user is equipped with only one antenna
N = 2 # number of antennas at the base station
Tt = 20
NK = N*K
NTt = N*Tt
KTt = K*Tt

DFT_matrix = dft(Tt)  # DFT matrix of size Tt
Theta = DFT_matrix[:,range(1,K+1)]
Theta_real = np.real(Theta)
Theta_imag = np.imag(Theta)
pilot_known = True

B = 1000 # batch training size

L = 8   # number of layers
c = 1.702

min_snr  = -10.0   # minimum simulated SNR
max_snr  = 30.0   # maximum simulated SNR
snr_step = 5.0   # SNR step
snr_dB = np.arange(min_snr, max_snr+snr_step, snr_step) # snr in dB

snr_len = len(snr_dB)

Xt_trained = np.zeros((2*Tt, K))
alpha_trained = np.zeros((snr_len,L))
beta_trained = np.zeros(snr_len)

sqrt2 = np.sqrt(2)
pi = np.pi

def generate_train_data():
    h_batch = np.zeros((B,2*NK))
    z_batch = np.zeros((B,2*NTt))
    N0_batch = np.zeros((B,1))

    for b in range(B):
        h_batch[b,:] = np.random.normal(0,np.sqrt(0.5),(2*NK))
        
        rho_dB = np.random.uniform(low=snr,high=snr,size=1)
        rho = 10**(rho_dB/10.0)
        N0 = 1/rho
        z_batch[b,:] = np.sqrt(0.5*N0)*np.random.normal(0,1,(2*NTt)) # noise in real domain
        N0_batch[b,:] = N0

    return h_batch, z_batch, N0_batch


for i in range(snr_len,0,-1):
    snr = snr_dB[i-1]
    
    tf.reset_default_graph()

    sess = tf.InteractiveSession()

    # ==================== Building nueral network model ==========================
    # placeholders
    h = tf.placeholder(tf.float32,shape=[B, 2*NK])
    z = tf.placeholder(tf.float32,shape=[B, 2*NTt])
    N0 = tf.placeholder(tf.float32,shape=[B,1])
    
    ones_mat = tf.ones([B, 2*NTt])
    
    # trainable parameters
    alpha = []
    for _ in range(L):
        alpha.append(tf.Variable(0.1,trainable=True))

    beta = tf.Variable(5.0,trainable=True)
    
    if pilot_known == True:
        Xt_real = tf.constant(Theta_real,dtype=tf.float32)
        Xt_imag = tf.constant(Theta_imag,dtype=tf.float32)
        Xt = tf.concat([Xt_real, Xt_imag],0)
    else:
            
        if i == snr_len:
            #Xt_real = tf.Variable(tf.random.normal([Tt, K], stddev=1e-2),trainable=True)
            #Xt_imag = tf.Variable(tf.random.normal([Tt, K], stddev=1e-2),trainable=True)
            
            if resolution_bit == 1:
                Xt = tf.get_variable('Xt', [2*Tt, K], initializer=tf.contrib.layers.xavier_initializer(), \
                                     constraint=keras.constraints.MinMaxNorm(min_value=Tt**0.5,max_value=Tt**0.5,axis=0),trainable=True)
            else:
                Xt = tf.get_variable('Xt', [2*Tt, K], initializer=tf.contrib.layers.xavier_initializer(), \
                                 constraint=keras.constraints.MaxNorm(max_value=Tt**0.5,axis=0),trainable=True)
        
            Xt_real = Xt[0:Tt,:]
            Xt_imag = Xt[Tt:2*Tt,:]
        else:
            Xt_real = tf.constant(Xt_trained[0:Tt,:],dtype=tf.float32)
            Xt_imag = tf.constant(Xt_trained[Tt:2*Tt,:],dtype=tf.float32)

    # building matrix P based on Xt_real and Xt_imag
    Imat = tf.eye(N)
    Imat_operator = tf.linalg.LinearOperatorFullMatrix(Imat)
    
    Xt_real_operator = tf.linalg.LinearOperatorFullMatrix(Xt_real)
    Xt_imag_operator = tf.linalg.LinearOperatorFullMatrix(Xt_imag)
    
    P_real_operator = tf.linalg.LinearOperatorKronecker([Xt_real_operator,Imat_operator])
    P_imag_operator = tf.linalg.LinearOperatorKronecker([Xt_imag_operator,Imat_operator])
    
    P_real = P_real_operator.to_dense()
    P_imag = P_imag_operator.to_dense()
    
    P = tf.concat([tf.concat([P_real, -P_imag], 1), \
                   tf.concat([P_imag,  P_real], 1)], 0)

    r = tf.matmul(h,P,transpose_b=True) + z
    

    #Qup = tf_tanh_soft_quantizer_up(r, Delta*tf.sqrt(0.5*(K+N0)), resolution_bit)
    #Qlow = tf_tanh_soft_quantizer_low(r, Delta*tf.sqrt(0.5*(K+N0)), resolution_bit)
    
    Qup = tf_relu_soft_quantizer_up(r, Delta*tf.sqrt(0.5*(K+N0)), resolution_bit)
    Qlow = tf_relu_soft_quantizer_low(r, Delta*tf.sqrt(0.5*(K+N0)), resolution_bit)
    
    # initial point
    h0 = tf.zeros([B,2*NK],dtype=tf.float32)
    
    hl = []
    hl.append(h0)
    
    
    for l in range(L):
        HX = tf.matmul(hl[l],P,transpose_b=True)
        temp = ones_mat - tf.sigmoid(beta*(HX-Qup)) - tf.sigmoid(beta*(HX-Qlow))
        hl.append(hl[l] + alpha[l]*tf.matmul(temp,P))
        
    hhat = hl[-1]
    loss = tf.reduce_mean(tf.square(hhat-h))
    
    startingLearningRate = 0.002
    decay_factor = 0.97
    decay_step = 100
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(startingLearningRate, global_step, decay_step, decay_factor, staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    init_op = tf.global_variables_initializer()
    
    
    if i == snr_len:
        train_iter = 8000
    else:
        train_iter = 6000

    sess.run(init_op)
    for j in range(train_iter): #num of train iter
        h_batch, z_batch, N0_batch  = generate_train_data()
        train_step.run(feed_dict={h: h_batch, z: z_batch, N0: N0_batch})
        if np.mod(j,10)==0:
            print('train_iter = '+str(j))
            print('loss = ' + 
                  str(10.0*np.log10(2.0*loss.eval(feed_dict={h: h_batch, z: z_batch, N0: N0_batch}))))
            print('-----')
    if i == snr_len:
        Xt_trained = Xt.eval()
    for l in range(L):
        alpha_trained[i-1,l] = alpha[l].eval()
        #beta_trained[i-1,l]  = beta[l].eval()
    beta_trained[i-1] = beta.eval()
    sess.close()


if pilot_known == True:
    filename = './DNN_trained_results/known_P/Xt_trained_'+str(K)+'K_'+str(L)+'L_'+str(Tt)+'Tt_'+str(resolution_bit)+'bits.txt'
    np.savetxt(filename, Xt_trained, fmt='%.15f', delimiter=', ') # save runtime to a text file
    
    filename = './DNN_trained_results/known_P/alpha_trained_'+str(K)+'K_'+str(L)+'L_'+str(Tt)+'Tt_'+str(resolution_bit)+'bits.txt'
    np.savetxt(filename, alpha_trained, fmt='%.15f', delimiter=', ') # save runtime to a text file
    
    filename = './DNN_trained_results/known_P/beta_trained_'+str(K)+'K_'+str(L)+'L_'+str(Tt)+'Tt_'+str(resolution_bit)+'bits.txt'
    np.savetxt(filename, beta_trained, fmt='%.15f', delimiter=', ') # save runtime to a text file
else:
    filename = './DNN_trained_results/trainable_P/Xt_trained_'+str(K)+'K_'+str(L)+'L_'+str(Tt)+'Tt_'+str(resolution_bit)+'bits.txt'
    np.savetxt(filename, Xt_trained, fmt='%.15f', delimiter=', ') # save runtime to a text file
    
    filename = './DNN_trained_results/trainable_P/alpha_trained_'+str(K)+'K_'+str(L)+'L_'+str(Tt)+'Tt_'+str(resolution_bit)+'bits.txt'
    np.savetxt(filename, alpha_trained, fmt='%.15f', delimiter=', ') # save runtime to a text file
    
    filename = './DNN_trained_results/trainable_P/beta_trained_'+str(K)+'K_'+str(L)+'L_'+str(Tt)+'Tt_'+str(resolution_bit)+'bits.txt'
    np.savetxt(filename, beta_trained, fmt='%.15f', delimiter=', ') # save runtime to a text file


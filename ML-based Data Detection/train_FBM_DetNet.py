import numpy as np
import tensorflow as tf
import random
from Quantizers import *

random.seed(100)

mod_scheme = 'QPSK' # modulation schemes: 'BPSK','QPSK','16QAM','64QAM'
if mod_scheme=='BPSK':
    Alphabet = np.array([-1.0, 1.0])
    constel_size = 2
elif mod_scheme=='QPSK':
    Alphabet = np.array([-1.0-1.0j, -1.0+1.0j, 1.0-1.0j, 1.0+1.0j])
    constel_size = 4
    constel_size_real = 2;
elif mod_scheme=='8PSK':
    Alphabet = np.array([np.exp(0.0), np.exp(1.0j*np.pi/4.0), np.exp(3.0j*np.pi/4.0),\
                         np.exp(1.0j*np.pi/2.0), np.exp(1.0j*7.0*np.pi/4.0), \
                         np.exp(1.0j*3.0*np.pi/2.0), np.exp(1.0j*np.pi), \
                         np.exp(1.0j*5.0*np.pi/4.0)])
    constel_size = 8
elif mod_scheme=='16QAM':
    Alphabet = np.array([-3.0-3.0j, -3.0-1.0j,  -3.0+3.0j,  -3.0+1.0j, \
                         -1.0-3.0j, -1.0-1.0j,  -1.0+3.0j,  -1.0+1.0j, \
                         +3.0-3.0j, +3.0-1.0j,  +3.0+3.0j,  +3.0+1.0j, \
                         +1.0-3.0j, +1.0-1.0j,  +1.0+3.0j,  +1.0+1.0j])
    constel_size = 16
    constel_size_real = 4;

Alphabet = Alphabet/np.sqrt(np.mean(np.abs(Alphabet)**2)); # normalize symbol energy
card = Alphabet.shape[0] # constellation size
bps = np.log2(card)      # number of bits per symbol
power = np.ones((card,1))*(2**np.arange(np.log2(card)))
# bit sequence of the constellation
bits = np.floor((np.array(range(card)).reshape(card,1)%(2*power))/power).T

Kb = 41 # K and b values (last digit is b, the rest is K)

if mod_scheme == 'QPSK':
    N = 32 # number of antennas at the base station
elif mod_scheme == '16QAM':
    N = 64
    
if Kb == 41:
    K = 4
    resolution_bit = 1
    L = 8
    Lt = 8
if Kb == 81:
    K = 8
    resolution_bit = 1
    L = 16
    Lt = 16
if Kb == 82:
    K = 8
    resolution_bit = 2
    L = 16
    Lt = 16
if Kb == 162:
    K = 16
    resolution_bit = 2
    L = 24
    Lt = 24
if Kb == 163:
    K = 16
    resolution_bit = 3
    L = 24
    Lt = 24
if Kb == 243:
    K = 24
    resolution_bit = 3
    L = 32
    Lt = 32
    
if Kb == 51:
    K = 5
    resolution_bit = 1 # resolution bit
    L = 10 # number of layers for data detection network
    Lt = 10 # number of layers for channel estimation network
if Kb == 102:
    K = 10
    resolution_bit = 2 # resolution bit
    L = 15 # number of layers for data detection network
    Lt = 15 # number of layers for channel estimation network
if Kb == 153:
    K = 15
    resolution_bit = 3 # resolution bit
    L = 20 # number of layers for data detection network
    Lt = 20 # number of layers for channel estimation network
if Kb == 203:
    K = 20
    resolution_bit = 3
    L = 25
    Lt = 25


num_levels = 2**resolution_bit

# step sizes that minimizes MSE for unit variance standard Gaussian random variables
#                      1-bit , 2-bit , 3-bit , 4-bit 
Delta_list = np.array([1.5958, 0.9957, 0.5860, 0.3352])
Delta = Delta_list[resolution_bit-1] # step size of resolution_bit


B = 1000 # batch training size
#L = 15   # number of layers
c = 1.702
min_snr  = -10.0   # minimum simulated SNR
max_snr  = 30.0   # maximum simulated SNR
snr_step = 5.0   # SNR step
snr_dB = np.arange(min_snr, max_snr+snr_step, snr_step) # snr in dB
snr = 10**(snr_dB/10.0)  # snr in linear scale

snr_len = len(snr_dB)

perfect_CSI = False

known_P = False

alpha_trained = np.zeros((snr_len,L))
beta_trained = np.zeros(snr_len)
t_trained = np.zeros((snr_len,L))

Tt = 5*K

if perfect_CSI == False:
    if known_P == True:
        filename_t1 = './ChaNet_trained_results/known_P/Xt_trained_'+str(K)+'K_'+str(Lt)+'L_'+str(Tt)+'Tt_'+str(resolution_bit)+'bits.txt'
        Xt_trained = np.genfromtxt(filename_t1, delimiter=', ')
        
        filename_t2 = './ChaNet_trained_results/known_P/alpha_trained_'+str(K)+'K_'+str(Lt)+'L_'+str(Tt)+'Tt_'+str(resolution_bit)+'bits.txt'
        alphat_trained = np.genfromtxt(filename_t2, delimiter=', ')
        
        filename_t3 = './ChaNet_trained_results/known_P/beta_trained_'+str(K)+'K_'+str(Lt)+'L_'+str(Tt)+'Tt_'+str(resolution_bit)+'bits.txt'
        betat_trained = np.genfromtxt(filename_t3, delimiter=', ')
    else:
        filename_t1 = './ChaNet_trained_results/trainable_P/Xt_trained_'+str(K)+'K_'+str(Lt)+'L_'+str(Tt)+'Tt_'+str(resolution_bit)+'bits.txt'
        Xt_trained = np.genfromtxt(filename_t1, delimiter=', ')
        
        filename_t2 = './ChaNet_trained_results/trainable_P/alpha_trained_'+str(K)+'K_'+str(Lt)+'L_'+str(Tt)+'Tt_'+str(resolution_bit)+'bits.txt'
        alphat_trained = np.genfromtxt(filename_t2, delimiter=', ')
        
        filename_t3 = './ChaNet_trained_results/trainable_P/beta_trained_'+str(K)+'K_'+str(Lt)+'L_'+str(Tt)+'Tt_'+str(resolution_bit)+'bits.txt'
        betat_trained = np.genfromtxt(filename_t3, delimiter=', ')

    Xt_real = Xt_trained[0:Tt,:]
    Xt_imag = Xt_trained[Tt:2*Tt,:]

    #print(np.linalg.norm(Xt_real + 1j*Xt_imag,axis=0))

    Xt = np.concatenate((np.concatenate((Xt_real, -Xt_imag),1), \
                         np.concatenate((Xt_imag,  Xt_real),1)),0)
    OneMat = np.ones((2*Tt,N),float)

## ------------------------ Channel estimator -------------------------
def sigmoid(x):
    return np.reciprocal(1.0+np.exp(-x))

def FBM_ChaNet(Qupt, Qlowt, alphat, betat):
    Hhat = np.zeros((2*K,N),float)
    for l in range(Lt):
        XtH = np.matmul(Xt,Hhat)
        Hhat = Hhat + alphat[l]*np.matmul(Xt.T,(OneMat - sigmoid(betat*(XtH - Qupt)) - \
                         sigmoid(betat*(XtH - Qlowt))))
    Hhat_real = np.transpose(Hhat[0:K,:])
    Hhat_imag = np.transpose(Hhat[K:2*K,:])
    Hhat_d = np.concatenate((np.concatenate((Hhat_real,-Hhat_imag),1), \
                             np.concatenate((Hhat_imag, Hhat_real),1)),0)
    return Hhat_d

def generate_train_data():
    H_batch = np.zeros((B,2*N,2*K))
    Qup_batch = np.zeros((B,2*N,1))
    Qlow_batch = np.zeros((B,2*N,1))
    X_batch = np.zeros((B,2*K,1))
    rho_batch = np.zeros((B,1,1))
    
    rho_dB = np.random.uniform(low=snr,high=snr,size=1)
    rho = 10**(rho_dB/10.0)
    N0 = 1/rho
            
    chanest_error = 0.0
    for b in range(B):
        if np.mod(b,100) == 0:
            H_real = np.random.normal(0,np.sqrt(0.5),(N,K))   # real part of channel
            H_imag = np.random.normal(0,np.sqrt(0.5),(N,K))   # imaginary part of channel
            H = np.concatenate((np.concatenate((H_real,-H_imag),1), \
                                np.concatenate((H_imag, H_real),1)),0)
                
            if perfect_CSI == False:
                Ht = np.concatenate((H_real.T, H_imag.T),0)
                
                # transmission phase
                Zt = np.random.normal(0,np.sqrt(0.5*N0),(2*Tt,N)) # Noise in real domain
                Rt = np.matmul(Xt, Ht) + Zt # received data signal in complex domain
                _, Qupt, Qlowt = np_quantizer(Rt, 0.5*(K+N0),Delta,num_levels)
            
                H_hat = FBM_ChaNet(Qupt, Qlowt, alphat_trained[i-1,:], betat_trained[i-1])
                chanest_error = chanest_error + np.square(np.linalg.norm(H-H_hat,ord='fro'))/(2*N*K)
            else:
                H_hat = H
            
        # data transmission phase
        tx_idx = np.random.randint(0,card,(K,1)) # indices of data symbols
        x_complex = Alphabet[tx_idx] # transmitted signal in complex domain
        x = np.concatenate((np.real(x_complex),np.imag(x_complex)),0) # transmitted singal in real domain
    
        noise = np.sqrt(0.5*N0)*np.random.normal(0,1,(2*N,1)) # noise in real domain
        r = np.matmul(H,x) + noise   # received signal
        quantized_signal, q_up, q_low = np_quantizer(r,0.5*(K+N0),Delta,num_levels)

        H_batch[b,:,:] = H_hat
        Qup_batch[b,:,:] = q_up
        Qlow_batch[b,:,:] = q_low
        X_batch[b,:,:] = x
        rho_batch[b,:,:] = rho
    #print(10.0*np.log10(chanest_error/(B/100)))
    return H_batch, Qup_batch, Qlow_batch, X_batch, rho_batch

for i in range(snr_len,0,-1):
    snr = snr_dB[i-1]
    print('snr = '+str(snr)+' dB')
    tf.reset_default_graph()

    sess = tf.InteractiveSession()
    # ==================== Building nueral network model ==========================
    # placeholders
    H = tf.placeholder(tf.float32,shape=[B, 2*N, 2*K])
    Qup  = tf.placeholder(tf.float32,shape=[B, 2*N, 1])
    Qlow  = tf.placeholder(tf.float32,shape=[B, 2*N, 1])
    X  = tf.placeholder(tf.float32,shape=[B, 2*K, 1  ])
    rho = tf.placeholder(tf.float32,shape=[B,1,1])
    
    ones_mat = tf.ones([B, 2*N, 1])
    
    # trainable parameters
    alpha = []
    t = []
    for _ in range(L):
        alpha.append(tf.Variable(0.01,trainable=True))
        if mod_scheme == 'QPSK':
            t.append(tf.Variable(0.5,trainable=True)) # relu-based projector
            #t.append(tf.Variable(10.0,trainable=True)) # tanh-based projector
        elif mod_scheme == '16QAM':
            # --> relu-based projector
            t.append(tf.Variable(initial_value=0.15, \
                                constraint=lambda z: tf.clip_by_value(z, 1e-10, 1e2), trainable=True))
            # --> tanh-based projector
            #t.append(tf.Variable(initial_value=10.0, \
            #                    constraint=lambda z: tf.clip_by_value(z, 1e-10, 1e2), trainable=True))
        
    beta = tf.Variable(5.0,trainable=True)
    
    # initial point
    X0 = tf.zeros([B,2*K,1],dtype=tf.float32)
    
    Xl = []
    Xl.append(X0)
    LOSS = []
    for l in range(L):
        HX = tf.matmul(H,Xl[l])
        temp1 = ones_mat - tf.sigmoid(beta*(HX-Qup)) - tf.sigmoid(beta*(HX-Qlow))
        temp2 = (Xl[l] + alpha[l]*tf.matmul(tf.transpose(H,perm=[0,2,1]),temp1))

        #Xl.append(temp2)
        if mod_scheme=='QPSK':
            x_est = tf_relu_projector(temp2, tf.sqrt(2.0), 1, t[l])
            Xl.append(x_est)
        if mod_scheme=='16QAM':
            x_est = tf_relu_projector(temp2, 2.0/tf.sqrt(10.0), 2, t[l])
            Xl.append(x_est)
    
    Xhat = Xl[-1]
    loss = tf.reduce_mean(tf.square(Xhat-X))
    
    startingLearningRate = 0.002
    decay_factor = 0.97
    decay_step = 100
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(startingLearningRate, global_step, decay_step, decay_factor, staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    init_op = tf.global_variables_initializer()


    train_iter = 2000

    sess.run(init_op)
    for j in range(train_iter): #num of train iter
        H_batch, Qup_batch, Qlow_batch, X_batch, rho_batch = generate_train_data()
        train_step.run(feed_dict={H: H_batch, Qup: Qup_batch, Qlow: Qlow_batch, X: X_batch, rho: rho_batch})
        if np.mod(j+1,2)==0:
            print('train_iter = '+str(j+1))
            print('loss = ' + \
                  str(loss.eval(feed_dict={H: H_batch, Qup: Qup_batch, Qlow: Qlow_batch, X: X_batch, rho: rho_batch})))
            #print('---------------------------')
    for ll in range(L):
        alpha_trained[i-1,ll] = alpha[ll].eval()
        t_trained[i-1,ll] = t[ll].eval()
    beta_trained[i-1] = beta.eval()
    print('=======================================')
    sess.close()


if perfect_CSI == True:
    filename = './DetNet_trained_results/perfect_CSI/alpha_trained_'+mod_scheme+'_'+str(K)+'K_'+str(N)+'N_'+str(L)+'L_'+str(resolution_bit)+'bits.txt'
    np.savetxt(filename, alpha_trained, fmt='%.15f', delimiter=', ') # save runtime to a text file
    
    filename = './DetNet_trained_results/perfect_CSI/beta_trained_'+mod_scheme+'_'+str(K)+'K_'+str(N)+'N_'+str(L)+'L_'+str(resolution_bit)+'bits.txt'
    np.savetxt(filename, beta_trained, fmt='%.15f', delimiter=', ') # save runtime to a text file
    
    filename = './DetNet_trained_results/perfect_CSI/t_trained_'+mod_scheme+'_'+str(K)+'K_'+str(N)+'N_'+str(L)+'L_'+str(resolution_bit)+'bits.txt'
    np.savetxt(filename, t_trained, fmt='%.15f', delimiter=', ') # save runtime to a text file
else:
    filename = './DetNet_trained_results/estimated_CSI/alpha_trained_'+mod_scheme+'_'+str(K)+'K_'+str(N)+'N_'+str(L)+'L_'+str(resolution_bit)+'bits.txt'
    np.savetxt(filename, alpha_trained, fmt='%.15f', delimiter=', ') # save runtime to a text file
    
    filename = './DetNet_trained_results/estimated_CSI/beta_trained_'+mod_scheme+'_'+str(K)+'K_'+str(N)+'N_'+str(L)+'L_'+str(resolution_bit)+'bits.txt'
    np.savetxt(filename, beta_trained, fmt='%.15f', delimiter=', ') # save runtime to a text file
    
    filename = './DetNet_trained_results/estimated_CSI/t_trained_'+mod_scheme+'_'+str(K)+'K_'+str(N)+'N_'+str(L)+'L_'+str(resolution_bit)+'bits.txt'
    np.savetxt(filename, t_trained, fmt='%.15f', delimiter=', ') # save runtime to a text file





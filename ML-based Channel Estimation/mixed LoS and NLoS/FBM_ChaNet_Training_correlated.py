import numpy as np
import tensorflow as tf
from scipy.linalg import eigh
from tensorflow import keras
import random
from scipy.linalg import dft  # import DFT matrix
from Quantizers import *

random.seed(0)

resolution_bit = 2 # number of bits in the uniform quantizer
num_levels = 2**resolution_bit
# step sizes that minimizes MSE for unit variance standard Gaussian random variables
#                      1-bit , 2-bit , 3-bit , 4-bit 
Delta_list = np.array([1.5958, 0.9957, 0.5860, 0.3352])
Delta = Delta_list[resolution_bit-1] # step size of resolution_bit

K = 4  # number of users, each user is equipped with only one antenna
N = 32 # number of antennas at the base station
Tt = 20
NK = N*K
NTt = N*Tt
KTt = K*Tt

B = 1500 # batch training size
L = 8    # number of layers
c = 1.702

filename1 = 'Covariance_Data\Lower_mat_'+str(N)+'N.txt'
filename2 = 'Covariance_Data\Cov_mat_'+str(N)+'N.txt'

Lower_mat = np.genfromtxt(filename1, delimiter=', ')
Lower_mat = Lower_mat[:,0:N] + 1j*Lower_mat[:,N:2*N]

Cov_mat = np.genfromtxt(filename2, delimiter=', ')
Cov_mat = Cov_mat[:,0:N] + 1j*Cov_mat[:,N:2*N]

M = int(np.shape(Lower_mat)[0]/N)

DFT_matrix = dft(Tt)  # DFT matrix of size Tt
Theta = DFT_matrix[:,range(1,K+1)]
Theta_real = np.real(Theta)
Theta_imag = np.imag(Theta)
Xt0 = np.concatenate((Theta_real,Theta_imag),0)
pilot_known = True

min_snr  = -5.0   # minimum simulated SNR
max_snr  = 30.0   # maximum simulated SNR
snr_step = 5.0   # SNR step
snr_dB = np.arange(min_snr, max_snr+snr_step, snr_step) # snr in dB

snr_len = len(snr_dB)

Xt_trained = np.zeros((2*Tt, K))
alpha_trained = np.zeros((snr_len,L))
beta_trained = np.zeros(snr_len)

c2 = 1000
infinity = 1e5

sqrt2 = np.sqrt(2)
pi = np.pi
    
#filename1 = './DNN_trained_results/trainable_P/Xt_trained_'+str(K)+'K_'+str(L)+'L_'+str(Tt)+'Tt_'+str(resolution_bit)+'bits.txt'
#Xt_trained = np.genfromtxt(filename1, delimiter=', ')
    

def Quantizer_up(x, input_power):
    lsb = Delta*tf.sqrt(input_power) # least significant bit
    L = int(num_levels/2 - 1)
    Q_up = tf.zeros(tf.shape(x))
    for i in range(L):
        Q_up = Q_up + tf.sigmoid(c2*(x-i*lsb)) - tf.sigmoid(c2*(-x-(i+1)*lsb))
    Q_up = tf.multiply(lsb,Q_up) + infinity*tf.sigmoid(c2*(x-L*lsb))

    return Q_up

def Quantizer_low(x, input_power):
    lsb = Delta*tf.sqrt(input_power) # least significant bit
    L = int(num_levels/2 - 1)
    Q_low = tf.zeros(tf.shape(x))
    for i in range(L):
        Q_low = Q_low + tf.sigmoid(c2*(x-(i+1)*lsb)) - tf.sigmoid(c2*(-x-i*lsb))
    Q_low = tf.multiply(lsb,Q_low) - infinity*tf.sigmoid(c2*(-x-L*lsb))

    return Q_low


def generate_train_data():
    h_batch = np.zeros((B,2*NK))
    #Cinv_batch = np.zeros((B,2*NK,2*NK))
    z_batch = np.zeros((B,2*NTt))
    rx_pow_batch = np.zeros((B,1))
    

    for p in range(B):
    
        # LoS component
        max_dis = 1000  # meters
        min_dis = 10  # meters
        dis = np.random.uniform(min_dis, max_dis, K)
        
        Kapa = 10**((13 - 0.03*dis)/10) # Rician factor
        
        shf = np.random.normal(0,4,K) # shadow fading
        gamma = 10**((-30.18 - 26*np.log10(dis) + shf)/10) # large-scale fading
        gamma = K*gamma/np.sum(gamma) # normalized        
        
        theta = np.random.uniform(-np.pi/3, np.pi/3, K)[None,:] # Angle-of-Arrival
        LoS_phase_shift = np.random.uniform(0,2*np.pi) # random LoS phase shift
        
        H_LoS = np.exp(1j*LoS_phase_shift)*\
            np.exp(1j*2.0*np.pi*0.5*(np.asarray(range(N))[:,None])*np.sin(theta))
        H_LoS = H_LoS*(np.sqrt(Kapa*gamma/(Kapa + 1.0))[None,:])
        
        H_NLoS_real = np.random.normal(0,np.sqrt(0.5),(N,K)) # real part of channel
        H_NLoS_imag = np.random.normal(0,np.sqrt(0.5),(N,K)) # imaginary part of channel
        H_NLoS = np.sqrt(gamma/(Kapa + 1.0))*(H_NLoS_real + 1j*H_NLoS_imag) # complex channel
        
        h_cplx = np.zeros((NK,1),dtype=complex)
        m_list = np.random.randint(0,M,K)
        for k in range(K):
            m = m_list[k]
            Lmat_k = Lower_mat[m*N:(m+1)*N,:]
            h_cplx[k*N:(k+1)*N, 0] = np.matmul(Lmat_k,H_NLoS[:,k]) + H_LoS[:,k]
        h = np.concatenate((np.real(h_cplx), np.imag(h_cplx)),0)
        
        h_batch[p,:] = np.squeeze(h)
        
        rho_dB = np.random.uniform(low=snr,high=snr,size=1)
        rho = 10**(rho_dB/10.0)
        N0 = 1/rho
        z_batch[p,:] = np.sqrt(0.5*N0)*np.random.normal(0,1,(2*NTt)) # noise in real domain
        rx_pow_batch[p,0] = 0.5*(K + N0)
            
    return h_batch, z_batch, rx_pow_batch

#snr = 10.0
#h_batch, z_batch, N0_batch = generate_train_data()


for i in range(snr_len,0,-1):
    snr = snr_dB[i-1]
    
    tf.reset_default_graph()

    sess = tf.InteractiveSession()
    
    # ==================== Building nueral network model ==========================
    # placeholders
    h = tf.placeholder(tf.float32,shape=[B, 2*NK])
    z = tf.placeholder(tf.float32,shape=[B, 2*NTt])
    rx_pow = tf.placeholder(tf.float32,shape=[B,1])
    
    ones_mat = tf.ones([B, 2*NTt])
    
    # trainable parameters
    alpha = []
    if i == snr_len:
        for l1 in range(L):
            alpha.append(tf.Variable(0.1,trainable=True))
        
        beta = tf.Variable(5.0,trainable=True)
    else:
        for l1 in range(L):
            alpha.append(tf.Variable(tf.cast(alpha_trained[i,l1],tf.float32),trainable=True))
            
        beta = tf.Variable(tf.cast(beta_trained[i],tf.float32),trainable=True)
    
    if pilot_known == True:
        Xt_real = tf.constant(Theta_real,dtype=tf.float32)
        Xt_imag = tf.constant(Theta_imag,dtype=tf.float32)
        Xt = tf.concat([Xt_real, Xt_imag],0)
    else:
        if i == snr_len:
            """
            if resolution_bit == 1:
                Xt = tf.get_variable('Xt', [2*Tt, K], initializer=tf.contrib.layers.xavier_initializer(), \
                                     constraint=keras.constraints.MinMaxNorm(min_value=Tt**0.5,max_value=Tt**0.5,axis=0),trainable=True)
            else:
                Xt = tf.get_variable('Xt', [2*Tt, K], initializer=tf.contrib.layers.xavier_initializer(), \
                                     constraint=keras.constraints.MaxNorm(max_value=Tt**0.5,axis=0),trainable=True)
            Xt_real = Xt[0:Tt,:]
            Xt_imag = Xt[Tt:2*Tt,:]
            """
            if resolution_bit == 1:
                Xt = tf.Variable(Xt0, trainable=True, constraint=keras.constraints.MinMaxNorm(min_value=Tt**0.5,max_value=Tt**0.5,axis=0), dtype=tf.float32)
            else:
                Xt = tf.Variable(Xt0, trainable=True, constraint=keras.constraints.MaxNorm(max_value=Tt**0.5,axis=0), dtype=tf.float32)
            Xt_real = Xt[0:Tt,:]
            Xt_imag = Xt[Tt:2*Tt,:]
            
        else:
            Xt_real = tf.constant(Xt_trained[0:Tt,:], dtype=tf.float32)
            Xt_imag = tf.constant(Xt_trained[Tt:2*Tt,:], dtype=tf.float32)
            Xt = tf.concat([Xt_real, Xt_imag],0)

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
    
    if pilot_known == True:
        Qup = Quantizer_up(r, rx_pow)
        Qlow = Quantizer_low(r, rx_pow)
    else:
        Qup = tf_relu_soft_quantizer_up(r, Delta*tf.sqrt(rx_pow), resolution_bit)
        Qlow = tf_relu_soft_quantizer_low(r, Delta*tf.sqrt(rx_pow), resolution_bit)
    
    # initial point
    h0 = tf.zeros([B,2*NK],dtype=tf.float32)
    
    hl = []
    hl.append(h0)

    for l in range(L):
        HX = tf.matmul(hl[l],P,transpose_b=True)
        temp1 = ones_mat - tf.sigmoid(beta*(HX-Qup)) - \
                           tf.sigmoid(beta*(HX-Qlow))
        temp2 = tf.matmul(temp1,P)
        temp3 = hl[l] + alpha[l]*temp2
        hl.append(temp3)
    
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
        train_iter = 6000
    else:
        train_iter = 500
    sess.run(init_op)
    for j in range(train_iter): #num of train iter
        h_batch, z_batch, rx_pow_batch = generate_train_data()
        train_step.run(feed_dict={h: h_batch, z: z_batch, rx_pow: rx_pow_batch})
        if np.mod(j,10)==0:
            print('train_iter = '+str(j))
            print('loss = ' + \
                  str(10*np.log10(2*loss.eval(feed_dict={h: h_batch, z: z_batch, rx_pow: rx_pow_batch}))))
            #print(np.linalg.norm(Xt_real.eval() + 1j*Xt_imag.eval(),axis=0))
            print('-----')
    if i == snr_len:
        Xt_trained = Xt.eval()
        #filename = './DNN_trained_results/Xt_trained_'+str(K)+'K_'+str(L)+'L_'+str(Tt)+'Tt_'+str(resolution_bit)+'bits.txt'
        #np.savetxt(filename, Xt_trained, fmt='%.15f', delimiter=', ') # save runtime to a text file"""

    for l2 in range(L):
        alpha_trained[i-1,l2] = alpha[l2].eval()
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






    














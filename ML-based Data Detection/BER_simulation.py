import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.special import erf
import random
from Quantizers import *
import time

random.seed(100)

mod_scheme = 'QPSK' # modulation schemes: 'BPSK','QPSK','16QAM','64QAM'
if mod_scheme=='BPSK':
    Alphabet = np.array([-1.0, 1.0])
elif mod_scheme=='QPSK':
    Alphabet = np.array([-1.0-1.0j, -1.0+1.0j, 1.0-1.0j, 1.0+1.0j])
    Alphabet_real = np.array([-1.0, 1.0])/np.sqrt(2.0)
elif mod_scheme=='8PSK':
    Alphabet = np.array([np.exp(0.0), np.exp(1.0j*np.pi/4.0), np.exp(3.0j*np.pi/4.0),\
                         np.exp(1.0j*np.pi/2.0), np.exp(1.0j*7.0*np.pi/4.0), \
                         np.exp(1.0j*3.0*np.pi/2.0), np.exp(1.0j*np.pi), \
                         np.exp(1.0j*5.0*np.pi/4.0)])
elif mod_scheme=='16QAM':
    Alphabet = np.array([-3.0-3.0j, -3.0-1.0j,  -3.0+3.0j,  -3.0+1.0j, \
                         -1.0-3.0j, -1.0-1.0j,  -1.0+3.0j,  -1.0+1.0j, \
                         +3.0-3.0j, +3.0-1.0j,  +3.0+3.0j,  +3.0+1.0j, \
                         +1.0-3.0j, +1.0-1.0j,  +1.0+3.0j,  +1.0+1.0j ])
    Alphabet_real = np.array([-3.0, -1.0, 1.0, 3.0,])/np.sqrt(10.0)
    B = np.array([-2.0, 0.0, 2.0])/np.sqrt(10.0)

Alphabet = Alphabet/np.sqrt(np.mean(np.abs(Alphabet)**2)) # normalize symbol energy
card = Alphabet.shape[0] # constellation size
card_real = np.log2(card)
bps = np.log2(card)      # number of bits per symbol
power = np.ones((card,1))*(2**np.arange(np.log2(card)))
# bit sequence of the constellation
bits = np.floor((np.array(range(card)).reshape(card,1)%(2*power))/power).T

perfect_CSI = False
known_P = False

Kb = 163 # K and b values (last digit is b, the rest is K)

if mod_scheme=='QPSK':
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

Tt = 5*K

method = 'DNN'

if method == 'DNN':
    if perfect_CSI == True:
        filename_d1 = './DetNet_trained_results/perfect_CSI/alpha_trained_'+mod_scheme+'_'+str(K)+'K_'+str(N)+'N_'+str(L)+'L_'+str(resolution_bit)+'bits.txt'
        alpha_trained = np.genfromtxt(filename_d1, delimiter=', ') 
        
        filename_d2 = './DetNet_trained_results/perfect_CSI/beta_trained_'+mod_scheme+'_'+str(K)+'K_'+str(N)+'N_'+str(L)+'L_'+str(resolution_bit)+'bits.txt'
        beta_trained = np.genfromtxt(filename_d2, delimiter=', ') 
    
        filename_d2 = './DetNet_trained_results/perfect_CSI/t_trained_'+mod_scheme+'_'+str(K)+'K_'+str(N)+'N_'+str(L)+'L_'+str(resolution_bit)+'bits.txt'
        t_trained = np.genfromtxt(filename_d2, delimiter=', ')
    else:
        filename_d1 = './DetNet_trained_results/estimated_CSI/alpha_trained_'+mod_scheme+'_'+str(K)+'K_'+str(N)+'N_'+str(L)+'L_'+str(resolution_bit)+'bits.txt'
        alpha_trained = np.genfromtxt(filename_d1, delimiter=', ') 
        
        filename_d2 = './DetNet_trained_results/estimated_CSI/beta_trained_'+mod_scheme+'_'+str(K)+'K_'+str(N)+'N_'+str(L)+'L_'+str(resolution_bit)+'bits.txt'
        beta_trained = np.genfromtxt(filename_d2, delimiter=', ') 
        
        filename_d2 = './DetNet_trained_results/estimated_CSI/t_trained_'+mod_scheme+'_'+str(K)+'K_'+str(N)+'N_'+str(L)+'L_'+str(resolution_bit)+'bits.txt'
        t_trained = np.genfromtxt(filename_d2, delimiter=', ')

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


idx_range = np.asarray(range(1,num_levels),float)[None]- num_levels/2

Delta_list = np.array([1.5958, 0.9957, 0.5860, 0.3352])

Delta = Delta_list[resolution_bit-1] # step size of resolution_bit

rho_q_list = np.array([0.1902, 0.1188, 0.0374, 0.0115])
rho_q = rho_q_list[resolution_bit-1]

Td = 500  # length of the block-fading interval
batch_size = 100
num_batch = np.int(Td/batch_size)

min_snr  = -10.0  # minimum simulated SNR
max_snr  = 30.0  # maximum simulated SNR
snr_step = 5.0   # SNR step
snr_dB = np.arange(min_snr, max_snr+snr_step, snr_step) # snr in dB
snr = 10**(snr_dB/10.0)  # snr in linear scale
c = 1.702

MSE = np.zeros(snr.shape,float)

# array to store BER
BER_BWZF  = np.zeros(snr.shape,float)
BER_BMMSE = np.zeros(snr.shape,float)
BER_DNN   = np.zeros(snr.shape,float)

runtime_BWZF  = np.zeros(snr.shape,float)
runtime_BMMSE = np.zeros(snr.shape,float)
runtime_DNN   = np.zeros(snr.shape,float)

## --------------- Nearest-neighbor symbol-by-symbol detection ----------------
def nearest_neighbor_sbs_detection(x_tilde_cplx):
    tx_idx_hat = np.argmin(np.abs(np.expand_dims(x_tilde_cplx,1) \
                           - np.expand_dims(Alphabet,0)),1)
    return tx_idx_hat

def sbs_detection(x_tilde):
    x_hat = np.zeros((2*K),int)
    if mod_scheme=='QPSK':
        for k in range(2*K):
            if x_tilde[k] < 0.0:
                x_hat[k] = Alphabet_real[0]
            else:
                x_hat[k] = Alphabet_real[1]
    if mod_scheme=='16QAM':
        for k in range(2*K):
            if x_tilde[k]<=B[1]:
                if x_tilde[k] <= B[0]:
                    i_hat = 0
                else:
                    i_hat = 1
            else:
                if x_tilde[k] <= B[2]:
                    i_hat = 2
                else:
                    i_hat = 3
            x_hat[k] = Alphabet_real[i_hat]
    return x_hat

## ------------------------- DNN data detection function --------------------------
def sigmoid(x):
    return np.reciprocal(1.0+np.exp(-x))

def DNN_receiver():
    start_time = time.time()
    X_tilde_real = np.empty((2*K,Td))

    for i in range(num_batch):
        Qup_batch = Qup[:,i*batch_size:(i+1)*batch_size]
        Qlow_batch = Qlow[:,i*batch_size:(i+1)*batch_size]
        X_batch = np.zeros((2*K,batch_size))
        for l in range(L):
            HX = np.matmul(Hhat,X_batch)
            up_branch = sigmoid(beta_trained[jj]*(HX-Qup_batch))
            low_branch = sigmoid(beta_trained[jj]*(HX-Qlow_batch))
            direction = np.matmul(Hhat.T,1.0-up_branch-low_branch)
            if mod_scheme == 'QPSK':
                X_batch = np_relu_projector(X_batch + alpha_trained[jj,l]*direction, \
                                       np.sqrt(2.0), 1, t_trained[jj,l])
            elif mod_scheme == '16QAM':
                X_batch = np_relu_projector(X_batch + alpha_trained[jj,l]*direction, \
                                       2.0/np.sqrt(10.0), 2, t_trained[jj,l])

        #X_tilde_real[:,i*batch_size:(i+1)*batch_size] = np.sqrt(K)*\
        #    np.reciprocal(np.linalg.norm(X_batch,2,0,keepdims=True))*X_batch # normalization
        X_tilde_real[:,i*batch_size:(i+1)*batch_size]= X_batch
    
    tx_idx_hat = np.zeros((K,Td),int) # use this function for BER
    for t in range(Td):
        x_tilde_cplx = X_tilde_real[0:K,t] + 1j*X_tilde_real[K:2*K,t] # use this function for BER
        tx_idx_hat[:,t] = nearest_neighbor_sbs_detection(x_tilde_cplx) # use this function for BER
    stop_time = time.time()
    runtime = stop_time - start_time
    return tx_idx_hat, runtime


## ----------------------- BWZF data detection function -----------------------
def phi(x):
    return (1/np.sqrt(2*np.pi))*np.exp(-0.5*x**2)

def PHI(x):
    return 0.5*(1-erf(-x/np.sqrt(2)))
    
def BWZF_receiver():
    start = time.time()
    sigma2 = 0.5*(np.linalg.norm(Hhat,axis=1,keepdims=True)**2+N0)
    sigma = np.sqrt(sigma2)
    sigma_inv = np.reciprocal(sigma)
    sigma2_inv = np.reciprocal(sigma2)
    temp_val = 0.5*(K+N0)
    val_range = -temp_val*(Delta*idx_range)**2
    Gb = (np.sqrt(temp_val)*Delta/np.sqrt(np.pi))*sigma_inv*\
        np.sum(np.exp(sigma2_inv*val_range),1,keepdims=True)
    B_mat = Gb*Hhat
    alpha = Qlow*sigma_inv
    beta  = Qup*sigma_inv
    temp_val1 = sigma2*((alpha*phi(alpha)-beta*phi(beta))*np.reciprocal(PHI(beta)-PHI(alpha)))
    temp_val2 = sigma*((phi(alpha)-phi(beta))*np.reciprocal(PHI(beta)-PHI(alpha)))
    temp_val22 = temp_val2**2
    first_term = sigma2 + temp_val1 - temp_val22
    sigma2_AWGN = 0.5*N0*Gb**2
    temp_val3 = first_term + temp_val22 + sigma2_AWGN
    
    rx_mat = np.empty((Td,2*K,2*N))
    for t in range(Td):
        W_mat = np.reciprocal(temp_val3[:,t] - 2*Y[:,t]*temp_val2[:,t] + Y[:,t]**2)
        BTW = B_mat.T*np.transpose(W_mat[:,None])
        rx_mat[t,:,:] = np.matmul(np.linalg.inv(np.matmul(BTW,B_mat)),BTW)
    
    X_tilde = np.empty((2*K,Td))
    for i in range(num_batch):
        i_range = range(i*batch_size,(i+1)*batch_size)
        X_tilde[:,i_range] = \
            np.squeeze(np.matmul(rx_mat[i_range,:,:],np.transpose(Y[:,i_range,None],(1,0,2))),2).T

    #X_tilde = np.sqrt(K)*np.reciprocal(np.linalg.norm(X_tilde,2,0,keepdims=True))*X_tilde # normalization
    
    tx_idx_hat = np.zeros((K,Td),int)
    for t in range(Td):
        x_tilde_cplx = X_tilde[0:K,t] + 1j*X_tilde[K:2*K,t]
        tx_idx_hat[:,t] = nearest_neighbor_sbs_detection(x_tilde_cplx)

    stop = time.time()
    return tx_idx_hat, stop-start

## ----------------------- BMMSE data detection function -----------------------
def BMMSE_receiver():
    start = time.time()
    sigma2 = 0.5*(np.linalg.norm(Hhat,axis=1,keepdims=True)**2+N0)
    sigma = np.sqrt(sigma2)
    sigma_inv = np.reciprocal(sigma)
    sigma2_inv = np.reciprocal(sigma2)
    temp_val = 0.5*(K+N0)
    val_range = -temp_val*(Delta*idx_range)**2
    Gb = (np.sqrt(temp_val)*Delta/np.sqrt(np.pi))*sigma_inv*\
        np.sum(np.exp(sigma2_inv*val_range),1,keepdims=True)

    GbH = Gb*Hhat
    GbHT = GbH.T
    sigma2_AWGN = 0.5*N0*Gb**2

    Cd = rho_q*sigma2
    rx_mat = np.matmul(0.5*GbHT,np.linalg.inv(0.5*np.matmul(GbH,GbHT)+np.diag(np.squeeze(sigma2_AWGN+Cd,1))))
    
    X_tilde = np.empty((2*K,Td))
    for i in range(num_batch):
        i_range = range(i*batch_size,(i+1)*batch_size)
        X_tilde[:,i_range] = np.matmul(rx_mat,Y[:,i_range])

    #x_hat = np.zeros((2*K,Td),float)
    tx_idx_hat = np.zeros((K,Td),int)
    for t in range(Td):
        x_tilde_cplx = X_tilde[0:K,t] + 1j*X_tilde[K:2*K,t]
        tx_idx_hat[:,t] = nearest_neighbor_sbs_detection(x_tilde_cplx)
    stop = time.time()
    return tx_idx_hat, stop - start

## ------------------------ Channel estimator -------------------------
def FBM_ChaNet(Qupt, Qlowt, alphat, betat):
    Hhat = np.zeros((2*K,N),float)
    for l in range(Lt):
        XtH = np.matmul(Xt,Hhat)
        Hhat = Hhat + alphat[l]*np.matmul(Xt.T,(OneMat - sigmoid(betat*(XtH - Qupt)) - \
                         sigmoid(betat*(XtH - Qlowt))))
    return np.transpose(Hhat[0:K,:]), np.transpose(Hhat[K:2*K,:])


## ========================== SIMULATION START HERE ===========================
run_time = 0.0
for jj in range(snr.shape[0]):
    print(snr_dB[jj])
    N0 = 1/snr[jj]
    csqrt2rho = c*np.sqrt(2*snr[jj])
    if snr_dB[jj]<=0.0:
        numChan = 1e2
    elif snr_dB[jj] == 5.0:
        numChan = 1e2
    elif snr_dB[jj] == 10.0:
        numChan = 1e2
    elif snr_dB[jj]>10.0 and snr_dB[jj]<=20.0:
        numChan = 1e2
    elif snr_dB[jj]>=25.0:
        numChan = 1e2
    
    alphat = alphat_trained[jj,:]
    betat = betat_trained[jj]
        
    A_size = 0
    for ii in range(np.int(numChan)):
        if np.mod(ii,100)==0:
            print(ii)
        H_real = np.random.normal(0,np.sqrt(0.5),(N,K)) # real part of channel
        H_imag = np.random.normal(0,np.sqrt(0.5),(N,K)) # imaginary part of channel
        H_cplx = H_real + 1j*H_imag # complex channel
        
        # channel estimation phase
        Ht = np.concatenate((H_real.T, H_imag.T),0)
        Zt = np.random.normal(0,np.sqrt(0.5*N0),(2*Tt,N)) # Noise in real domain
        Rt = np.matmul(Xt, Ht) + Zt # received data signal in complex domain
        _, Qupt, Qlowt = np_quantizer(Rt, 0.5*(K+N0),Delta,num_levels)
        
        Hhat_real, Hhat_imag = FBM_ChaNet(Qupt, Qlowt, alphat, betat)
    
        MSE[jj] = MSE[jj] + (np.linalg.norm(H_real - Hhat_real)**2 + \
                             np.linalg.norm(H_imag - Hhat_imag)**2)/(K*N)
        
        if perfect_CSI == True:
            Hhat = np.concatenate((np.concatenate((H_real,-H_imag),1), \
                                   np.concatenate((H_imag, H_real),1)),0)
        else:
            Hhat = np.concatenate((np.concatenate((Hhat_real,-Hhat_imag),1), \
                                   np.concatenate((Hhat_imag, Hhat_real),1)),0)
        
        # data transmission phase
        tx_idx = np.random.randint(0,card,(K,Td)) # indices of data symbols
        tx_bits = bits[:,tx_idx] # transmitted bits
        X_cplx = Alphabet[tx_idx] # transmitted signal in complex domain
        Z_cplx = np.random.normal(0,np.sqrt(0.5*N0),(N,Td)) + 1j*np.random.normal(0,np.sqrt(0.5*N0),(N,Td)) # Noise in complex domain
        R_cplx = np.matmul(H_cplx,X_cplx) + Z_cplx # received data signal in complex domain
        Y_real, Qup_real, Qlow_real = np_quantizer(np.real(R_cplx),0.5*(K+N0),Delta,num_levels)
        Y_imag, Qup_imag, Qlow_imag = np_quantizer(np.imag(R_cplx),0.5*(K+N0),Delta,num_levels)
        
        if method == 'BWZF':
            Qup = np.concatenate((Qup_real,Qup_imag),0)
            Qlow = np.concatenate((Qlow_real,Qlow_imag),0)
            Y = np.concatenate((Y_real,Y_imag),0)
            tx_idx_hat, curr_runtime = BWZF_receiver()
            runtime_BWZF[jj] = runtime_BWZF[jj] + curr_runtime
            tx_bits_hat = bits[:,tx_idx_hat] # detected bits
            BER_BWZF[jj] = BER_BWZF[jj] + np.sum((tx_bits_hat!=tx_bits)+0) # error count
        
        if method == 'BMMSE':
            Y = np.concatenate((Y_real,Y_imag),0)
            R = np.concatenate((np.real(R_cplx),np.imag(R_cplx)),0)
            tx_idx_hat, curr_runtime = BMMSE_receiver()
            runtime_BMMSE[jj] = runtime_BMMSE[jj] + curr_runtime
            tx_bits_hat = bits[:,tx_idx_hat] # detected bits
            BER_BMMSE[jj] = BER_BMMSE[jj] + np.sum((tx_bits_hat!=tx_bits)+0) # error count

        
        if method == 'DNN':
            Qup = np.concatenate((Qup_real,Qup_imag),0)
            Qlow = np.concatenate((Qlow_real,Qlow_imag),0)
            tx_idx_hat, curr_runtime = DNN_receiver()
            runtime_DNN[jj] = runtime_DNN[jj] + curr_runtime
            tx_bits_hat = bits[:,tx_idx_hat] # detected bits
            BER_DNN[jj] = BER_DNN[jj] + np.sum((tx_bits_hat!=tx_bits)+0) # error count
    
    MSE[jj] = 10.0*np.log10(MSE[jj]/numChan)
    
    BER_BWZF[jj]  = BER_BWZF[jj]/(numChan*K*Td*np.log2(card)) # BER
    BER_BMMSE[jj] = BER_BMMSE[jj]/(numChan*K*Td*np.log2(card)) # BER
    BER_DNN[jj]   = BER_DNN[jj]/(numChan*K*Td*np.log2(card)) # BER
    
    runtime_BWZF[jj]  = runtime_BWZF[jj]/numChan
    runtime_BMMSE[jj] = runtime_BMMSE[jj]/numChan
    runtime_DNN[jj]   = runtime_DNN[jj]/numChan

# Plot the BER
plt.figure()
plt.semilogy(snr_dB,BER_BWZF,"-o")
plt.semilogy(snr_dB,BER_BMMSE,"-s")
plt.semilogy(snr_dB,BER_DNN,"-^")
plt.xlabel('SNR - dB')
plt.ylabel('BER - dB')
plt.grid(b=1,which='major',axis='both')
plt.axis([-10, 30, 1e-6, 1])


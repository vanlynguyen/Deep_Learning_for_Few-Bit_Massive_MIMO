import numpy as np
import matplotlib.pyplot as plt
from Quantizers import *
np.random.seed(4)
resolution_bit = 2 # number of bits in the uniform quantizer
num_levels = 2**resolution_bit
# step sizes that minimizes MSE for unit variance standard Gaussian random variables
#                      1-bit , 2-bit , 3-bit , 4-bit 
Delta_list = np.array([1.5958, 0.9957, 0.5860, 0.3352])
Delta = Delta_list[resolution_bit-1] # step size of resolution_bit

K = 4  # number of users, each user is equipped with only one antenna
N = 32 # number of antennas at the base station
L = 8
Tt = 20
NK = N*K
NTt = N*Tt
KTt = K*Tt

pilot_known = True

if pilot_known == True:
    filename1 = './DNN_trained_results/known_P/Xt_trained_'+str(K)+'K_'+str(L)+'L_'+str(Tt)+'Tt_'+str(resolution_bit)+'bits.txt'
    Xt_trained = np.genfromtxt(filename1, delimiter=', ')
    
    filename2 = './DNN_trained_results/known_P/alpha_trained_'+str(K)+'K_'+str(L)+'L_'+str(Tt)+'Tt_'+str(resolution_bit)+'bits.txt'
    alpha_trained = np.genfromtxt(filename2, delimiter=', ')
    
    filename3 = './DNN_trained_results/known_P/beta_trained_'+str(K)+'K_'+str(L)+'L_'+str(Tt)+'Tt_'+str(resolution_bit)+'bits.txt'
    beta_trained = np.genfromtxt(filename3, delimiter=', ')
else:
    filename1 = './DNN_trained_results/trainable_P/Xt_trained_'+str(K)+'K_'+str(L)+'L_'+str(Tt)+'Tt_'+str(resolution_bit)+'bits.txt'
    Xt_trained = np.genfromtxt(filename1, delimiter=', ')
    
    filename2 = './DNN_trained_results/trainable_P/alpha_trained_'+str(K)+'K_'+str(L)+'L_'+str(Tt)+'Tt_'+str(resolution_bit)+'bits.txt'
    alpha_trained = np.genfromtxt(filename2, delimiter=', ')
    
    filename3 = './DNN_trained_results/trainable_P/beta_trained_'+str(K)+'K_'+str(L)+'L_'+str(Tt)+'Tt_'+str(resolution_bit)+'bits.txt'
    beta_trained = np.genfromtxt(filename3, delimiter=', ')
    
Xt_real = Xt_trained[0:Tt,:]
Xt_imag = Xt_trained[Tt:2*Tt,:]

Xt = np.concatenate((np.concatenate((Xt_real, -Xt_imag),1), \
                     np.concatenate((Xt_imag,  Xt_real),1)),0)
OneMat = np.ones((2*Tt,N),float)

min_snr  = -10.0  # minimum simulated SNR
max_snr  = 30.0  # maximum simulated SNR
snr_step = 5.0   # SNR step
snr_dB = np.arange(min_snr, max_snr+snr_step, snr_step) # snr in dB
snr = 10**(snr_dB/10.0)  # snr in linear scale
c = 1.702
sqrt2pi = np.sqrt(2*np.pi)

MSE   = np.zeros(snr.shape,float)

def sigmoid(X):
    return np.reciprocal(1.0 + np.exp(-X))

def FBM_ChaNet():
    Hhat = np.zeros((2*K,N),float)
    for l in range(L):
        XtH = np.matmul(Xt,Hhat)
        Hhat = Hhat + alpha[l]*np.matmul(Xt.T,(OneMat - sigmoid(beta*(XtH - Qup)) - \
                         sigmoid(beta*(XtH - Qlow))))
    return Hhat

def Constant_Stepsize_Grad_Descent():
    Hhat = np.zeros((2*K,N),float)
    for l in range(L):
        XtH = np.matmul(Xt,Hhat)
        Hhat = Hhat + (0.04/np.sqrt(rho))*csqrt2rho*np.matmul(Xt.T,(OneMat - sigmoid(csqrt2rho*(XtH - Qup)) - \
                         sigmoid(csqrt2rho*(XtH - Qlow))))
    return Hhat

def f_eval(s_up, s_low):
    f_val = np.sum(np.log(sigmoid(csqrt2rho*s_up) - sigmoid(csqrt2rho*s_low)))
    
    return f_val


## ========================== SIMULATION START HERE ===========================
for jj in range(snr.shape[0]):
    print(snr_dB[jj])
    N0 = 1/snr[jj]
    rho = snr[jj]
    csqrt2rho = c*np.sqrt(2*snr[jj])
    if snr_dB[jj]<=0.0:
        numChan = 1e3
    elif snr_dB[jj] == 5.0:
        numChan = 1e3
    elif snr_dB[jj] == 10.0:
        numChan = 1e3
    elif snr_dB[jj]>10.0 and snr_dB[jj]<=20.0:
        numChan = 2e3
    elif snr_dB[jj]>20.0:
        numChan = 4e3
        
    alpha = alpha_trained[jj,:]
    beta = beta_trained[jj]
    
    for ii in range(np.int(numChan)):
        if np.mod(ii,1000)==0:
            print(ii)

        H_real = np.random.normal(0,np.sqrt(0.5),(K,N)) # real part of channel
        H_imag = np.random.normal(0,np.sqrt(0.5),(K,N)) # imaginary part of channel
        H = np.concatenate((H_real, H_imag),0)

        # transmission phase
        Z = np.random.normal(0,np.sqrt(0.5*N0),(2*Tt,N)) # Noise in real domain
        R = np.matmul(Xt, H) + Z # received data signal in complex domain
        _, Qup, Qlow = np_quantizer(R, 0.5*(K+N0), Delta, num_levels)
        
        H_hat = FBM_ChaNet()
    
        MSE[jj] = MSE[jj] + (np.linalg.norm(H - H_hat)**2)/(K*N)

    MSE[jj]  = 10*np.log10(MSE[jj]/numChan) 


print(MSE)

# Plot the BER
plt.figure()
plt.plot(snr_dB,MSE,"-s")
plt.xlabel('SNR - dB')
plt.ylabel('MSE - dB')
plt.grid(b=1,which='major',axis='both')

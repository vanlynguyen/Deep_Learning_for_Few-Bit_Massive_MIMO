import numpy as np
import matplotlib.pyplot as plt
from Quantizers import *
np.random.seed(4)
resolution_bit = 1 # number of bits in the uniform quantizer
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

filename1 = 'Covariance_Data\Lower_mat_'+str(N)+'N.txt'
filename2 = 'Covariance_Data\Cov_mat_'+str(N)+'N.txt'

Lower_mat = np.genfromtxt(filename1, delimiter=', ')
Lower_mat = Lower_mat[:,0:N] + 1j*Lower_mat[:,N:2*N]

Cov_mat = np.genfromtxt(filename2, delimiter=', ')
Cov_mat = Cov_mat[:,0:N] + 1j*Cov_mat[:,N:2*N]

M = int(np.shape(Cov_mat)[0]/N)

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

print(np.linalg.norm(Xt_real + 1j*Xt_imag,axis=0))

Xt = np.concatenate((np.concatenate((Xt_real, -Xt_imag),1), \
                     np.concatenate((Xt_imag,  Xt_real),1)),0)
OneMat = np.ones((2*Tt,N),float)

min_snr  = -5.0  # minimum simulated SNR
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
        Hhat = Hhat + (0.05/np.sqrt(rho))*csqrt2rho*np.matmul(Xt.T,(OneMat - sigmoid(csqrt2rho*(XtH - Qup)) - \
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
        numChan = 2e3
    elif snr_dB[jj] == 5.0:
        numChan = 3e3
    elif snr_dB[jj] == 10.0:
        numChan = 4e3
    elif snr_dB[jj]>10.0 and snr_dB[jj]<=20.0:
        numChan = 5e3
    elif snr_dB[jj]>20.0:
        numChan = 6e3
        
    alpha = alpha_trained[jj,:]
    beta = beta_trained[jj]
    
    for ii in range(np.int(numChan)):
        if np.mod(ii,100)==0:
            print(ii)
        
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
        
        H_cplx = np.zeros((N,K),dtype=complex)
        m_list = np.random.randint(0,M,K)
        for k in range(K):
            m = m_list[k]
            Lmat_k = Lower_mat[m*N:(m+1)*N,:]
            H_cplx[:, k] = np.matmul(Lmat_k,H_NLoS[:,k]) + H_LoS[:,k]
        
        H = np.concatenate((np.real(H_cplx), np.imag(H_cplx)),1).T


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


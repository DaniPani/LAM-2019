import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

data = h5py.File('output/1d_r.h5')

I, Q, U, V = np.array(fits.open('data/5876_m1_20100316.fits')[0].data)

I_I_max = np.divide(I, np.transpose([np.max(I, axis=1)]))
Q_I_max = np.divide(Q, np.transpose([np.max(I, axis=1)]))
U_I_max = np.divide(U, np.transpose([np.max(I, axis=1)]))
V_I_max = np.divide(V, np.transpose([np.max(I, axis=1)]))

I_mean = np.mean(np.transpose(I_I_max[6:54]), axis=1)[300:550]
Q_mean = np.mean(np.transpose(Q_I_max[6:54]), axis=1)[300:550]
U_mean = np.mean(np.transpose(U_I_max[6:54]), axis=1)[300:550]
V_mean = np.mean(np.transpose(V_I_max[6:54]), axis=1)[300:550]

stokes = np.array([I_mean, Q_mean, U_mean, V_mean])

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
ax = ax.flatten()

label = ['i', 'q', 'u', 'v']

for i in range(4):

    ax[i].plot(data['spec1']['wavelength'][:], stokes[i,:])
    for j in range(2):
        np.savetxt('txt/{}_{}.dat'.format(label[i], j + 1), list(zip(data['spec1']['wavelength'][:], data['spec1']['stokes'][0,0,j,i,:])) )
        ax[i].plot(data['spec1']['wavelength'][:], data['spec1']['stokes'][0,0,j,i,:],  label='Cycle {0}'.format(j))

for i in range(4):
    ax[i].set_xlabel('Wavelength [$\AA$]')
    #ax[i].set_ylabel('{0}/Ic'.format(label[i]))

plt.legend()

plt.show()
import hazel
from astropy.io import fits
import os
import numpy as np

# CMD wsl /usr/bin/python3 /mnt/c/Users/paner/Documents/LAM/daniel/inversion_1d.py

# FITS TO 1D FILES
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

tmp = hazel.tools.File_observation(mode='single')
tmp.set_size(n_lambda=250, n_pixel=1)
tmp.obs['stokes'] = np.array([np.transpose(stokes)])
tmp.obs['sigma'] =  np.array([np.std(stokes[0,0:15])*np.ones((250,1)),  np.std(stokes[1,0:15])*np.ones((250,1)),  np.std(stokes[2,0:15])*np.ones((250,1)), np.std(stokes[3,0:15])*np.ones((250,1))]).transpose()
tmp.obs['los'][:] = np.full((1, 3), [90,0,90])
tmp.obs['boundary'][:] = np.full((1, 250,4), [0,0,0,0])
tmp.save('data/tmp_r')

# CHROMOSPHERE
tmp = hazel.tools.File_chromosphere(mode='single')
tmp.set_default(n_pixel=1, default='offlimb')
tmp.save('configurations/model_chromosphere_r')

# REMOVE USELESS FILES
os.remove('data/tmp_r.wavelength')
os.remove('data/tmp_r.weights')

# WITHOUT RANDOMIZATION
mod = hazel.Model('configurations/1d_r.ini', working_mode='inversion', verbose=3)
mod.read_observation()
mod.open_output()
mod.invert()
mod.write_output()
mod.close_output()
import hazel
from astropy.io import fits
import os
import numpy as np

n_lambda = 250
n_pixel = 4

I, Q, U, V = np.array(fits.open('data/5876_m1_20100316.fits')[0].data)

# FITS TO H5 FILES
I_I_max = np.divide(I, np.transpose([np.max(I, axis=1)]))
Q_I_max = np.divide(Q, np.transpose([np.max(I, axis=1)]))
U_I_max = np.divide(U, np.transpose([np.max(I, axis=1)]))
V_I_max = np.divide(V, np.transpose([np.max(I, axis=1)]))

I_means = np.mean(np.split(I_I_max[6:54], n_pixel), axis=1)[:, 300:550]
Q_means = np.mean(np.split(Q_I_max[6:54], n_pixel), axis=1)[:, 300:550]
U_means = np.mean(np.split(U_I_max[6:54], n_pixel), axis=1)[:, 300:550]
V_means = np.mean(np.split(V_I_max[6:54], n_pixel), axis=1)[:, 300:550]

stokes = np.array(I_means, Q_means, U_means, V_means)

tmp = hazel.tools.File_observation(mode='multi')
tmp.set_size(n_lambda=n_lambda, n_pixel=n_pixel)
tmp.obs['stokes'][:] = np.reshape(stokes, (n_pixel,n_lambda,4))
tmp.obs['sigma'][:] = np.zeros((n_pixel,n_lambda,4), dtype=np.float64)

for i in range(n_pixel):
    noise = np.std(tmp.obs['stokes'][i,0:15], axis = 0)
    
    tmp.obs['sigma'][i,:,:] = np.full((n_lambda, 4), noise)

tmp.obs['los'][:] = np.full((n_pixel, 3), [90,0,90])
tmp.obs['boundary'][:] = np.full((n_pixel,n_lambda,4), [0,0,0,0])

tmp.save('data/tmp')

# CHROMOSPHERE
tmp = hazel.tools.File_chromosphere(mode='multi')
tmp.set_default(n_pixel=n_pixel, default='offlimb')
tmp.save('configurations/model_chromosphere')

# REMOVE USELESS FILES
os.remove('data/tmp.wavelength')
os.remove('data/tmp.weights')
os.remove('data/tmp.mask')

# WITHOUT RANDOMIZATION
iterator = hazel.Iterator(use_mpi=True)
mod = hazel.Model('configurations/3d.ini', working_mode='inversion', verbose=3, rank=iterator.get_rank())
iterator.use_model(model=mod)
iterator.run_all_pixels()
import hazel
from astropy.io import fits
import os
import numpy as np

# FITS TO H5 FILES
stokes = np.array(fits.open('data/5876_m1_20100316.fits')[0].data, dtype=np.float64)[:, 16:45, 300:550]

n_lambda = 250
n_pixel = 29

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
iterator = hazel.Iterator(use_mpi=False)
mod = hazel.Model('configurations/3d.ini', working_mode='inversion')
iterator.use_model(model=mod)
iterator.run_all_pixels()
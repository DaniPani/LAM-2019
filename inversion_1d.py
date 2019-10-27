import hazel
from astropy.io import fits
import os
import numpy as np

# FITS TO 1D FILES
stokes = np.array(fits.open('data/tmp.fits')[0].data)

tmp = hazel.tools.File_observation(mode='single')
tmp.set_size(n_lambda=175, n_pixel=1)
tmp.obs['stokes'] = np.array([np.transpose(stokes)])
tmp.obs['sigma'] =  np.array([np.std(stokes[0,0:15])*np.ones((175,1)),  np.std(stokes[1,0:15])*np.ones((175,1)),  np.std(stokes[2,0:15])*np.ones((175,1)), np.std(stokes[3,0:15])*np.ones((175,1))]).transpose()
tmp.obs['los'] = np.array([[90,0,90]])
tmp.obs['boundary'] = np.array([[[0,0,0,0]]])
tmp.save('data/tmp')

# CHROMOSPHERE
tmp = hazel.tools.File_chromosphere(mode='single')
tmp.set_default(n_pixel=1, default='offlimb')
tmp.save('configurations/model_chromosphere')

# REMOVE USELESS FILES
os.remove('data/tmp.wavelength')
os.remove('data/tmp.weights')

# WITHOUT RANDOMIZATION
mod = hazel.Model('configurations/1d.ini', working_mode='inversion', verbose=3)
mod.read_observation()
mod.open_output()
mod.invert()
mod.write_output()
mod.close_output()
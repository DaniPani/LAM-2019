import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

data = h5py.File('output/3d.h5')
stokes = np.array(fits.open('data/5876_m1_20100316.fits')[0].data, dtype=np.float64)[:, 16:45, 300:550]


for k in range(2):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    ax = ax.flatten()
    for i in range(4):
        ax[i].plot(data['spec1']['wavelength'][:], stokes[k, i, :])
        for j in range(2):
            ax[i].plot(data['spec1']['wavelength'][:], data['spec1']['stokes'][k,0,j,i,:])
import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

data = h5py.File('output/1d.h5')
stokes = np.array(fits.open('data/tmp.1d.fits')[0].data)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
ax = ax.flatten()
for i in range(4):

    ax[i].plot(data['spec1']['wavelength'][:], stokes[i,:])
    for j in range(2):
        ax[i].plot(data['spec1']['wavelength'][:], data['spec1']['stokes'][0,0,j,i,:],  label='Cycle {0}'.format(j))

for i in range(4):
    ax[i].set_xlabel('Wavelength [$\AA$]')
    #ax[i].set_ylabel('{0}/Ic'.format(label[i]))

plt.legend()

plt.show()
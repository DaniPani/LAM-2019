%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import hazel 
mod = hazel.Model(working_mode='synthesis', verbose=True)

dict = {'Name': 'spec1', 'Wavelength': [5874.699, 5876.807, 250], 'topology': 'ch1',
    'LOS': [90.0,0.0,90.0], 'Boundary condition': [0.0,0.0,0.0,0.0]}
mod.add_spectral(dict)

mod.add_chromosphere({'Name': 'ch1', 'Spectral region': 'spec1', 'Height': 29.5,
    'Line': '5876', 'Wavelength': [5874, 5878], 'Coordinates for magnetic field vector':'spherical'})

# Bx, By, Bz, τ, v, Δv, β, a
mod.atmospheres['ch1'].set_parameters([0.0,0.0,0.0,0.0,0.0,8.0,1.0,0.0],1.0)

mod.setup()

mod.synthesize()

for i in range(4):
    ax[i].plot(data['spec1']['wavelength'][:], data['spec1']['stokes'][0,0,0,i,:])

plt.legend()

plt.show()

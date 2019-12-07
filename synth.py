import matplotlib.pyplot as plt
import numpy as np
import hazel 
from astropy.io import fits
import h5py

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

mod = hazel.Model(working_mode='synthesis', verbose=True)


dict = {'Name': 'spec1', 'Wavelength': [5874.699, 5876.807, 250], 'topology': 'ch1',
    'LOS': [90.0,0.0,90.0], 'Boundary condition': [0.0,0.0,0.0,0.0]}
mod.add_spectral(dict)

mod.add_chromosphere({'Name': 'ch1', 'Spectral region': 'spec1', 'Height': 29.5,
    'Line': '5876', 'Wavelength': [5874, 5878]})

n_pixel = 73
relative_error = np.zeros((300, 4))
relative_error_3d = list()
multi = np.zeros((300, 4))
data = h5py.File('output/r_output_perfect.h5')

def variance(expt, x):
    return np.sqrt(np.mean(np.power(x/np.abs(expt) - 1, 2), axis=1))


""" for i, val in enumerate(np.linspace(0.0001, 0.49837, 300)):
    mod.atmospheres['ch1'].set_parameters([-0.7007, -1.246, 1.3643,val, 1.0893912236181507,6.988807129255939,1.0,0.001601483168434396],1.0)
    mod.setup()
    mod.synthesize()
    
    relative_error[i] = variance(data['spec1']['stokes'][0,0,1,:,:], mod.spectrum['spec1'].stokes) * 100
    #relative_error[i] = np.mean(np.absolute(np.divide(data['spec1']['stokes'][0,0,1,:,:] - mod.spectrum['spec1'].stokes, data['spec1']['stokes'][0,0,1,:,:])), axis=1) * 100
 """

std = np.array([np.std(stokes[0,0:15])*np.ones((250,1)),  np.std(stokes[1,0:15])*np.ones((250,1)),  np.std(stokes[2,0:15])*np.ones((250,1)), np.std(stokes[3,0:15])*np.ones((250,1))]).transpose()

def chi_2(sim, stokes, std):
    return 1/(4 * 250) * np.sum(np.divide(np.square(np.subtract(sim, stokes)), np.square(np.transpose(std[0]))))

for delta_thB, delta_chiB in np.transpose(np.meshgrid(np.linspace(-180, 180, n_pixel), np.linspace(-180, 180, n_pixel))).reshape(-1, 2):
    print(delta_thB, delta_chiB)
    r = 1.9760
    thB = 46
    chiB = 60

    x = r * np.sin(np.radians(delta_thB + thB)) * np.cos(np.radians(delta_chiB + chiB))
    y = r * np.sin(np.radians(delta_thB + thB)) * np.sin(np.radians(delta_chiB + chiB))
    z = r * np.cos(np.radians(delta_thB + thB))

    mod.atmospheres['ch1'].set_parameters([x, y, z, 0.49837, 1.0893912236181507,6.988807129255939,1.0,0.001601483168434396],1.0)
    mod.setup()
    mod.synthesize()
    relative_error_3d.append(chi_2(mod.spectrum['spec1'].stokes, stokes, std))

np.savetxt('txt/chi_2_3d.dat', np.concatenate((np.transpose(np.meshgrid(np.linspace(-180, 180, n_pixel), relative_error_3d)))))

label = ['i', 'q', 'u', 'v']

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
ax = ax.flatten()

""" for i in range(4):
    #ax[i].plot(relative_error[:, i])
    #ax[i].plot(data['spec1']['stokes'][0,0,1,i,:])
    np.savetxt('txt/err_3d_{}.dat'.format(label[i]), np.concatenate((np.transpose(np.meshgrid(np.linspace(-180, 180, n_pixel), np.linspace(-180, 180, n_pixel))).reshape(-1, 2), np.array(relative_error_3d)[:, i].reshape(n_pixel * n_pixel,1)), axis=1))
    #ax[i].plot(mod.spectrum['spec1'].stokes[i]) """
plt.savefig('img/plot')

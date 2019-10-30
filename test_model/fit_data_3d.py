import numpy as np

from astropy.io import fits
from astropy.constants import c, k_B, u
from astropy import units

from scipy.optimize import curve_fit

I = np.array(fits.open('data/5876_m1_20100316.fits')[0].data)[0, 16:45, 300:550]
I = np.divide(I, np.transpose([np.max(I, axis=1)]))
I_mean = np.mean(np.transpose(I), axis=1)

standard_deviation = np.std(I, axis=0)

helium_mass = 4.002602 * u
medium_lifetime = (1 / 7.0708e+07) * units.s
#delta_L = (1 / (2 * np.pi * medium_lifetime)).to(units.Hz)

def D(nu, nu_0, delta_D):
    return 2 * np.sqrt(np.log(2))/(np.sqrt(np.pi) * delta_D) * np.exp(-(2 * np.sqrt(np.log(2))/delta_D * (nu - nu_0))**2)

def L(nu, nu_0, delta_L):
    return 1/ np.pi * (delta_L/2)/((nu - nu_0)**2 + (delta_L/2)**2)

def V(nu, nu_0, delta_D, delta_L):
    delta = ( delta_D**5 +  2.69269 * delta_D**4 * delta_L + 2.42843 * delta_D**3 * delta_L**2 + 4.47163 * delta_D**2 * delta_L**3 + 0.07842 * delta_D * delta_L**4+ delta_L**5)**(1./5.)
    eta = 1.36603 * ( delta_L / delta ) - 0.47719 * ( delta_L / delta )**2 + 0.11116 * ( delta_L / delta )**3

    f = eta * L(nu, nu_0, delta_L) + ( 1 - eta ) * D( nu, nu_0, delta_D)
    return f / np.max(f)

optimal_values, covariance = curve_fit(V, np.arange(0, 250, 1), I_mean, maxfev=1000, bounds=(0, [150, 40, 40]))

print(optimal_values)
print(np.sqrt(np.diag(covariance))/optimal_values)

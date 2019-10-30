from astropy.io import fits
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


if os.path.exists('data/tmp.1d.fits'):
    os.remove('data/tmp.1d.fits')

I_prob, Q_prob, U_prob, V_prob = np.array(fits.open('data/5876_m1_red.fits')[0].data)
I_back, Q_back, U_back, V_back = np.array(fits.open('data/5876_m2_red.fits')[0].data)

etas = []

I = []
Q = []
U = []
V = []

def GM(values):
    return np.exp(np.mean(np.log(values)))

# ETAS
etas = np.apply_along_axis(GM,1, np.divide(I_prob[:, np.r_[:550, 850:]], I_back[:, np.r_[:550, 850:]]))

# STOKES I
I = np.subtract(I_prob, np.multiply(np.transpose([etas]), I_back))

# STOKES Q, U, V
Q = np.multiply(Q_prob, I)
U = np.multiply(U_prob, I)
V = np.multiply(V_prob, I)

# STOKES I_I_max, Q_I_max, U_I_max, V_I_max
I_I_max = np.divide(I, np.transpose([np.max(I, axis=1)]))
Q_I_max = np.divide(Q, np.transpose([np.max(I, axis=1)]))
U_I_max = np.divide(U, np.transpose([np.max(I, axis=1)]))
V_I_max = np.divide(V, np.transpose([np.max(I, axis=1)]))

# MEANS
I_mean = np.mean(np.transpose(I_I_max[93:103]), axis=1)[600:775]
Q_mean = np.mean(np.transpose(Q_I_max[93:103]), axis=1)[600:775]
U_mean = np.mean(np.transpose(U_I_max[93:103]), axis=1)[600:775]
V_mean = np.mean(np.transpose(V_I_max[93:103]), axis=1)[600:775]

# SAVE MEANS
hdu = fits.PrimaryHDU(np.array([I_mean, Q_mean, U_mean, V_mean]))
hdul = fits.HDUList([hdu])
hdul.writeto('data/tmp.1d.fits')

# BACKGROUND MEANS
I_back_mean = np.mean(np.transpose(np.divide(I_back, np.transpose([np.max(I_back, axis=1)]))), axis=1)

# PLOT 1
plt.figure(1)

plt.subplot(4,1,1)
plt.title('Stokes I, Q, U, V')
plt.xlabel("Pixels")
plt.ylabel("Rows")
plt.pcolormesh(I, cmap="gray", vmin=650)

plt.subplot(4,1,2)
plt.ylabel("Rows")
plt.pcolormesh(Q, cmap="gray", vmin=3)

plt.subplot(4,1,3)
plt.ylabel("Rows")
plt.pcolormesh(U, cmap="gray")

plt.subplot(4,1,4)
plt.ylabel("Rows")
plt.pcolormesh(V, cmap="gray")

# PLOT 2
index = 95 
fig, ax = plt.subplots()

plt.subplot(4,1,1)
f_I, = plt.plot(I_I_max[index])
plt.xlabel('Pixel')
plt.ylabel('I/Imax')

plt.subplot(4,1,2)
f_Q, = plt.plot(Q_I_max[index])
plt.xlabel('Pixel')
plt.ylabel('Q/Imax')

plt.subplot(4,1,3)
f_U, = plt.plot(U_I_max[index])
plt.xlabel('Pixel')
plt.ylabel('U/Imax')

plt.subplot(4,1,4)
f_V, = plt.plot(V_I_max[index])
plt.xlabel('Pixel')
plt.ylabel('V/Imax')


axis = plt.axes([0.25, .03, 0.50, 0.02])
slider = Slider(axis, 'Rows', 0, 139, valinit=95, valstep=1)
def update(val):
    index = slider.val
    f_I.set_ydata(I_I_max[int(index)])
    f_Q.set_ydata(Q_I_max[int(index)])
    f_U.set_ydata(U_I_max[int(index)])
    f_V.set_ydata(V_I_max[int(index)])
    fig.canvas.draw_idle()
slider.on_changed(update) 

# PLOT 3
fig_1, ax_1 = plt.subplots()

plt.subplot(4,1,1)
plt.plot(I_mean)
plt.ylabel('Mean I/Imax')

plt.subplot(4,1,2)
plt.plot(Q_mean)
plt.ylabel('Mean Q/Imax')

plt.subplot(4,1,3)
plt.plot(U_mean)
plt.ylabel('Mean U/Imax')

plt.subplot(4,1,4)
plt.plot(V_mean)
plt.ylabel('Mean V/Imax')

# PLOT 3
plt.figure(4)

plt.plot(I_back_mean)

# PLOT 4
fig_2, ax_2 = plt.subplots()

plt.subplot(5,1,1)
plt.plot(np.arange(5874.918, 5876.397, 8.452433e-3), I_mean)
plt.ylabel('I_mean')

plt.subplot(5,1,2)
plt.plot(np.arange(5874.918, 5876.397, 8.452433e-3), Q_mean)
plt.ylabel('Q_mean')

plt.subplot(5,1,3)
plt.plot(np.arange(5874.918, 5876.397, 8.452433e-3), U_mean)
plt.ylabel('U_mean')

plt.subplot(5,1,4)
plt.plot(np.arange(5874.918, 5876.397, 8.452433e-3), V_mean)
plt.ylabel('V_mean')

plt.subplot(5,1,5)
plt.plot(np.arange(5874.918, 5876.397, 8.452433e-3), I_back_mean[600:775])
plt.ylabel('I_background_mean')

plt.show()
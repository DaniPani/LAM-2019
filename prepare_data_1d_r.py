from astropy.io import fits
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

rowToInvert = 1

I, Q, U, V = np.array(fits.open('data/5876_m1_20100316.fits')[0].data)

I_I_max = np.divide(I, np.transpose([np.max(I, axis=1)]))
Q_I_max = np.divide(Q, np.transpose([np.max(I, axis=1)]))
U_I_max = np.divide(U, np.transpose([np.max(I, axis=1)]))
V_I_max = np.divide(V, np.transpose([np.max(I, axis=1)]))

I_means = np.mean(np.split(I_I_max[6:54],rowToInvert), axis=1)[:, 300:550]
Q_means = np.mean(np.split(Q_I_max[6:54],rowToInvert), axis=1)[:, 300:550]
U_means = np.mean(np.split(U_I_max[6:54],rowToInvert), axis=1)[:, 300:550]
V_means = np.mean(np.split(V_I_max[6:54],rowToInvert), axis=1)[:, 300:550]

I_mean = np.mean(np.transpose(I_I_max[6:54]), axis=1)
Q_mean = np.mean(np.transpose(Q_I_max[6:54]), axis=1)
U_mean = np.mean(np.transpose(U_I_max[6:54]), axis=1)
V_mean = np.mean(np.transpose(V_I_max[6:54]), axis=1)

# PLOT 1
plt.figure(1)

plt.subplot(4,1,1)
plt.title('Stokes I, Q, U, V')
plt.xlabel("Pixels")
plt.ylabel("Rows")
plt.pcolormesh(I, cmap="gray", vmin=0.00000005)

plt.subplot(4,1,2)
plt.ylabel("Rows")
plt.pcolormesh(Q, cmap="gray", vmin=0.0015)

plt.subplot(4,1,3)
plt.ylabel("Rows")
plt.pcolormesh(U, cmap="gray")

plt.subplot(4,1,4)
plt.ylabel("Rows")
plt.pcolormesh(V, cmap="gray")

# PLOT 2
index = 0
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
slider = Slider(axis, 'Rows', 0, 140, valinit=40, valstep=1)
def update(val):
    index = int(slider.val)
    f_I.set_ydata(I_I_max[index])
    f_Q.set_ydata(Q_I_max[index])
    f_U.set_ydata(U_I_max[index])
    f_V.set_ydata(V_I_max[index])
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

plt.show()
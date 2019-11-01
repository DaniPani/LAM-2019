from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

img = np.array(fits.open('img/img.fits')[0].data)

plt.pcolormesh(img, cmap="gray")

plt.show()
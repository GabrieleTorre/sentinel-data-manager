import numpy as np
import rasterio

import matplotlib.pyplot as plt

files = [
    '/Volumes/SSD1/Sentinel-2/test-italia/S2B_MSIL2A_20211114T101159_N0301_R022_T32TPQ_20211114T120913.SAFE/GRANULE/L2A_T32TPQ_A024501_20211114T101201/IMG_DATA/R10m/T32TPQ_20211114T101159_B02_10m.jp2',
    '/Volumes/SSD1/Sentinel-2/test-italia/S2B_MSIL2A_20211114T101159_N0301_R022_T32TPQ_20211114T120913.SAFE/GRANULE/L2A_T32TPQ_A024501_20211114T101201/IMG_DATA/R10m/T32TPQ_20211114T101159_B03_10m.jp2',
    '/Volumes/SSD1/Sentinel-2/test-italia/S2B_MSIL2A_20211114T101159_N0301_R022_T32TPQ_20211114T120913.SAFE/GRANULE/L2A_T32TPQ_A024501_20211114T101201/IMG_DATA/R10m/T32TPQ_20211114T101159_B04_10m.jp2',
        ]

masks = ['/Volumes/SSD1/Sentinel-2/test-italia/S2B_MSIL2A_20211114T101159_N0301_R022_T32TPQ_20211114T120913.SAFE/GRANULE/L2A_T32TPQ_A024501_20211114T101201/QI_DATA/MSK_CLDPRB_20m.jp2']

band = np.array([rasterio.open(f, driver="JP2OpenJPEG").read() for f in files])
mask = rasterio.open(masks[0], driver="JP2OpenJPEG").read()
band = band.squeeze().transpose(1,2,0)/10000

fig, ax = plt.subplots(1, figsize=(10, 10))
im = ax.imshow(band)

fig, ax = plt.subplots(1, figsize=(10, 10))
im = ax.imshow(mask.squeeze(), cmap='gray')
plt.show()

import pdb; pdb.set_trace()

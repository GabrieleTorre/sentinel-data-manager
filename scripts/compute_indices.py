import matplotlib.pyplot as plt
from datetime import datetime

from sentinel_2 import data_manager
from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np
from random import sample

dirs = glob('/Volumes/SSD1/Sentinel-2/test-italia-numpy/T32TPQ/*/')

out = []
for d in tqdm(sample(dirs, 125)):
    dm = data_manager(d)
    ind = dm.get_index_ts(dm.ndvi, veg_type=None).rename(d)
    out.append(ind)

out = pd.concat(out, axis=1)
out.plot(figsize=(20, 8), legend=None)
plt.show()

selection_df = out.loc[datetime(2021, 10, 15)].sort_values(ascending=False)

fig, ax = plt.subplots(6, 10, figsize=(10, 3))

NIR_ind = dm.bands['B08']
R_ind = dm.bands['B04']

for i, file in enumerate(selection_df.iloc[0:10].index):
    img = np.load(file+'20211114.npy').astype(np.float64)
    ax[0, i].imshow(np.swapaxes(img[:3], 0, 2)/10000)

    ndvi = np.divide((img[NIR_ind] - img[R_ind]), (img[NIR_ind] + img[R_ind]))
    ax[1, i].imshow(ndvi.T)

    mask = np.load(file+'/cloudProb/20211114.npy').astype(np.float64)[0]
    ax[2, i].imshow(mask.T, cmap='gray')

for i, file in enumerate(selection_df.iloc[-10:].index):
    img = np.load(file+'20211114.npy').astype(np.float64)
    ax[3, i].imshow(np.swapaxes(img[:3], 0, 2)/10000)

    ndvi = np.divide((img[NIR_ind] - img[R_ind]), (img[NIR_ind] + img[R_ind]))
    ax[4, i].imshow(ndvi.T)

    mask = np.load(file+'/cloudProb/20211114.npy').astype(np.float64)[0]
    ax[5, i].imshow(mask.T, cmap='gray')

#plt.show()

high_vals = data_manager(selection_df.index[0]).get_data()
lowr_vals = data_manager(selection_df.index[-1]).get_data()
high_vals = high_vals.astype(np.float64)
lowr_vals = lowr_vals.astype(np.float64)

fig, ax = plt.subplots(1, 3, figsize=(15, 6))

ax[0].plot(high_vals[:, NIR_ind].mean(axis=1).mean(axis=1))
ax[0].plot(high_vals[:, R_ind].mean(axis=1).mean(axis=1))

ax[1].plot(lowr_vals[:, NIR_ind].mean(axis=1).mean(axis=1))
ax[1].plot(lowr_vals[:, R_ind].mean(axis=1).mean(axis=1))

num = high_vals[:, NIR_ind] - high_vals[:, R_ind]
den = high_vals[:, NIR_ind] + high_vals[:, R_ind]
ndvi = np.divide(num, den)

ax[2].plot(ndvi.mean(axis=1).mean(axis=1))
plt.show()

import pdb; pdb.set_trace()

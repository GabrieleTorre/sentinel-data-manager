from datetime import datetime
from os.path import join
from glob import glob
import pandas as pd

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

class Indices:

    def normalize(self, band):
        p2, p98 = np.percentile(band, (0.2, 99.8))
        band = np.clip(band, p2, p98)
        return band

    def get_valid_data(self, x):
        imgs = np.load(x['data'])
        mask = np.load(x['masks']).sum(axis=0)
        mask = (mask < self.prob_thresh)
        return imgs.astype(np.float64), mask

    def ndvi(self, x):
        NIR_ind = self.bands['B08']
        R_ind = self.bands['B04']
        x, mask = self.get_valid_data(x)
        ndvi = (x[NIR_ind] - x[R_ind]) / (x[NIR_ind] + x[R_ind])
        return ndvi, mask

    def ndwi(self, x):
        NIR_ind = self.bands['B08']
        SWIR_ind = self.bands['B11']
        x, mask = self.get_valid_data(x)
        ndwi = np.divide((x[SWIR_ind] - x[NIR_ind]), (x[NIR_ind] + x[SWIR_ind]))
        ndwi[(x[NIR_ind] + x[SWIR_ind]) == 0] = np.nan
        return ndwi, mask

    def ndwig(self, x):
        NIR_ind = self.bands['B08']
        GRN_ind = self.bands['B03']
        x, mask = self.get_valid_data(x)
        ndwig = np.divide((x[GRN_ind] - x[NIR_ind]), (x[NIR_ind] + x[GRN_ind]))
        return ndwig, mask

    def msi(self, x):
        NIR_ind = self.bands['B08']
        SWIR_ind = self.bands['B11']
        x, mask = self.get_valid_data(x)
        msi = np.divide(x[SWIR_ind], x[NIR_ind])
        return msi, mask

    def mavi(self, x):
        NIR_ind = self.bands['B08']
        RED_ind = self.bands['B04']
        x, mask = self.get_valid_data(x)
        mavi = (2 * x[NIR_ind] + 1 - ((2 * x[NIR_ind] + 1)**2 - 8 * \
                (x[NIR_ind] - x[RED_ind]))**(1/2)) / 2
        return mavi, mask


class data_manager(Indices):

    def __init__(self, source_dir, prob_thresh=5, img_thresh=0.95):

        self.prob_thresh = prob_thresh # mask threshold probability level
        self.img_thresh = img_thresh   # max percentage of masked pixels allowed

        bands = ["B02", "B03", "B04", "B05", "B06",
                 "B07", "B08", "B8A", "B11", "B12"]
        self.bands = {b: i for i, b in enumerate(bands)}
        
        self.data_fnames = glob(join(source_dir, '20*.npy'))
        self.mask_fnames = glob(join(*[source_dir, 'cloudProb', '20*.npy']))
        
        self.sem_fname  = glob(join(*[source_dir, 'crops_semantic.npy']))[0]
        self.pan_fname  = glob(join(*[source_dir, 'crops_panoptic.npy']))[0]
        
        self.df_records = self.__get_df__()

        valid_selection = self.validate_cloud_snow_coverage()
        self.df_records = self.df_records[valid_selection]

    def __get_df_from_filelist__(self, x):
        index = [datetime.strptime(i.split('/')[-1].split('.')[0], '%Y%m%d')
                 for i in x]
        return pd.Series(data=x, index=index).sort_index()

    def __get_df__(self):
        return pd.concat([
                self.__get_df_from_filelist__(self.data_fnames).rename('data'),
                self.__get_df_from_filelist__(self.mask_fnames).rename('masks')
            ], axis=1)


    def __validate_weather__(self, im):
        return (im < self.prob_thresh).mean() >= self.img_thresh

    def validate_cloud_snow_coverage(self):
        _funct_ = lambda x: self.__validate_weather__(np.load(x)[0])
        return self.df_records.masks.apply(_funct_)

    def get_data(self):
        return np.stack([np.load(x) for x in self.data_fnames], axis=0)

    def descrive_veg_type_(self, seg_type='semantic'):
        if seg_type == 'semantic':
            mask = np.load(self.sem_fname)[0]
        elif seg_type == 'panoptic':
            mask = np.load(self.pan_fname)[0]
        values, counts = np.unique(mask, return_counts=True)
        out =  pd.Series(index=values.astype(int), 
                         data=counts/np.product(mask.shape))
        return out[out.index.isin([0, 19]) == False]
    
    def get_index_ts(self, function, veg_type=None, seg_type='semantic'):
        index, values = [], []
        veg_mask = np.load(self.sem_fname if seg_type == 'semantic' else self.pan_fname)
            
        for i, row in self.df_records.iterrows():

            VI, MK = function(row)
            if veg_type:
                VMK = (veg_mask[0] == veg_type)
            else:
                VMK = (veg_mask[0] != 0) & (veg_mask[0] != 19)
            npost = np.where((MK & VMK).squeeze())

            values.append(np.median(VI[npost]))
            index.append(i)

        return pd.Series(values, index=index, dtype=np.float64)

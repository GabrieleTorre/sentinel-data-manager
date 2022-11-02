from sentinel_2 import data_manager as dm

from shapely.geometry import Polygon, Point
from rasterio.io import MemoryFile
from shapely.ops import transform
import geopandas as gpd
import numpy as np
import rasterio
import pyproj
import cv2
import os


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


class crops_manager():

    def __init__(self, source_dir, tile):
        self.source_dir = source_dir
        self.tile = tile
        self.metadata = self.__get_metadata__()

        self.profile = {'driver': "GTiff",
                        'height': 128,
                        'width': 128,
                        'dtype': 'uint16',
                        'nodata': None,
                        'crs': "EPSG:4326",
                        'tiled': True
                        }

    def __get_metadata__(self):
        meta_df = gpd.GeoDataFrame.from_file(os.path.join(self.source_dir, self.tile, 'metadata.json')).set_index('id')
        return meta_df

    def __retrieve_geo_ref__(self, id_crop, count=1):
        bounds = self.metadata.loc[id_crop].geometry.bounds
        _transform = rasterio.transform.from_bounds(*(bounds + (128, 128)))
        return MemoryFile().open(**{**self.profile, **{'transform': _transform, 'count': count}})

    def __veg_shapes__(self, contours, tiff_f):
        out = []
        for co in contours[1::2]:
            if PolyArea(co[:, 0, 0], co[:, 0, 1]) > 100:
                _shape = []
                for pcol, prow in co.squeeze():
                    _shape.append(tiff_f.xy(prow, pcol))
                out.append(Polygon(_shape))
        return out

    def getId_by_longlat(self, long, lat):
        return self.metadata[self.metadata.contains(Point(long, lat))].index.values

    def getId_by_distance(self, long, lat, min_distance=0, max_distance=2000):
        # The ETRS Lambert Azimuthal Equal Area projection (epsg=3035) to flat up the surface of North Europe.
        project = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3035', always_xy=True).transform
        my_point = transform(project, Point(long, lat))
        _d = self.metadata.to_crs(epsg=3035).centroid.distance(my_point)
        return _d[(_d >= min_distance) & (_d < max_distance)].sort_values()

    def get_veg_shapes(self, id_crop, seg_type='semantic'):
        tiff_f = self.__retrieve_geo_ref__(id_crop)

        veg_mask = dm(os.path.join(self.source_dir, self.tile, id_crop)).get_veg_mask(seg_type=seg_type)[0]
        labels = np.unique(veg_mask).astype(int)
        labels = labels[(labels > 0) & (labels < 19)]

        out = []
        for _l in labels:
            c, h = cv2.findContours((veg_mask == _l).astype(int),
                                    cv2.RETR_FLOODFILL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            shapes = self.__veg_shapes__(c, tiff_f)
            out = out + [{'label': _l, 'geometry': _sh} for _sh in shapes]
        tiff_f.close()
        return gpd.GeoDataFrame(out)

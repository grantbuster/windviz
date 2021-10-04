# -*- coding: utf-8 -*-
"""
Test windviz on simple test data
"""
import os
import shutil
import numpy as np
import pandas as pd

from windviz import WindViz
from datetime import datetime


FP = './test_data.h5'
TEST_DATA_DIR = os.path.dirname(os.path.realpath(__file__))
os.chdir(TEST_DATA_DIR)


def make_data():
    """Utility script for making test data"""
    from reV import Outputs

    lon = np.linspace(-180, 180, 100)
    lat = np.linspace(-90, 90, 50)
    lon, lat = np.meshgrid(lon, lat)
    lon = lon.flatten()
    lat = lat.flatten()
    meta = pd.DataFrame({'longitude': lon, 'latitude': lat})

    time_index = pd.date_range('20210101', '20210131', freq='1h')

    ws = np.abs(np.ones(len(meta)) * lon * lat)
    ws /= np.max(ws)
    ws *= 10
    ws = np.expand_dims(ws, axis=1)
    ws = np.tile(ws, len(time_index)).T

    di = np.zeros((len(time_index), len(meta)))
    di[:, (meta.longitude < 0) & (meta.latitude < 0)] = 225
    di[:, (meta.longitude > 0) & (meta.latitude < 0)] = 135
    di[:, (meta.longitude < 0) & (meta.latitude > 0)] = 315
    di[:, (meta.longitude > 0) & (meta.latitude > 0)] = 45

    with Outputs(FP, mode='w') as f:
        f.meta = meta
        f.time_index = time_index

    f.add_dataset(FP, 'windspeed_100m', ws, dtype=np.int16,
                  attrs={'units': 'm/s', 'scale_factor': 10})
    f.add_dataset(FP, 'winddirection_100m', di, dtype=np.int16,
                  attrs={'units': 'degrees', 'scale_factor': 0.1})


def test_windviz():
    """Test the creation of wind particle flow images"""
    date0 = datetime(2021, 1, 1)
    date1 = datetime(2021, 1, 3)

    WindViz.run(FP, date0, date1, dist_per_vel=50.0, cbar_range=(0, 10),
                n_segments=10, n_saved_steps=9, n_lines=1000,
                dist_threshold=100, linewidth=1,
                random_reset=0.0,
                vel_threshold=0,
                init_all_segs=False,
                make_resource_maps=False,
                marker_size=20,
                fp_gif_out='./test.gif')

    os.remove('./meta_kdtree.pkl')
    shutil.rmtree('./images/')


if __name__ == '__main__':
    make_data()
    test_windviz()

# -*- coding: utf-8 -*-
"""
Wind Visualization

author : Grant Buster
created : 5/8/2020
"""

import math
import os
import pickle
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from scipy.spatial import cKDTree
from rex import Resource
import geopandas as gpd
from datetime import datetime
from PIL import Image


class WindViz:
    """Framework to simulate particle flow for wind data visualization"""

    EARTH_RADIUS = 6371
    TO_RAD = math.pi / 180
    TO_DEG = 180 / math.pi

    def __init__(self,
                 fp,
                 fp_shape=None,
                 ws_dset='windspeed_100m'
                 di_dset='winddirection_100m'
                 fp_out_base='./images/image_{}.png'
                 figsize=(10, 5)
                 dpi=200
                 xlim=(-129, -64)
                 ylim=(23, 51)
                 line_buffer=2
                 cbar_label='100m Windspeed (m/s)'
                 print_timestamp=True
                 logo='./nrel.png'
                 logo_scale=0.7
                 n_lines=int(1000)
                 n_segments=20
                 n_saved_steps=18
                 line_init_option=1
                 sub_iterations=1
                 vel_to_dist=5 * (60 / 1000)
                 vel_threshold=1
                 dist_threshold=1
                 random_reset=0.1
                 linewidth=0.1
                 face_color='k'
                 text_color='#969696'
                 cmap='viridis'
                 max_ws=15
                 min_ws=0
                 shape_color='#2f2f2f'
                 shape_edge_color='k'
                 shape_line_width=1
                 shape_aspect=1.3
                 make_resource_maps=False
                 resource_map_interval=10
                 marker_size=1
                 tree_file='./meta_kdtree.pkl'
                 ):

        self.fp = fp
        self.fp_shape = fp_shape
        self.ws_dset = ws_dset
        self.di_dset = di_dset
        self.fp_out_base = fp_out_base

        self.figsize = figsize
        self.dpi = dpi
        self.xlim = xlim
        self.ylim = ylim
        self.line_buffer = line_buffer
        self.cbar_label = cbar_label
        self.print_timestamp = print_timestamp
        self.logo = logo
        self.logo_scale = logo_scale
        self.n_lines = n_lines
        self.n_segments = n_segments
        self.n_saved_steps = n_saved_steps
        self.line_init_option = line_init_option
        self.sub_iterations = sub_iterations
        self.vel_to_dist = vel_to_dist
        self.vel_threshold = vel_threshold
        self.dist_threshold = dist_threshold
        self.random_reset = random_reset
        self.linewidth = linewidth
        self.face_color = face_color
        self.text_color = text_color
        self.cmap = cmap
        self.max_ws = max_ws
        self.min_ws = min_ws
        self.shape_color = shape_color
        self.shape_edge_color = shape_edge_color
        self.shape_line_width = shape_line_width
        self.shape_aspect = shape_aspect
        self.make_resource_maps = make_resource_maps
        self.resource_map_interval = resource_map_interval
        self.marker_size = marker_size
        self.tree_file = tree_file

        if self.xlim is None:
            self.xlim = (meta.longitude.min(), meta.longitude.max())
        if self.ylim is None:
            self.ylim = (meta.latitude.min(), meta.latitude.max())

        self.offset_x = meta.longitude.min()
        self.offset_y = meta.latitude.min()
        self.scale_x = meta.longitude.max() - meta.longitude.min()
        self.scale_y = meta.latitude.max() - meta.latitude.min()

        cmap_obj = plt.get_cmap(cmap)
        cNorm  = colors.Normalize(vmin=min_ws, vmax=max_ws)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap_obj)
        mpl.rcParams['text.color'] = text_color
        mpl.rcParams['axes.labelcolor'] = text_color

        i_fname = 0

    def init_arrays():
        velocities = np.nan * np.ones((self.n_segments, 1, self.n_lines))
        lines = np.nan * np.ones((self.n_segments, 2, self.n_lines))
        return velocities, lines

    @staticmethod
    def get_time_index(fp, date0, date1):
        """
        Parameters
        ----------
        fp : str
            Filepath to wtk h5 file.
        date0 : datetime.datetime
            Datetime object that is the start of the time frame of
            interest (inclusive).
        date1 : datetime.datetime
            Datetime object that is the end of the time frame of
            interest (exclusive)

        Returns
        -------
        time_slice : slice
            Row slice object that can be used to slice data from fp.
        time_index : pd.Datetimeindex
            Datetimeindex for the time frame of interest
        """

        with Resource(fp) as res:
            time_index = res.time_index

        mask = (time_index >= date0) & (time_index < date1)
        ilocs = np.where(mask)[0]
        time_slice = slice(ilocs[0], ilocs[-1])
        time_index = time_index[time_slice]

        return time_slice, time_index

    @staticmethod
    def get_data(fp, time_slice, ws_dset, di_dset):
        """
        Parameters
        ----------
        fp : str
            Filepath to wtk h5 file.
        time_slice : slice
            Row slice object that can be used to slice data from fp.
        ws_dset : str
            Windspeed dataset (e.g. windspeed_100m)
        di_dset : str
            Wind direction dataset (e.g. winddirection_100m)

        Returns
        -------
        ws : np.ndarray
            2D array (time x sites) of windspeed data in m/s
        di : np.ndarray
            2D array (time x sites) of windspeed direction data in radians
            from west.
        meta : pd.DataFrame
            Meta data from fp.
        """

        logger.info('Reading data...')
        with Resource(fp) as res:
            ws = res[ws_dset, time_slice, :]
            di = res[di_dset, time_slice, :]
            di = 270 - di.copy()
            di = di * (np.pi / 180)
            meta = res.meta

        logger.debug('Data stats: {} {} {}'
                     .format(ws.min(), ws.mean(), ws.max()))
        logger.info('Data read complete')
        return ws, di, meta


    @staticmethod
    def make_tree(meta, tree_file):
        """
        Parameters
        ----------
        meta : pd.DataFrame
            Meta data from fp.
        tree_file : str
            Filepath to pickled ckdtree.

        Returns
        -------
        tree : cKDTree
        """
        if not os.path.exists(tree_file):
            logger.info('making tree...')
            tree = cKDTree(meta[['longitude', 'latitude']])
            logger.info('tree complete')
            with open(tree_file, 'wb') as pkl:
                pickle.dump(tree, pkl)
        else:
            logger.info('loading tree...')
            with open(tree_file, 'rb') as pkl:
                tree = pickle.load(pkl)
            logger.info('Tree loaded')

        return tree

    @classmethod
    def dist_to_latitude(cls, dist):
        """Calculate change in latitude in decimal degrees given distance
        differences north/south in km..

        Parameters
        ----------
        dist : np.ndarray
            Array of change in east/west location in km.

        Returns
        -------
        delta_lat : np.ndarray
            Array of change in north/south location in decimal degrees.
        """
        delta_lat = (dist / cls.EARTH_RADIUS) * cls.TO_DEG
        return delta_lat


    @classmethod
    def dist_to_longitude(cls, latitude, dist):
        """Calculate change in longitude in decimal degrees given a latitude
        and distance differences east/west in km.

        Parameters
        ----------
        latitude : np.ndarray
            Array of latitude values in decimal degrees.
        dist : np.ndarray
            Array of change in east/west location in km.

        Returns
        -------
        delta_lon : np.ndarray
            Array of change in east/west location in decimal degrees.
        """
        # Find the radius of a circle around the earth at given latitude.
        r = cls.EARTH_RADIUS * np.cos(latitude * cls.TO_RAD)
        delta_lon = (dist / r) * cls.TO_DEG
        return delta_lon



    def run_lines(self, ti, timestamp, lines, velocities)
        lines = np.roll(lines, self.n_saved_steps, axis=0)
        lines[self.n_saved_steps:, :, :] = np.nan

        # generate new initial coordinates at every timestep
        init_coords = np.random.random((1, 2, self.n_lines))
        init_coords[:, 0, :] *= scale_x
        init_coords[:, 1, :] *= scale_y
        init_coords[:, 0, :] += offset_x
        init_coords[:, 1, :] += offset_y

        # reset initial coordinates for lines that have escaped
        v_reset_mask = velocities[-1, 0, :] < self.vel_threshold
        x_reset_mask = ((lines[-1, 0, :] > np.max(self.xlim) + self.line_buffer)
                        | (lines[-1, 0, :] < np.min(self.xlim) - self.line_buffer))
        y_reset_mask = ((lines[-1, 1, :] > np.max(self.ylim) + self.line_buffer)
                        | (lines[-1, 1, :] < np.min(self.ylim) - self.line_buffer))

        reset_mask = x_reset_mask | y_reset_mask | v_reset_mask
        if random_reset is not None:
            n_rand_reset = int(random_reset * self.n_lines)
            rand_ind = np.random.choice(self.n_lines, n_rand_reset)
            reset_mask[rand_ind] = True

        lines[:, :, reset_mask] = np.nan
        velocities[:, :, reset_mask] = np.nan

        # initial coordinates for nan values
        mask = np.isnan(lines)
        init_mask = mask[0, :, :].reshape((1, 2, self.n_lines))
        line_init_mask = mask.copy()
        line_init_mask[1:, :, :] = False
        lines[line_init_mask] = init_coords[init_mask]
        if line_init_option == 1:
            for i in range(1, lines.shape[0]):
                temp_mask = mask[0, :, :]
                lines[i, temp_mask] = init_coords[init_mask]


        for i in range(1, lines.shape[0]):
            query_coords = lines[i-1, :, :].T
            d, ind = tree.query(query_coords)
            vel = ws[ti][ind]
            vel[d > dist_threshold] = 0.0
            velocities[i, 0, :] = vel
            direction = di_rad[ti][ind]

            dx = vel_to_dist * vel * np.cos(direction)
            dy = vel_to_dist * vel * np.sin(direction)
            dy = dist_to_latitude(dy)
            lat = lines[i-1, 1, :]
            dx = dist_to_longitude(lat, dx)
            mask = np.isnan(lines[i, 0, :])
            lines[i, 0, mask] = lines[i-1, 0, mask] + dx[mask]
            lines[i, 1, mask] = lines[i-1, 1, mask] + dy[mask]



    def make_resource_maps():
        if make_resource_maps:
            dset_names = [ws_dset, di_dset]
            res_datasets = [ws, di_raw]
            for dset_name, res_data in zip(dset_names, res_datasets):
                res_plot_fname = dset_name + '_{}.png'
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot()
                col_slice = slice(None)
                if resource_map_interval:
                    col_slice = slice(None, None, resource_map_interval)
                if fp_shape is not None:
                    gdf = gpd.GeoDataFrame.from_file(fp_shape)
                    gdf = gdf.to_crs({'init': 'epsg:4326'})
                    gdf.plot(ax=ax, color=shape_color,
                             edgecolor=shape_edge_color,
                             linewidth=shape_line_width)
                    if shape_aspect:
                        ax.set_aspect(shape_aspect)
                a = ax.scatter(meta.longitude.values[col_slice],
                               meta.latitude.values[col_slice],
                               c=res_data[ti, col_slice],
                               cmap=cmap, s=marker_size)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                cbar = plt.colorbar(a)
                cbar.set_label(dset_name)
                plt.savefig(res_plot_fname.format(ti))
                plt.close()

    def draw():

        fig = plt.figure(figsize=figsize, facecolor=face_color)
        ax = fig.add_subplot()
        fig.patch.set_facecolor(face_color)
        ax.patch.set_facecolor(face_color)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if fp_shape is not None:
            gdf = gpd.GeoDataFrame.from_file(fp_shape)
            gdf = gdf.to_crs({'init': 'epsg:4326'})
            gdf.plot(ax=ax, color=shape_color, edgecolor=shape_edge_color,
                     linewidth=shape_line_width)
            if shape_aspect:
                ax.set_aspect(shape_aspect)

        line_inputs = []
        colors = []
        for k in range(lines.shape[2]):
            v_line = np.nanmean(velocities[:, 0, k])
            color = scalarMap.to_rgba(v_line)
            line_inputs.append(lines[:, :, k])
            colors.append(color)

        collection = mpl.collections.LineCollection(line_inputs,
                                                    colors=colors,
                                                    linewidth=linewidth)
        ax.add_collection(collection)

        n_cbar = 100
        b = ax.scatter([1e6] * n_cbar, [1e6] * n_cbar,
                       c=np.linspace(min_ws, max_ws, n_cbar),
                       s=0.00001, cmap=cmap)
        cbar = plt.colorbar(b)
        ticks = plt.getp(cbar.ax, 'yticklabels')
        plt.setp(ticks, color=text_color)
        cbar.ax.tick_params(which='minor', color=text_color)
        cbar.ax.tick_params(which='major', color=text_color)
        cbar.set_label(cbar_label, color=text_color)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if print_timestamp:
            plt.text(0.12, 0.87, str(timestamp), transform=fig.transFigure)

        if isinstance(logo, str):
            if not isinstance(logo_scale, float):
                logo_scale = 1.0
            im = Image.open(logo)
            size = (int(logo_scale * im.size[0]), int(logo_scale * im.size[1]))
            im = im.resize(size)
            im = np.array(im).astype(float) / 255
            fig.figimage(im, 0, 0)

        fn = fn_base.format(i_fname)
        i_fname += 1
        fig.savefig(fn, dpi=dpi, bbox_inches='tight', facecolor=face_color)
        logger.info('saved: ', fn)
        plt.close()


    def run():
        for ti, timestamp in enumerate(time_index):
            logger.info('working on ti {}, timestamp: {}'
                        .format(ti, timestamp))
            for _ in range(sub_iterations):
                out = self.run_lines()
                self.draw(out)

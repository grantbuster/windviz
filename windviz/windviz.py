# -*- coding: utf-8 -*-
"""
Wind Visualization with particle flow.

author : Grant Buster
created : 5/8/2020
"""

import math
import os
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from scipy.spatial import cKDTree
from rex import Resource
from PIL import Image
import logging


logger = logging.getLogger(__name__)


class WindViz:
    """Framework to simulate particle flow for wind data visualization"""

    EARTH_RADIUS = 6371
    TO_RAD = math.pi / 180
    TO_DEG = 180 / math.pi

    def __init__(self, fp_res, date0, date1,
                 dset='windspeed_100m',
                 di_dset='winddirection_100m',
                 di_option='from',
                 fp_out_base='./images/image_{}.png',
                 figsize=(10, 5),
                 dpi=200,
                 xlim=(-180, 180),
                 ylim=(-90, 90),
                 line_buffer=2.0,
                 cbar_label='100m Windspeed (m/s)',
                 cbar_range=(0, 15),
                 print_timestamp=True,
                 timestamp_loc=(0.12, 0.87),
                 fp_logo=None,
                 logo_scale=0.7,
                 n_lines=int(1000),
                 n_segments=20,
                 n_saved_steps=18,
                 init_all_segs=False,
                 sub_iterations=1,
                 dist_per_vel=0.3,
                 vel_threshold=0.0,
                 dist_threshold=1.0,
                 random_reset=0.1,
                 linewidth=0.1,
                 face_color='k',
                 text_color='#969696',
                 cmap='viridis',
                 fp_shape=None,
                 shape_color='#2f2f2f',
                 shape_edge_color='k',
                 shape_line_width=1.0,
                 shape_aspect=1.3,
                 make_resource_maps=False,
                 resource_map_interval=1,
                 marker_size=1.0,
                 tree_file='./meta_kdtree.pkl',
                 ):
        """
        Parameters
        ----------
        fp_res : str
            Filepath to resource file (wtk h5 file).
        date0 : datetime.datetime
            Datetime object that is the start of the time frame of
            interest (inclusive).
        date1 : datetime.datetime
            Datetime object that is the end of the time frame of
            interest (exclusive)
        dset : str
            Magnitude/color dataset (e.g. windspeed_100m)
        di_dset : str
            Direction dataset (e.g. winddirection_100m). Typically this is the
            source direction in degrees from north unless di_option="to"
        di_option : str
            Option to specify whether the direction is "from" or "to"
        fp_out_base : str
            Filepath to output image with "{}" to format the image index.
        figsize : tuple
            matplotlib output image size in inches
        dpi : int
            Dots per inch for output image
        xlim : tuple
            X axis bounds (longitude) in decimal degrees.
        ylim : tuple
            Y axis bounds (latitude) in decimal degrees.
        line_buffer : float
            Buffer on xlim and ylim to allow for parts of lines to exit the
            image bounds.
        cbar_label : str
            Colorbar label
        cbar_range : tuple
            Data range for the colorbar corresponding to the dset data
        print_timestamp : bool
            Flag to print the timestamp in the image.
        timestamp_loc : tuple
            Location to print timestamp
        fp_logo : str | None
            Optional filepath to image file to place in bottom left of image.
        logo_scale : float
            Scaling factor to size the logo.
        n_lines : int
            Number of lines to plot at one time.
        n_segments : int
            Number of segments to plot for each line
        n_saved_steps : int
            Number of line segments to save from the previous timestep.
        init_all_segs : bool
            Flag to initialize a line with all segments or just one.
        sub_iterations : int
            Number of times to advance lines per data timestep.
        dist_per_vel : float
            Scale factor specifying how far to advance each line given a
            velocity of the line in (km / (m/s))
        vel_threshold : float
            Velocity value (m/s) below which lines are reset.
        dist_threshold : float
            Distance threshold in decimal degrees. Lines further than this
            value from the nearest data will be respawned.
        random_reset : float
            Value between 0 and 1 specifying a rate at which lines will
            be randomly reset.
        linewidth : float
            Width of the particle flow traces
        face_color : str
            Color of the image background.
        text_color : str
            Color of label and timestamp text.
        cmap : str
            matplotlib colormap name
        fp_shape : str | None
            Optional filepath to a shape file to plot on the image with
            geopandas.
        shape_color : str
            Color of the shape face.
        shape_edge_color : str
            Color of the shape borders.
        shape_line_width : float
            Line width of the shape borders.
        shape_aspect : float
            Plotting aspect ratio of the shape. This can affect the
            figure size.
        make_resource_maps : bool
            Flag to save bitmaps of dset and di_dset
        resource_map_interval : int
            Interval to plot every n pixels for resource bitmaps.
        marker_size : float
            Marker size for the resource bitmaps
        tree_file : str
            Filepath for cKDTree caching.
        """

        self.fp_res = fp_res
        self.date0 = date0
        self.date1 = date1
        self.dset = dset
        self.di_dset = di_dset
        self.di_option = di_option
        self.fp_out_base = fp_out_base

        self.data = None
        self.di = None
        self.tree = None
        self.meta = None
        self.time_slice = None
        self.time_index = None

        self.figsize = figsize
        self.dpi = dpi
        self.xlim = xlim
        self.ylim = ylim
        self.line_buffer = line_buffer
        self.cbar_label = cbar_label
        self.print_timestamp = print_timestamp
        self.timestamp_loc = timestamp_loc
        self.fp_logo = fp_logo
        self.logo_scale = logo_scale
        self.n_lines = int(n_lines)
        self.n_segments = int(n_segments)
        self.n_saved_steps = int(n_saved_steps)
        self.init_all_segs = init_all_segs
        self.sub_iterations = sub_iterations
        self.dist_per_vel = dist_per_vel
        self.vel_threshold = vel_threshold
        self.dist_threshold = dist_threshold
        self.random_reset = random_reset
        self.linewidth = linewidth
        self.face_color = face_color
        self.text_color = text_color
        self.cmap = cmap
        self.cbar_range = cbar_range
        self.fp_shape = fp_shape
        self.shape_color = shape_color
        self.shape_edge_color = shape_edge_color
        self.shape_line_width = shape_line_width
        self.shape_aspect = shape_aspect
        self.make_resource_maps = make_resource_maps
        self.resource_map_interval = resource_map_interval
        self.marker_size = marker_size
        self.tree_file = tree_file

        self.get_meta()

        if self.xlim is None:
            self.xlim = (self.meta.longitude.min(), self.meta.longitude.max())
        if self.ylim is None:
            self.ylim = (self.meta.latitude.min(), self.meta.latitude.max())

        self.offset_x = self.meta.longitude.min()
        self.offset_y = self.meta.latitude.min()
        self.scale_x = self.meta.longitude.max() - self.meta.longitude.min()
        self.scale_y = self.meta.latitude.max() - self.meta.latitude.min()

        cmap_obj = plt.get_cmap(cmap)
        cNorm = colors.Normalize(vmin=np.min(self.cbar_range),
                                 vmax=np.max(self.cbar_range))
        self.scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap_obj)
        mpl.rcParams['text.color'] = self.text_color
        mpl.rcParams['axes.labelcolor'] = self.text_color
        self.i_fname = 0

        img_dir = os.path.dirname(os.path.abspath(self.fp_out_base))
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

    def get_time_index(self):
        """
        Returns
        -------
        time_slice : slice
            Row slice object that can be used to slice data from fp_res.
        time_index : pd.Datetimeindex
            Datetimeindex for the time frame of interest
        """

        with Resource(self.fp_res) as res:
            time_index = res.time_index.tz_localize(None)

        mask = (time_index >= self.date0) & (time_index < self.date1)
        ilocs = np.where(mask)[0]
        self.time_slice = slice(ilocs[0], ilocs[-1])
        self.time_index = time_index[self.time_slice]

        return self.time_slice, self.time_index

    def get_meta(self):
        """
        Returns
        -------
        meta : pd.DataFrame
            Meta data from fp_res.
        """

        with Resource(self.fp_res) as res:
            self.meta = res.meta

        return self.meta

    def get_data(self):
        """
        Returns
        -------
        data : np.ndarray
            2D array (time x sites) of the primary dset (e.g. windspeed in m/s)
        di : np.ndarray
            2D array (time x sites) of direction data in radians measured
            counter-clockwise from east such that dx = delta*np.cos(di) and
            dy = delta*np.sin(di). This also implies that if
            self.di_option="from" then this di output has converted the
            direction to "to"
        """

        logger.info('Reading data...')
        with Resource(self.fp_res) as res:
            self.data = res[self.dset, self.time_slice, :]
            di = res[self.di_dset, self.time_slice, :]

            if self.di_option.lower() == 'from':
                di = 270 - di.copy()
            elif self.di_option.lower() == 'to':
                di = 90 - di.copy()

            self.di = di * (np.pi / 180)

        logger.debug('Data stats: {} {} {}'
                     .format(self.data.min(), self.data.mean(),
                             self.data.max()))
        logger.info('Data read complete')

        return self.data, self.di

    def make_tree(self):
        """
        Returns
        -------
        tree : cKDTree
            cKDTree of meta lat/lon
        """
        if not os.path.exists(self.tree_file):
            logger.info('making tree...')
            self.tree = cKDTree(self.meta[['longitude', 'latitude']])
            logger.info('tree complete')
            with open(self.tree_file, 'wb') as pkl:
                pickle.dump(self.tree, pkl)
        else:
            logger.info('loading tree...')
            with open(self.tree_file, 'rb') as pkl:
                self.tree = pickle.load(pkl)
            logger.info('Tree loaded')

        return self.tree

    def init_arrays(self):
        """
        Returns
        -------
        velocities : np.ndarray
            3D array of velocities for each line:
            (number_of_segments_per_line, 1, number_of_lines)
        lines : np.ndarray
            3D array of coordinates for each segment of each lines:
            (number_of_segments_per_line, xy_coords, number_of_lines)
        """
        velocities = np.nan * np.ones((self.n_segments, 1, self.n_lines))
        lines = np.nan * np.ones((self.n_segments, 2, self.n_lines))
        return velocities, lines

    @staticmethod
    def get_line_means(velocities, lines):
        """Get line mean coordinates and mean velocities

        Returns
        -------
        mean_coord : np.ndarray
            2D array of line coordinates (line, coordinate) where coordinate
            is 2 columns for x and y.
        mean_velocities : np.ndarray
            1D array of mean line velocities
        """
        return lines.mean(axis=0).T, np.nanmean(velocities[:, 0, :], axis=0)

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

    def advance_lines(self, ti, velocities, lines):
        """Advance the position of the lines based on their nearest velocities
        and directions.

        Parameters
        ----------
        ti : int
            Enumerated time index corresponding to timestamp
        velocities : np.ndarray
            3D array of velocities for each line:
            (number_of_segments_per_line, 1, number_of_lines)
        lines : np.ndarray
            3D array of coordinates for each segment of each lines:
            (number_of_segments_per_line, xy_coords, number_of_lines)

        Returns
        -------
        velocities : np.ndarray
            3D array of velocities for each line:
            (number_of_segments_per_line, 1, number_of_lines)
        lines : np.ndarray
            3D array of coordinates for each segment of each lines:
            (number_of_segments_per_line, xy_coords, number_of_lines)
        """
        logger.info('Starting line advancement for {}'.format(ti))

        # roll line coordinates forward in the n_segments axis
        lines = np.roll(lines, self.n_saved_steps, axis=0)
        lines[self.n_saved_steps:, :, :] = np.nan

        # reset initial coordinates for lines that have escaped the xy lims
        xmax = np.max(self.xlim) + self.line_buffer
        xmin = np.min(self.xlim) - self.line_buffer
        ymax = np.max(self.ylim) + self.line_buffer
        ymin = np.min(self.ylim) - self.line_buffer
        v_reset_mask = velocities[-1, 0, :] < self.vel_threshold
        x_reset_mask = (lines[-1, 0, :] > xmax) | (lines[-1, 0, :] < xmin)
        y_reset_mask = (lines[-1, 1, :] > ymax) | (lines[-1, 1, :] < ymin)

        # set a mask saying which lines to reset
        reset_mask = x_reset_mask | y_reset_mask | v_reset_mask

        # reset a certain number of lines randomly
        if self.random_reset is not None:
            n_rand_reset = int(self.random_reset * self.n_lines)
            rand_ind = np.random.choice(self.n_lines, n_rand_reset)
            reset_mask[rand_ind] = True

        # reset all coordinates and velocities for each line in reset_mask
        lines[:, :, reset_mask] = np.nan
        velocities[:, :, reset_mask] = np.nan

        # generate new initial coordinates at every timestep
        init_coords = np.random.random((1, 2, self.n_lines))
        init_coords[:, 0, :] *= self.scale_x
        init_coords[:, 1, :] *= self.scale_y
        init_coords[:, 0, :] += self.offset_x
        init_coords[:, 1, :] += self.offset_y

        # initial coordinates for nan values
        mask = np.isnan(lines)
        init_mask = mask[0, :, :].reshape((1, 2, self.n_lines))
        line_init_mask = mask.copy()
        line_init_mask[1:, :, :] = False
        lines[line_init_mask] = init_coords[init_mask]

        # Use line init option
        if not self.init_all_segs:
            for i in range(1, lines.shape[0]):
                temp_mask = mask[0, :, :]
                lines[i, temp_mask] = init_coords[init_mask]

        # use velocities to calculate the full line segments
        for i in range(1, lines.shape[0]):
            query_coords = lines[i - 1, :, :].T
            d, ind = self.tree.query(query_coords)
            vel = self.data[ti][ind]
            vel[d > self.dist_threshold] = 0.0
            velocities[i, 0, :] = vel
            direction = self.di[ti][ind]
            dx = self.dist_per_vel * vel * np.cos(direction)
            dy = self.dist_per_vel * vel * np.sin(direction)
            dy = self.dist_to_latitude(dy)
            lat = lines[i - 1, 1, :]
            dx = self.dist_to_longitude(lat, dx)

            mask = np.isnan(lines[i, 0, :])
            lines[i, 0, mask] = lines[i - 1, 0, mask] + dx[mask]
            lines[i, 1, mask] = lines[i - 1, 1, mask] + dy[mask]

        return velocities, lines

    def save_resource_maps(self, ti):
        """Save resource (windspeed and direction) bitmaps for a given time
        index

        Parameters
        ----------
        ti : int
            Enumerated time index
        """
        dset_names = [self.dset, self.di_dset]
        res_datasets = [self.data, self.di]
        for dset_name, res_data in zip(dset_names, res_datasets):
            res_plot_fname = self.fp_out_base.replace(
                '.png', '_{}.png'.format(dset_name))
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot()
            col_slice = slice(None)
            if self.resource_map_interval:
                col_slice = slice(None, None, self.resource_map_interval)
            if self.fp_shape is not None:
                import geopandas as gpd
                gdf = gpd.GeoDataFrame.from_file(self.fp_shape)
                gdf = gdf.to_crs({'init': 'epsg:4326'})
                gdf.plot(ax=ax, color=self.shape_color,
                         edgecolor=self.shape_edge_color,
                         linewidth=self.shape_line_width)
                if self.shape_aspect:
                    ax.set_aspect(self.shape_aspect)

            a = ax.scatter(self.meta.longitude.values[col_slice],
                           self.meta.latitude.values[col_slice],
                           c=res_data[ti, col_slice],
                           cmap=self.cmap, s=self.marker_size)

            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            cbar = plt.colorbar(a)
            cbar.set_label(dset_name)
            plt.savefig(res_plot_fname.format(ti))
            plt.close()

    def draw(self, velocities, lines, timestamp):
        """Draw a particle flow image and save to file.

        Parameters
        ----------
        velocities : np.ndarray
            3D array of velocities for each line:
            (number_of_segments_per_line, 1, number_of_lines)
        lines : np.ndarray
            3D array of coordinates for each segment of each lines:
            (number_of_segments_per_line, xy_coords, number_of_lines)
        timestamp : timestamp
            Single timestamp from the pandas Datetimeindex object.

        Returns
        -------
        fp_out : str
            Filepath to the output image
        """

        fig = plt.figure(figsize=self.figsize, facecolor=self.face_color)
        ax = fig.add_subplot()
        fig.patch.set_facecolor(self.face_color)
        ax.patch.set_facecolor(self.face_color)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if self.fp_shape is not None:
            import geopandas as gpd
            gdf = gpd.GeoDataFrame.from_file(self.fp_shape)
            gdf = gdf.to_crs({'init': 'epsg:4326'})
            gdf.plot(ax=ax, color=self.shape_color,
                     edgecolor=self.shape_edge_color,
                     linewidth=self.shape_line_width)
            if self.shape_aspect:
                ax.set_aspect(self.shape_aspect)

        line_inputs = []
        colors = []
        for k in range(lines.shape[2]):
            v_line = np.nanmean(velocities[:, 0, k])
            color = self.scalarMap.to_rgba(v_line)
            line_inputs.append(lines[:, :, k])
            colors.append(color)

        collection = mpl.collections.LineCollection(line_inputs,
                                                    colors=colors,
                                                    linewidth=self.linewidth)
        ax.add_collection(collection)

        n_cbar = 100
        b = ax.scatter([1e6] * n_cbar, [1e6] * n_cbar,
                       c=np.linspace(np.min(self.cbar_range),
                                     np.max(self.cbar_range), n_cbar),
                       s=0.00001, cmap=self.cmap)
        cbar = plt.colorbar(b)
        ticks = plt.getp(cbar.ax, 'yticklabels')
        plt.setp(ticks, color=self.text_color)
        cbar.ax.tick_params(which='minor', color=self.text_color)
        cbar.ax.tick_params(which='major', color=self.text_color)
        cbar.set_label(self.cbar_label, color=self.text_color)
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        if self.print_timestamp:
            plt.text(self.timestamp_loc[0], self.timestamp_loc[1],
                     str(timestamp), transform=fig.transFigure)

        if isinstance(self.fp_logo, str):
            if not isinstance(self.logo_scale, float):
                self.logo_scale = 1.0
            im = Image.open(self.fp_logo)
            size = (int(self.logo_scale * im.size[0]),
                    int(self.logo_scale * im.size[1]))
            im = im.resize(size)
            im = np.array(im).astype(float) / 255
            fig.figimage(im, 0, 0)

        fp_out = self.fp_out_base.format(self.i_fname)
        self.i_fname += 1
        fig.savefig(fp_out, dpi=self.dpi, bbox_inches='tight',
                    facecolor=self.face_color)
        logger.info('Saved: {}'.format(fp_out))
        plt.close()
        return fp_out

    @classmethod
    def run(cls, fp_res, date0, date1, fp_gif_out=None, duration=100,
            **kwargs):
        """Run the windviz and save images to disk.

        Parameters
        ----------
        fp_res : str
            Filepath to wtk h5 file.
        date0 : datetime.datetime
            Datetime object that is the start of the time frame of
            interest (inclusive).
        date1 : datetime.datetime
            Datetime object that is the end of the time frame of
            interest (exclusive)
        fp_gif_out : str | None
            Optional filepath to make a gif from the generated images
        duration : int
            Duration of each image frame in the gif in ms
        kwargs : dict
            Namespace of kwargs for WindViz initialization.
        """

        obj = cls(fp_res, date0, date1, **kwargs)

        obj.get_time_index()
        obj.get_data()
        obj.make_tree()
        obj.init_arrays()
        velocities, lines = obj.init_arrays()

        fps = []
        for ti, timestamp in enumerate(obj.time_index):
            logger.info('Working on index {}, timestamp: {}'
                        .format(ti, timestamp))
            for _ in range(obj.sub_iterations):
                velocities, lines = obj.advance_lines(ti, velocities, lines)
                fp_out = obj.draw(velocities, lines, timestamp)
                fps.append(fp_out)
                if obj.make_resource_maps:
                    obj.save_resource_maps(ti)

        if fp_gif_out is not None:
            img, *imgs = [Image.open(fp) for fp in fps]
            img.save(fp=fp_gif_out, format='GIF', append_images=imgs,
                     save_all=True, duration=duration, loop=0)
            logger.info('Saved gif: {}'.format(fp_gif_out))

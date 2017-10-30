# codes to import packages needed
import os
import glob
import sys
from os.path import join as pjoin
import subprocess

import numpy as np
from scipy import (io, special, spatial,
                   interpolate, integrate, optimize)

import matplotlib
from matplotlib import (pyplot as plt, animation, colors,
                        ticker, path, patches, patheffects)

from mpl_toolkits.axes_grid1 import make_axes_locatable

import string

import warnings
warnings.simplefilter("ignore")

from sklearn import (decomposition, preprocessing, cluster,
                     preprocessing, cluster, metrics,
                     model_selection, neighbors)

simps = integrate.simps
curve_fit = optimize.curve_fit
Path = path.Path
PathPatch = patches.PathPatch
Convex_Hull = spatial.ConvexHull
erf = special.erf

# Function to save figures
def savefig(filename, **kwargs):

    # Saves figures at EPS
    if print_EPS:
        plt.savefig(filename + '.eps', format='eps',
                    dpi=600, bbox_inches='tight')

    # Saves figures as PNG
    if print_PNG:
        plt.savefig(filename + '.png', format='png',
                    dpi=600, bbox_inches='tight')



# Function to add text labels to figure
def labelfigs(axis, number, style='wb', loc='br', string_add='',size=14,text_pos = 'center'):

    # Sets up various color options
    formating_key = {'wb': dict(color='w',
                                linewidth=1.5),
                     'b': dict(color='k',
                               linewidth=0),
                     'w': dict(color='w',
                               linewidth=0)}

    # Stores the selected option
    formatting = formating_key[style]

    # finds the position for the label
    x_min, x_max = axis.get_xlim()
    y_min, y_max = axis.get_ylim()
    x_value = .08 * (x_max - x_min) + x_min

    if loc == 'br':
        y_value = y_max - .1 * (y_max - y_min)
        x_value = .08 * (x_max - x_min) + x_min
    elif loc == 'tr':
        y_value = y_max - .9 * (y_max - y_min)
        x_value = .08 * (x_max - x_min) + x_min
    elif loc == 'bl':
        y_value = y_max - .1 * (y_max - y_min)
        x_value = x_max -.08 * (x_max - x_min)
    elif loc == 'tl':
        y_value = y_max - .9 * (y_max - y_min)
        x_value = x_max -.08 * (x_max - x_min)
    elif loc == 'tm':
        y_value = y_max - .9 * (y_max - y_min)
        x_value = x_min + (x_max - x_min)/2
    elif loc == 'bm':
        y_value = y_max - .1 * (y_max - y_min)
        x_value = x_min + (x_max - x_min)/2

    if string_add == '':

        # Turns to image number into a label
        if number < 26:
            axis.text(x_value, y_value, string.ascii_lowercase[number],
                      size=14, weight='bold', ha=text_pos,
                      va='center', color=formatting['color'],
                      path_effects=[patheffects.withStroke(linewidth=formatting['linewidth'],
                                                           foreground="k")])

        # allows for double letter index
        else:
            axis.text(x_value, y_value, string.ascii_lowercase[0] + string.ascii_lowercase[number - 26],
                      size=14, weight='bold', ha=text_pos,
                      va='center', color=formatting['color'],
                      path_effects=[patheffects.withStroke(linewidth=formatting['linewidth'],
                                                           foreground="k")])
    else:

        axis.text(x_value, y_value, string_add,
                      size=14, weight='bold', ha=text_pos,
                      va='center', color=formatting['color'],
                      path_effects=[patheffects.withStroke(linewidth=formatting['linewidth'],
                                                           foreground="k")])

def add_scalebar_to_figure(axis, image_size, scale_size, units='nm', loc='br'):

    x_lim, y_lim = axis.get_xlim(), axis.get_ylim()
    x_size, y_size = np.abs(np.floor(x_lim[1] - x_lim[0])), np.abs(np.floor(y_lim[1] - y_lim[0]))

    fract = scale_size / image_size

    x_point = np.linspace(x_lim[0], x_lim[1], np.floor(image_size))
    y_point = np.linspace(y_lim[0], y_lim[1], np.floor(image_size))

    if loc == 'br':
        x_start = x_point[np.int(.9 * image_size // 1)]
        x_end = x_point[np.int((.9 - fract) * image_size // 1)]
        y_start = y_point[np.int(.1 * image_size // 1)]
        y_end = y_point[np.int((.1 + .025) * image_size // 1)]
        y_label_height = y_point[np.int((.1 + .075) * image_size // 1)]
    elif loc == 'tr':
        x_start = x_point[np.int(.9 * image_size // 1)]
        x_end = x_point[np.int((.9 - fract) * image_size // 1)]
        y_start = y_point[np.int(.9 * image_size // 1)]
        y_end = y_point[np.int((.9 - .025) * image_size // 1)]
        y_label_height = y_point[np.int((.9 - .075) * image_size // 1)]

    path_maker(axis, [x_start, x_end, y_start, y_end], 'w', 'k', '-', 1)

    axis.text((x_start + x_end) / 2,
              y_label_height,
              '{0} {1}'.format(scale_size, units),
              size=14, weight='bold', ha='center',
              va='center', color='w',
              path_effects=[patheffects.withStroke(linewidth=1.5,
                                                   foreground="k")])

# function which plot the piezoelectric hysteresis loop and loop fitted results
def plot_loop_and_fit(Voltagedata, LoopFitResults, Loopdata, color, just_fit=False):

    if just_fit:
        # Plots the top branch
        plt.plot(Voltagedata[0:len(Voltagedata) // 2 + 1],
                 LoopFitResults['Branch1'],
                 'r', lw=.5)

        # Plots the bottom branch
        plt.plot(Voltagedata[0:len(Voltagedata) // 2 + 1],
                 LoopFitResults['Branch2'],
                 'r', lw=.5)
    else:
        # Plots the top branch
        plt.plot(Voltagedata[0:len(Voltagedata) // 2 + 1],
                 LoopFitResults['Branch1'] - np.mean(Loopdata),
                 'k')

        # Plots the bottom branch
        plt.plot(Voltagedata[0:len(Voltagedata) // 2 + 1],
                 LoopFitResults['Branch2'] - np.mean(Loopdata),
                 'k')

        # Plots the amplitude normalized piezoelectric hysteresis loop
        plt.plot(Voltagedata,
                 Loopdata - np.mean(Loopdata),
                 '-s', color=color, markerfacecolor='none')

def Construct_Linear_Spline(PCA_reconstructed_data,
                            convex_hull,
                            voltage, start_value):

    # Finds the start of the array
    low_ind = np.argmin(convex_hull.vertices)

    # Reorders the array to start at the begining
    hull_ordered = np.append(convex_hull.vertices[low_ind::], convex_hull.vertices[0:low_ind])

    # Finds the high and low index
    high_ind = np.argmax(hull_ordered)
    low_ind = np.argmin(hull_ordered)

    # Computes the linear spline of the convex vertexes. Produces convex hull construction
    convex_values = hull_ordered[0:high_ind + 1]
    spline_fit = interpolate.interp1d(voltage[convex_values],
                                      PCA_reconstructed_data[convex_values + start_value],
                                      kind='linear')

    return (spline_fit, convex_values)

def Construct_Low_Rank_Representation(PC_Reconstruct, loop_data):

    """
    Computes the low rank representation of the piezoelectric hystersis loops.\n
    Used for statistical denoising of data. Uses PCA as an autoencoder

    Parameters
    ----------
    PC_Reconstruct : int
        number of principal components to include
    loop_data : numpy array
        array of loops

    Returns
    -------
    loop_data_low_rank_representation : numpy array
        low rank representation of piezoelectric hysteresis loops
    """

    # Defines the random seed for consistent clustering
    np.random.seed(42)

    # Sets the number of componets to save
    pca = decomposition.PCA(n_components=PC_Reconstruct)

    # Computes the PCA of the piezoelectric hysteresis loops
    PCA_data = pca.fit(loop_data)

    # Does the inverse tranform - creates a low rank representation of the data
    PCA_reconstructed_data = pca.inverse_transform(pca.transform(loop_data))

    return (PCA_reconstructed_data)

def compute_convex_hull(voltage_top, voltage_bottom, PCA_reconstructed_data):

    # Preallocates the matrix
    convex_hull_diff = np.zeros(PCA_reconstructed_data.shape)
    convex_hull_data = np.zeros(PCA_reconstructed_data.shape)

    for i in range(PCA_reconstructed_data.shape[0]):

        voltage_len = len(voltage_top) + len(voltage_bottom)

        # Calculates the convex hull of the top and bottom branches
        hull_top = Convex_Hull(np.vstack((voltage_top,
                                          PCA_reconstructed_data[i, 0:voltage_len // 2])).T)
        hull_bot = Convex_Hull(np.vstack((voltage_bottom,
                                          PCA_reconstructed_data[i, voltage_len // 2:])).T)

        # Calculates a linear spline for the convex hull
        spline_fit_top, _ = Construct_Linear_Spline(PCA_reconstructed_data[i],
                                                    hull_top,
                                                    voltage_top,
                                                    0)

        spline_fit_bottom, _ = Construct_Linear_Spline(PCA_reconstructed_data[i],
                                                       hull_bot,
                                                       voltage_bottom,
                                                       voltage_len // 2)

        # Calculates the difference between the convex hull and piezoelectric hysteresis loop
        convex_difference_top = np.abs(
            PCA_reconstructed_data[i, 0:voltage_len // 2] - spline_fit_top(voltage_top))
        convex_difference_bot = PCA_reconstructed_data[i,
                                                       voltage_len // 2::] - spline_fit_bottom(voltage_bottom)

        # Combines and saves the convex hull and difference as a single array
        convex_hull_diff[i] = np.concatenate((convex_difference_top,
                                              convex_difference_bot))
        convex_hull_data[i] = np.concatenate((spline_fit_top(voltage_top),
                                              spline_fit_bottom(voltage_bottom)))

    return (convex_hull_data, convex_hull_diff)

def locate_concavities(convex_hull_diff):

    num_of_pixels = convex_hull_diff.shape[0]

    # Creates an array to store the concavity fitting results
    # [Pixel number, Peak, Intial/final index location]
    values_of_peaks = np.zeros((num_of_pixels, 20, 2))

    # Creates an array to store the number of peaks found
    number_of_peaks = np.zeros(num_of_pixels)

    # Loops around all pixels in the image
    for i in range(num_of_pixels):

        # Preallocates space for the Temp values data
        # This is where the first and last index of each peaks size >2 are saved
        temp_values = np.zeros((20, 2))

        # Counts the number of peaks
        count = 0

        # Running variable which finds the stop point
        stop_point = -1

        # Loops around each index
        for ii in range(convex_hull_diff.shape[1]):

            # Finds the first non-zero point (convexHulldiff[i,ii] > 0)
            # Makes sure that it does not double count those within the next stop point (ii > StopPoint)
            # Checks that the size of the section is at least 2 points and that it is
            # not the last index (convexHulldiff[i,ii+1] > 0 or ii == 96)
            if convex_hull_diff[i, ii] > 0 \
                and ii > stop_point \
                and (convex_hull_diff[i, ii + 1] > 0
                     or ii == convex_hull_diff.shape[1]):

                # Saves the initial + final index of the various peaks in the data
                temp_values[count] = [ii, np.where(convex_hull_diff[i, ii::] == 0)[0][0] + ii - 1]

                # holds the stop point
                stop_point = temp_values[count, 1]

                # iterates count
                count += 1

        # Saves the Number of Peaks
        number_of_peaks[i] = count

        # Values of peaks
        values_of_peaks[i] = temp_values

    return (values_of_peaks, number_of_peaks)

def plot_hystersis_convex_concavitiy(fig, axes_position,
                                     voltage,
                                     convex_hull_data,
                                     PCA_reconstructed_data,
                                     convex_hull_diff,
                                     loop_ylim, loop_x_lim,
                                     scale,
                                     loop_color,
                                     concav_color,
                                     range_selected='none',
                                     values_of_peaks_=[],
                                     data_set='mixed',
                                     color_convex=True):

    # stores the length of the voltage vector
    length = len(voltage)

    # creates the plots
    ax_hys = fig.add_subplot(axes_position[0], axes_position[1],
                             axes_position[2])  # is there a better way
    ax_concave = fig.add_subplot(axes_position[0], axes_position[1], axes_position[2],
                                 sharex=ax_hys, frameon=False)

    # Plots the graphs
    hys_loop = ax_hys.plot(voltage, PCA_reconstructed_data,
                           loop_color)

    # Plots the convex hull if passed
    if convex_hull_data != []:
        conv_loop = ax_hys.plot(voltage, convex_hull_data,
                                'k')

    # creates a vector of the top and bottom branches
    convex_hull_top_branch = -1 * convex_hull_diff.squeeze()[0:length // 2] + scale
    convex_hull_bottom_branch = convex_hull_diff.squeeze()[length // 2::]

    # Plots the concavity curves split to top and bottom.
    ax_concave.plot(voltage[0:length // 2],
                    convex_hull_top_branch,
                    concav_color, linewidth=2)
    ax_concave.plot(voltage[length // 2::],
                    convex_hull_bottom_branch,
                    concav_color, linewidth=2)

    # Checks if the values of the peaks were given
    if values_of_peaks_ != []:

        if data_set == 'mixed':

            # defines the colors
            color_def = {range(11, 23): '#772675',  # Purple
                         range(23, 35): '#772675',  # Purple
                         range(35, 48): '#C71585',  # Pink
                         range(49, 73): '#40B6C4',  # Light Blue
                         range(73, 86): '#2C7FB9',  # Medium Blue
                         range(86, 96): '#293990',  # Dark Blue
                         }

        elif data_set == 'caca':

            color_def = {range(0, 16): '#772675',  # Purple
                         range(16, 32): '#C71585',  # Pink
                         range(33, 41): '#40B6C4',  # Light Blue
                         range(42, 48): '#2C7FB9',  # Medium Blue
                         range(48, 64): '#293990',  # Dark Blue
                         }

        # Loops around all peaks
        for i, range_ in enumerate(values_of_peaks_):

            # excludes concavities that are less than 5 voltage steps
            if range_[1] - range_[0] <= 5:
                continue

            # defines the midpoint
            midpoint = np.floor(range_[0] + (range_[1] - range_[0]) / 2)
            range_ = range_.astype(int)
            range_[1] += 2
            range_[0] -= 1

            # checks the range and defines a color
            for key in color_def:

                if midpoint in key:

                    if color_convex:

                        # Colors the convex hull of the bottom branch
                        if midpoint > length / 2:

                            ax_concave.fill_between(voltage[range(*range_)],
                                                    0,
                                                    convex_hull_diff.squeeze()[range(*range_)],
                                                    facecolor=color_def[key])

                        # Colors the convex hull of the top branch
                        else:

                            ax_concave.fill_between(voltage[range(*range_)],
                                                    scale,
                                                    -1 *
                                                    convex_hull_diff.squeeze()[
                                range(*range_)] + scale,
                                facecolor=color_def[key])

                    # Colors the convex hull of the piezoelectric hysteresis loop
                    if convex_hull_data != []:

                        ax_hys.fill_between(voltage[range(*range_)],
                                            convex_hull_data[range(*range_)],
                                            PCA_reconstructed_data[range(*range_)],
                                            facecolor=color_def[key])

    # formats the axes
    ax_concave.yaxis.tick_right()
    ax_hys.yaxis.tick_left()
    ax_hys.set_xticks(loop_x_lim)
    ax_concave.yaxis.set_label_position("right")
    ax_hys.yaxis.set_label_position('left')
    ax_hys.set_ylabel('Piezorespones (Arb.U.)')
    ax_hys.set_xlabel('Voltage (V)')
    ax_concave.set_ylabel('Concavity (Arb.U.)', rotation=270, labelpad=15)
    ax_hys.yaxis.get_major_formatter().set_powerlimits((0, 1))
    ax_hys.set_ylim(loop_ylim)
    ax_concave.set_yticklabels([])
    ax_hys.set_yticklabels([])

    # Labels the figures
    labelfigs(ax_concave, axes_position[2] - 1)

    # Plots the concavity curves if provided
    if range_selected is not 'none':

        if range_selected[1] <= length / 2:

            ax_concave.plot(voltage[range_selected[0]:range_selected[1]],
                            -1 * convex_hull_diff.squeeze()[range_selected[0]
                                                          :range_selected[1]] + scale,
                            'b', linewidth=4)
        else:
            ax_concave.plot(voltage[range_selected[0]:range_selected[1]],
                            convex_hull_diff.squeeze()[range_selected[0]:range_selected[1]],
                            'b', linewidth=4)

    return ax_concave

    # Function to draw a box on a figure
def path_maker(axes, locations, facecolor, edgecolor, linestyle, lineweight):

    vertices = []
    codes = []
    codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
    vertices = [(locations[0], locations[2]),
                (locations[1], locations[2]),
                (locations[1], locations[3]),
                (locations[0], locations[3]),
                (0, 0)]
    vertices = np.array(vertices, float)
    path = Path(vertices, codes)
    pathpatch = PathPatch(path, facecolor=facecolor, edgecolor=edgecolor,ls=linestyle,lw=lineweight)
    axes.add_patch( pathpatch )

def plot_figs_for_movie(Nd_mat, voltage_reshape, signal_clim, folder,
                        x_range='None', y_range='None'):

    cyc_length = Nd_mat.shape[3]
    cycles = Nd_mat.shape[4]

# Cycles around each switching cycle
    for cycle in range(cycles):

        for i in range(cyc_length):

            # Defines the location of the axes
            fig = plt.figure(figsize=(8, 12))
            ax1 = plt.subplot2grid((3, 2), (0, 0))
            ax2 = plt.subplot2grid((3, 2), (0, 1))
            ax3 = plt.subplot2grid((3, 2), (1, 0))
            ax4 = plt.subplot2grid((3, 2), (1, 1))
            ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)

            axes = (ax1, ax2, ax3, ax4)

            # Sets the format of the axes
            for ax in axes:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks(np.linspace(0, 50, 5))
                ax.set_yticks(np.linspace(0, 50, 5))
                ax.set_facecolor((.55, .55, .55))

            for j, (signals, colorscale) in enumerate(signal_clim.items()):

                (signal_name, signal, formspec) = signals

                if x_range == 'None':
                    x_range_ = slice(0, Nd_mat.shape[0])
                else:
                    x_range_ = slice(x_range[0],x_range[1])

                if y_range == 'None':
                    y_range_ = slice(0, Nd_mat.shape[1])
                else:
                    y_range_ = slice(y_range[0],y_range[1])

                # Plots and formats the graph
                im = axes[j].imshow(Nd_mat[x_range_, y_range_,1,i,cycle][signal])

                axes[j].set_title(signal_name)
                im.set_clim(colorscale)

                # Sets the colorbar
                divider = make_axes_locatable(axes[j])
                cax = divider.append_axes('right', size='10%', pad=0.05)
                cbar = plt.colorbar(im, cax=cax, format=formspec)

                if signal_name == 'Dissipation':
                    add_scalebar_to_figure(axes[j], 1500, 500)

                # Plots the voltage graph
                im5 = ax5.plot(voltage_reshape, 'ok')
                ax5.plot(i + (cycle * cyc_length),
                        voltage_reshape[i + (cycle * cyc_length)], 'rs', markersize=12)
                ax5.set_xlabel('Time Steps')
                ax5.set_ylabel('Voltage (V)')

            # Generates the filename
            filename = 'M{0:02d}_{1:03d}'.format(cycle, i)

            plt.tight_layout()

            # Saves the figure
            fig.savefig(pjoin(folder, filename + '.png'), format='png',
                        dpi=200)

            # Closes the figures
            plt.close(fig)

        # Plots each voltage steps
        #Parallel(n_jobs=11)(delayed(plot_movie_images)(cycle, i, signal_clim, voltage_reshape) for i in range(voltage_steps))

def Interpolate_missing_points(loop_data):
    """
    Interpolates bad pixels in piezoelectric hystereis loops.\n
    The interpolation of missing points alows for machine learning operations

    Parameters
    ----------
    loop_data : numpy array
        arary of loops

    Returns
    -------
    loop_data_cleaned : numpy array
        arary of loops
    """

    # Loops around the x index
    for i in range(loop_data.shape[0]):

        # Loops around the y index
        for j in range(loop_data.shape[1]):

            # Loops around the number of cycles
            for k in range(loop_data.shape[3]):

               if any(~np.isfinite(loop_data[i,j,:,k])):

                    true_ind = np.where(~np.isnan(loop_data[i,j,:,k]))
                    point_values = np.linspace(0,1,loop_data.shape[2])
                    spline = interpolate.InterpolatedUnivariateSpline(point_values[true_ind],
                                                            loop_data[i,j,true_ind,k].squeeze())
                    ind = np.where(np.isnan(loop_data[i,j,:,k]))
                    val = spline(point_values[ind])
                    loop_data[i,j,ind,k] = val

    return loop_data

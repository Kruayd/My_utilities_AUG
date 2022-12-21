#!/usr/bin/python3

# Python code for X-point position reconstruction made by:
# - Luca Cinnirella
#
# and modified/updated by:
#

# Last update: 06.12.2022

# IMPORTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import sys
import warnings
import argparse
import pandas as pd
import numpy as np
import aug_sfutils as sf
import scipy.signal as sg
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from nptyping import NDArray, Number

import sig_proc as sgpr


# GLOBAL VARIABLES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Sampling frequency (in KHz)
F_S_BOL = 500

# DLX geometric parameters
ORGIN_DLX = (1.85859275, -0.92514217)
M_STAR_DLX = 0.0377995
M_REF_DLX = -0.19819659730761274
OFFSET_DLX = 8

# DDC geometric parameters
ORGIN_DDC = (1.41729701, -1.08434093)
M_STAR_DDC = -0.0359498
M_REF_DDC = 3.7782005603072735
OFFSET_DDC = 24


# FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fitting function to apply over axis via numpy
def fit_gauss_time(array: NDArray[Number]) -> Number:
    '''
    Function to be applied along NDArray axis that fits a Gaussian and returns
    its expected value and square rooted variance


    Parameters
    ----------
        array : numpy.ndarray of np.number
            array elements from 0 to 4 included are to be fit
            array elements from 5 to 9 included are weights


    Returns
    -------
        m : int or float or np.number
            Gaussian expected value
        s : int or float or np.number
            Gaussian square rooted variance
    '''
    # array elements from 0 to 4 included are to be fit
    # array elements from 5 to 9 included are weights
    p0 = [array[7], array[2], 1]
    try:
        coeff, _ = opt.curve_fit(gauss, array[:5], array[5:], p0=p0)
        return coeff[1], coeff[2]
    except (ValueError, RuntimeError):
        return np.NaN, np.NaN


# Define model function to be used in case of gaussian fit
def gauss(x: NDArray[Number],
          a: Number, m: Number, s: Number) -> NDArray[Number]:
    '''
    Simple function that returns the values of a Gaussian at given points


    Parameters
    ----------
        x : numpy.ndarray of np.number
            points at which the Gaussian is evaluated

        a : int or float or np.number
            amplitude of the Gaussian

        m : int or float or np.number
            expected value of the Gaussian

        s : int or float or np.number
             square rooted variance of the Gaussian


    Returns
    -------
        y : numpy.ndarray of np.number
            Gaussian values evaluated at x
    '''
    return a * np.exp(-(x - m)**2 / (2. * s**2))


# Real position of XPR
# Check notes!!!
def get_real_position(origin_1: NDArray[Number], m_star_1: Number,
                      m_ref_1: Number, offset_1: int,
                      coord_1: NDArray[Number],
                      origin_2: NDArray[Number], m_star_2: Number,
                      m_ref_2: Number, offset_2: int,
                      coord_2: NDArray[Number]) -> NDArray[Number]:
    '''
    Function that returns the real position of a point given its DLX and DDC
    coordinates


    Parameters
    ----------
    origin_1 : numpy.ndarray of np.number
        real coordinates of the first camera pinhole

    m_star_1 : int or float or np.number
        distance between a diode center and the subsequent one in units of the
        distance between the diode array and the pinhole

    m_ref_1 : int or float or np.number
        tangent of the angle between the normal direction to the diode array
        and the unit vector parallel to the real coordinate R

    offset_1 : int
        line of sight from which we start to count

    coord_1 : numpy.ndarray of np.number
        coordinates in term of the bolometer line of sights

    for parameters with suffix = 2 the descriptions are the same except for
    m_ref_2, where the unit vector parallel to the real coordinate R is
    parallel to Z instead


    Returns
    -------
        x : numpy.ndarray of np.number
            array of R components of converted coordinates
        y : numpy.ndarray of np.number
            array of Z components of converted coordinates
    '''
    m_p_1 = (-2*(coord_1 - offset_1) + 1) * m_star_1
    m_p_2 = (-2*(coord_2 - offset_2) + 1) * m_star_2
    m_1 = (m_p_1 + m_ref_1)/(1 - m_p_1*m_ref_1)
    m_2 = (m_p_2 + m_ref_2)/(1 - m_p_2*m_ref_2)
    c_1 = origin_1[1] - m_1 * origin_1[0]
    c_2 = origin_2[1] - m_2 * origin_2[0]
    d = m_1 - m_2
    x = (c_2 - c_1)/d
    y = (c_2 * m_1 - c_1 * m_2)/d
    return x, y


# Find peaks function to apply over dlx and ddc axis
# through numpy
def find_peaks_2d(array: NDArray[Number], **kwargs) -> Number:
    '''
    Function to be applied along NDArray axis that returns the index of the
    first peak to be found and 0 or np.NaN whether anything was found


    Parameters
    ----------
        array : numpy.ndarray of np.number
            array in which peaks are to be seeked

        kwargs : kwargs
            kwargs to be passed to scipy.signal.find_peaks


    Returns
    -------
        x : int
            index of the peak
    '''
    try:
        peaks, _ = sg.find_peaks(array, **kwargs)
        return peaks[0], 0
    except IndexError:
        return 0, np.NaN


# OPTIONS HANDLER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
parser = argparse.ArgumentParser(description='Plot XPR position ' +
                                 'reconstruction and, if avialable, ' +
                                 'temperature at the X-point')
parser.add_argument('shot', metavar='SHOT_NUMBER', type=int,
                    help='Execute the code for shot #SHOT_NUMBER')
parser.add_argument('-w', '--time_window', metavar='TIME_WINDOW', type=float,
                    default=50.,
                    help='Set the time window (in ms) length on which to ' +
                    'apply median filter.\nThe step size of the median ' +
                    'filter is also set to 1/3 of the given window length. ' +
                    'In this way  a decimation is also applied together ' +
                    'with the filter (default is 50)')
parser.add_argument('-m', '--mag_equ_diag', type=str, default='EQH',
                    choices=['FPP', 'EQI', 'EQH'],
                    help='Select which diagnostic is used for magnetic ' +
                    'reconstruction (FPP, EQI or EQH, default is EQH)')
parser.add_argument('-d', '--depth', metavar='CONTOUR_DEPTH', type=int,
                    default=50,
                    help='Set matplotlib contourf depth to CONTOUR_DEPTH ' +
                    '(default is 50)')
parser.add_argument('-f', '--fps', metavar='FRAMES_PER_SECOND', type=float,
                    default=30,
                    help='Set matplotlib animation fps (default is 30)')
parser.add_argument('-o', '--output_to_csv', action='store_true',
                    default=False,
                    help='Output a csv file containing data about the ' +
                    'x-point radiator position, the time base and, finally, ' +
                    'another variable about which peak detection method was ' +
                    'used.\nIf the gaussian fit was used, "_gaussian" is ' +
                    'prepended before the file extension')
args = parser.parse_args()

if args.mag_equ_diag == 'FPP':
    FPG_DIAG = 'FPG'
elif args.mag_equ_diag == 'EQI':
    FPG_DIAG = 'GQI'
elif args.mag_equ_diag == 'EQH':
    FPG_DIAG = 'GQH'

shot = args.shot
time_window = args.time_window
mag_equ_diag = args.mag_equ_diag
f_interval = int(1000/args.fps)


# READING SHOT TIME BOUNDARIES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
df_shot_time = pd.read_csv('./csvs/equilibria.csv', index_col=0).loc[shot]

shot_start = df_shot_time['start']
shot_end = df_shot_time['end']


# QUERYING SHOTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get shot-files relative to Divertor Thompson scattering, DLX and DDC
# bolometers, calibration of DLX and DDC bolometers, equilibrium and wall
# components contour
dtn = sf.SFREAD(shot, 'DTN')
blc = sf.SFREAD(sf.previousshot('BLC', shot),
                'BLC')
xvt = sf.SFREAD(shot, 'XVT')
xvs = sf.SFREAD(shot, 'XVS')
equ = sf.EQU(shot, diag=mag_equ_diag)
gc_d = sf.getgc()


# QUERYING SIGNALS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get calibrated time traces for electron temperature and density of DTN and
# for radiation detected by DLX and DDC bolometers (respectively: first 12 line
# of sights and from 17 to 32)

# Electron temperature and density array
# We transpose the data from Te_ld and Ne_ld so that we can easily select the
# ith-core with Te_ld[i-1] and Ne_ld[i-1]
if dtn.status:
    Te_ld = dtn.getobject('Te_ld', cal=True)
    Te_ld = Te_ld.transpose()
    time_Te_ld = dtn.gettimebase('Te_ld')

    Ne_ld = dtn.getobject('Ne_ld', cal=True)
    Ne_ld = Ne_ld.transpose()
    time_Ne_ld = dtn.gettimebase('Ne_ld')
else:
    warnings.warn('Error while loading DTN')
    time_Te_ld = np.arange(shot_start, shot_end, 0.001)
    Te_ld = np.zeros((26, time_Te_ld.size))
    time_Ne_ld = np.arange(shot_start, shot_end, 0.001)
    Ne_ld = np.zeros((26, time_Te_ld.size))

# Calibration parameters set (dictionary)
if blc.status:
    dlx_par = blc.getparset('DLX')
    ddc_par = blc.getparset('DDC')
    dhc_par = blc.getparset('DHC')
else:
    sys.exit('Error while loading BLC')

# DLX bolometer arrays
# The data array must contain all the possible line of sight and must be
# selected as the electron temperature data array
if xvt.status:
    dlx_arr = list()
    for i in range(0, 12):
        signal = 'S4L0A'+str(i).zfill(2)
        dlx_arr.append(xvt.getobject(signal, cal=True))
    dlx = np.array(dlx_arr)*100
    time_dlx = xvt.gettimebase('S4L0A00')
else:
    sys.exit('Error while laoding XVT')

# DDC bolometer arrays
# The data array must contain all the possible line of sight and must be
# selected as the electron temperature data array
if xvs.status:
    ddc_arr = list()
    for i in range(0, 16):
        signal = 'S2L1A'+str(i).zfill(2)
        ddc_arr.append(xvs.getobject(signal, cal=True))
    ddc = np.array(ddc_arr)
    time_ddc = xvs.gettimebase('S2L1A00')
else:
    sys.exit('Error while laoding XVS')

print('Querying: done')


# SIGNAL PRE-PROCESSING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DLX bolometer
# Select only active line of sights
dlx_sights = np.arange(1, 13) * dlx_par['active'][:12]
dlx_sights = dlx_sights[dlx_sights != 0]
dlx = dlx[dlx_sights - 1]

# Remove offsets
dlx = (dlx.transpose() - np.mean(dlx[:, :10000], 1)).transpose()

# DDC bolometer
# Select only active line of sights
ddc_sights = np.arange(17, 33) * ddc_par['active'][16:32]
ddc_sights = ddc_sights[ddc_sights != 0]
ddc = ddc[ddc_sights - 17]

# Remove offsets
ddc = (ddc.transpose() - np.mean(ddc[:, :10000], 1)).transpose()

print('Pre-processing: done')


# UNIVERSAL TIME SELECTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get maximum common time interval boundaries
start = max(time_Te_ld[0], time_Ne_ld[0], time_dlx[0], time_ddc[0], shot_start)
end = min(time_Te_ld[-1], time_Ne_ld[-1], time_dlx[-1], time_ddc[-1], shot_end)

# Get indexes of boundaries for ddc
start_index_ddc = sgpr.find_nearest_index(time_ddc, start)
end_index_ddc = sgpr.find_nearest_index(time_ddc, end)

# Get indexes of boundaries for dlx
start_index_dlx = sgpr.find_nearest_index(time_dlx, start)
end_index_dlx = sgpr.find_nearest_index(time_dlx, end)

# Get indexes of boundaries for Te_ld
start_index_Te_ld = sgpr.find_nearest_index(time_Te_ld, start)
end_index_Te_ld = sgpr.find_nearest_index(time_Te_ld, end)

# Get indexes of boundaries for Ne_ld
start_index_Ne_ld = sgpr.find_nearest_index(time_Ne_ld, start)
end_index_Ne_ld = sgpr.find_nearest_index(time_Ne_ld, end)

# Slice ddc and relative time
ddc = ddc[:, start_index_ddc:end_index_ddc+1]
time_ddc = time_ddc[start_index_ddc:end_index_ddc+1]

# Slice dlx and relative time
dlx = dlx[:, start_index_dlx:end_index_dlx+1]
time_dlx = time_dlx[start_index_dlx:end_index_dlx+1]

# Slice Te_ld and relative time
Te_ld = Te_ld[:, start_index_Te_ld:end_index_Te_ld+1]
time_Te_ld = time_Te_ld[start_index_Te_ld:end_index_Te_ld+1]

# Slice Ne_ld and relative time
Ne_ld = Ne_ld[:, start_index_Ne_ld:end_index_Ne_ld+1]
time_Ne_ld = time_Ne_ld[start_index_Ne_ld:end_index_Ne_ld+1]

print('Time selection: done')


# SLICING AND PROCESSING FOR DIFFERENT OPERATIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Apply median median_filter to dlx
time_dlx_flt, dlx_flt = sgpr.median_filter(F_S_BOL, time_window, time_dlx, dlx)

# Apply median median_filter to ddc
time_ddc_flt, ddc_flt = sgpr.median_filter(F_S_BOL, time_window, time_ddc, ddc)

# Sometimes time_dlx_flt and time_ddc_flt can be different in length, leading
# to errors in subsequent computations. It's better to re-slice them in order
# to let them have same array.shape[-1]
length_index = min(time_dlx_flt.size, time_ddc_flt.size)
dlx_flt = dlx_flt[:, :length_index]
time_dlx_flt = time_dlx_flt[:length_index]
ddc_flt = ddc_flt[:, :length_index]
time_ddc_flt = time_ddc_flt[:length_index]

# Remove first line of sight from DLX bolometer (usually it is just  a
# reflection of the actual radiator) and bring everityhing above 0 (necessary
# for weighted average)
dlx_index_rd = (dlx_sights != 1)
dlx_flt_rd = dlx_flt[dlx_index_rd] - dlx_flt[dlx_index_rd].min(axis=0)
dlx_sights_rd = dlx_sights[dlx_index_rd]

# Remove sights 17, 18, 31, 32 from DDC bolometer (usually they are just
# radiation from the separatrix) and bring everityhing above 0 (necessary for
# weighted average)
ddc_index_rd = ((ddc_sights != 32) & (ddc_sights != 31)
                & (ddc_sights != 18) & (ddc_sights != 17))
ddc_flt_rd = ddc_flt[ddc_index_rd]
ddc_sights_rd = ddc_sights[ddc_index_rd]
ddc_flt_rd -= ddc_flt_rd.min(axis=0)

# The XPR position is firstly approximated by queryng for the index of the
# peaks in the bolometer signals. For what concers DDC, is's better to
# distinguish between positions found with argmax() or find_peaks
xpr_dlx_pk_idx, dlx_pk = np.apply_along_axis(find_peaks_2d, 0, dlx_flt_rd,
                                             distance=dlx_flt_rd.shape[0]*0.66,
                                             height=dlx_flt_rd.max()/4)
xpr_dlx_pk_idx = xpr_dlx_pk_idx.astype(int)

xpr_ddc_pk_idx, ddc_pk = np.apply_along_axis(find_peaks_2d, 0, ddc_flt_rd,
                                             distance=ddc_flt_rd.shape[0]*0.66)
xpr_ddc_pk_idx = xpr_ddc_pk_idx.astype(int)

# In order to get a better position we can try to evaluate the weighted average
# (deterministic) or do a gaussian fit (non deterministic) but, to do so we
# need our coordinate arrays (dlx/ddc_sights_rd) to have the same dimension
# of the signal arrays. Therefore, they need to be extended through time:
dlx_sights_rd_matrix = np.tile(dlx_sights_rd,
                               (time_dlx_flt.size, 1)).transpose()
ddc_sights_rd_matrix = np.tile(ddc_sights_rd,
                               (time_ddc_flt.size, 1)).transpose()

# The weighted avarage (or the fitting) will be done on 5 points centered
# around the maximum. Hence, to prevent any slicing error, we extend the
# coordinates and the signal arrays by two rows to the top and to the bottom.
# The extensions are generated in such a manner that they will be ignored both
# by the weighted average and the gaussian fit DLX extension
# DLX extension
dlx_sights_rd_matrix = np.concatenate((dlx_sights_rd_matrix[0:2] - 2,
                                       dlx_sights_rd_matrix,
                                       dlx_sights_rd_matrix[-2:] + 2))

dlx_flt_rd_extension = np.zeros((2, time_dlx_flt.size))
dlx_flt_rd = np.concatenate((dlx_flt_rd_extension,
                             dlx_flt_rd,
                             dlx_flt_rd_extension))

# DDC extension
ddc_sights_rd_matrix = np.concatenate((ddc_sights_rd_matrix[0:2] - 2,
                                       ddc_sights_rd_matrix,
                                       ddc_sights_rd_matrix[-2:] + 2))

ddc_flt_rd_extension = np.zeros((2, time_ddc_flt.size))
ddc_flt_rd = np.concatenate((ddc_flt_rd_extension,
                             ddc_flt_rd,
                             ddc_flt_rd_extension))

# We can finally do some advanced numpy slicing in order to get the desired 5
# points arrays.
# Just remember that we have add 2 more rows at the beginning of the arrays so
# every index should be shifted up of 2 positions
# (xpr_dlx_pk_idx -2 -----> xpr_dlx_pk_idx
#  xpr_dlx_pk_idx -1 -----> xpr_dlx_pk_idx +1
#  and so on...)
# DLX slicing
dlx_flt_rd_row_slicer = np.array([xpr_dlx_pk_idx,
                                  xpr_dlx_pk_idx + 1,
                                  xpr_dlx_pk_idx + 2,
                                  xpr_dlx_pk_idx + 3,
                                  xpr_dlx_pk_idx + 4])
dlx_flt_rd_column_slicer = np.arange(0, time_dlx_flt.size)
dlx_flt_rd = dlx_flt_rd[dlx_flt_rd_row_slicer,
                        dlx_flt_rd_column_slicer]
dlx_sights_rd_matrix = dlx_sights_rd_matrix[dlx_flt_rd_row_slicer,
                                            dlx_flt_rd_column_slicer]
# set bad peaks to NaN
dlx_flt_rd += dlx_pk

# DDC slicing
ddc_flt_rd_row_slicer = np.array([xpr_ddc_pk_idx,
                                  xpr_ddc_pk_idx + 1,
                                  xpr_ddc_pk_idx + 2,
                                  xpr_ddc_pk_idx + 3,
                                  xpr_ddc_pk_idx + 4])
ddc_flt_rd_column_slicer = np.arange(0, time_ddc_flt.size)
ddc_flt_rd = ddc_flt_rd[ddc_flt_rd_row_slicer,
                        ddc_flt_rd_column_slicer]
ddc_sights_rd_matrix = ddc_sights_rd_matrix[ddc_flt_rd_row_slicer,
                                            ddc_flt_rd_column_slicer]
# set bad peaks to NaN
ddc_flt_rd += ddc_pk

# Fit
print('Fitting')
par_dlx = np.apply_along_axis(fit_gauss_time, 0,
                              np.concatenate((dlx_sights_rd_matrix,
                                              dlx_flt_rd)))
xpr_dlx = par_dlx[0]
xpr_dlx_lo = par_dlx[0] - par_dlx[1]
xpr_dlx_up = par_dlx[0] + par_dlx[1]

par_ddc = np.apply_along_axis(fit_gauss_time, 0,
                              np.concatenate((ddc_sights_rd_matrix,
                                              ddc_flt_rd)))
xpr_ddc = par_ddc[0]

# Convert from bolometers coordinates to real coordinates
xpr_x, xpr_y = get_real_position(ORGIN_DLX, M_STAR_DLX, M_REF_DLX, OFFSET_DLX,
                                 xpr_dlx, ORGIN_DDC, M_STAR_DDC, M_REF_DDC,
                                 OFFSET_DDC, xpr_ddc)
_, xpr_y_lo = get_real_position(ORGIN_DLX, M_STAR_DLX, M_REF_DLX, OFFSET_DLX,
                                xpr_dlx_lo, ORGIN_DDC, M_STAR_DDC, M_REF_DDC,
                                OFFSET_DDC, xpr_ddc)
_, xpr_y_up = get_real_position(ORGIN_DLX, M_STAR_DLX, M_REF_DLX, OFFSET_DLX,
                                xpr_dlx_up, ORGIN_DDC, M_STAR_DDC, M_REF_DDC,
                                OFFSET_DDC, xpr_ddc)

nan_mask = np.isnan(xpr_x)
xpr_rho = list()
xpr_rho_up = list()
xpr_rho_lo = list()
for index, time in enumerate(time_dlx_flt):
    xpr_rho.append(sf.rz2rho(equ, xpr_x[index], xpr_y[index], t_in=time,
                             coord_out='rho_pol').flatten()[0])
    xpr_rho_up.append(sf.rz2rho(equ, xpr_x[index], xpr_y_lo[index],
                                t_in=time, coord_out='rho_pol').flatten()[0])
    xpr_rho_lo.append(sf.rz2rho(equ, xpr_x[index], xpr_y_up[index],
                                t_in=time, coord_out='rho_pol').flatten()[0])
xpr_rho = np.array(xpr_rho)
xpr_rho[nan_mask] = np.NaN

xpr_rho_up = np.array(xpr_rho_up)
xpr_rho_up[nan_mask] = np.NaN

xpr_rho_lo = np.array(xpr_rho_lo)
xpr_rho_lo[nan_mask] = np.NaN

print('Further processing: done')


# PRINTING FILE (OPTIONAL)
if args.output_to_csv:
    FILE_NAME = './csvs/XPR_position/' + str(shot) + '_XPR_position.csv'
    DF_DATA = np.r_['0,2',
                    time_dlx_flt,
                    xpr_x,
                    xpr_y,
                    xpr_y_lo,
                    xpr_y_up,
                    xpr_rho,
                    xpr_rho_lo,
                    xpr_rho_up].T
    df = pd.DataFrame(DF_DATA, columns=['t (s)',
                                        'R (m)',
                                        'z (m)',
                                        'z lower limit (m)',
                                        'z upper limit (m)',
                                        'rho pol',
                                        'rho pol lower limit',
                                        'rho pol upper limit'])
    df.to_csv(FILE_NAME, index=False, na_rep='NaN')


# PLOTTING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# General settings
plt.style.use('bmh')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
px = 1/plt.rcParams['figure.dpi']  # from pixel to inches
fig = plt.figure(figsize=(1600*px, 1000*px))
plt.suptitle(f'SHOT #{shot}', fontsize=32, fontweight='bold')
frames = time_dlx_flt.size

# Subplots
ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=1, colspan=2)
axtext = plt.subplot2grid((3, 3), (0, 2), rowspan=1, colspan=1)
ax2 = plt.subplot2grid((3, 3), (1, 2), rowspan=2, colspan=1, aspect='equal')
ax3 = plt.subplot2grid((3, 3), (1, 0), rowspan=1, colspan=1)
ax4 = plt.subplot2grid((3, 3), (1, 1), rowspan=1, colspan=1)
ax5 = plt.subplot2grid((3, 3), (2, 0), rowspan=1, colspan=1)
ax6 = plt.subplot2grid((3, 3), (2, 1), rowspan=1, colspan=1)

# Te_ld subplot
ax1.plot(time_Te_ld, Te_ld[9], label='10th core')
ax1.plot(time_Te_ld, Te_ld[10], label='11th core')
ax1.plot(time_Te_ld, Te_ld[11], label='12th core')
ax1.set_title('DTS electron temperature nearby X-point')
ax1.set_xlabel('s')
ax1.set_ylabel('eV')
ax1.legend()

# X-point radiator position subplot
# TO ANIMATE BEGIN
sep_image, = ax2.plot([], [], '-', color=colors[1])
xpr_image, = ax2.plot([], [], 'o', color=colors[0])
r_sep, z_sep = sf.rho2rz(equ, 1, t_in=time_dlx_flt, coord_in='rho_pol')
# TO ANIMATE END
# Bolometers drawing
dlx_start_R = dlx_par['R_Blende'][dlx_sights - 1]
dlx_start_Z = dlx_par['z_Blende'][dlx_sights - 1]
dlx_end_R = dlx_par['R_end'][dlx_sights - 1]
dlx_end_Z = dlx_par['z_end'][dlx_sights - 1]
ddc_start_R = ddc_par['R_Blende'][ddc_sights - 1]
ddc_start_Z = ddc_par['z_Blende'][ddc_sights - 1]
ddc_end_R = ddc_par['R_end'][ddc_sights - 1]
ddc_end_Z = ddc_par['z_end'][ddc_sights - 1]
for i in range(0, dlx_start_R.size):
    ax2.plot([dlx_start_R[i], dlx_end_R[i]], [dlx_start_Z[i], dlx_end_Z[i]],
             '-', color='gray')
for i in range(0, ddc_start_R.size):
    ax2.plot([ddc_start_R[i], ddc_end_R[i]], [ddc_start_Z[i], ddc_end_Z[i]],
             '-', color='gray')
# Vessel drawing
for gc in gc_d.values():
    ax2.plot(gc.r, gc.z, '-', color='gray')
# Settings
ax2.set_title('X-point radiator position')
ax2.set_xlabel('R')
ax2.set_ylabel('z')
ax2.set_xlim(1.1, 1.8)
ax2.set_ylim(-1.3, -0.6)

# DLX subplot
cont_1 = ax3.contourf(time_dlx_flt, dlx_sights, dlx_flt, args.depth,
                      cmap='inferno')
ax3.plot(time_dlx_flt, xpr_dlx)
ax3.set_title('DLX detected radiation')
ax3.set_xlabel('s')
ax3.set_ylabel('sight')
ax3.set_ylim(dlx_sights.min(), dlx_sights.max())
fig.colorbar(cont_1, ax=ax3)

# DDC subplot
cont_2 = ax4.contourf(time_ddc_flt, ddc_sights, ddc_flt, args.depth,
                      cmap='inferno')
ax4.plot(time_ddc_flt, xpr_ddc)
ax4.set_title('DDC detected radiation')
ax4.set_xlabel('s')
ax4.set_ylabel('sight')
ax4.set_ylim(ddc_sights.min(), ddc_sights.max())
fig.colorbar(cont_2, ax=ax4)

# Evolution of DLX signal subplot
# TO ANIMATE BEGIN
dlx_image, = ax5.plot([], [])
dlx_peak_image = ax5.axvline(color=colors[1])
# TO ANIMATE END
ax5.set_title('DLX signal evolution')
ax5.set_xlabel('sight')
ax5.set_ylabel('W $m^{-2}$')
ax5.set_xlim(dlx_sights[0]-0.5, dlx_sights[-1]+0.5)
ax5.set_ylim(dlx_flt.min(), dlx_flt.max())

# Evolution of DDC signal subplot
# TO ANIMATE BEGIN
ddc_image, = ax6.plot([], [])
ddc_peak_image = ax6.axvline(color=colors[1])
# TO ANIMATE END
ax6.set_title('DDC signal evolution')
ax6.set_xlabel('sight')
ax6.set_ylabel('W $m^{-2}$')
ax6.set_xlim(ddc_sights[0]-0.5, ddc_sights[-1]+0.5)
ax6.set_ylim(ddc_flt.min(), ddc_flt.max())

# Timestamp subplot
# TO ANIMATE BEGIN
timestamp = axtext.text(0.5, 0.5, f'Time:  {0:.3f} s', fontsize=32,
                        fontweight='bold', ha='center', va='center',
                        transform=axtext.transAxes)
# TO ANIMATE END
axtext.set_facecolor('white')
axtext.get_xaxis().set_visible(False)
axtext.get_yaxis().set_visible(False)


def update_ani(frame):
    sep_image.set_data(r_sep[frame][0], z_sep[frame][0])
    xpr_image.set_data(xpr_x[..., frame], xpr_y[..., frame])
    xpr_image.set_color(colors[0])
    dlx_image.set_data(dlx_sights, dlx_flt[..., frame])
    dlx_peak_image.set_xdata(xpr_dlx[frame])
    ddc_image.set_data(ddc_sights, ddc_flt[..., frame])
    ddc_peak_image.set_xdata(xpr_ddc[frame])
    timestamp.set_text(f'Time:  {time_dlx_flt[frame]:.3f} s')
    return sep_image, xpr_image, dlx_image, dlx_peak_image, ddc_image, \
        ddc_peak_image, timestamp


ani = FuncAnimation(fig, update_ani, frames=frames, interval=f_interval,
                    blit=True)


# WID = 5.512
# HIG = WID / 1.618
# X_LABEL = r'Time (s)'
# Y_LABEL = r'DLX line of sight'
# C_LABEL = r'Radiation flux (W m$^{-2}$)'


# plt.style.use('bmh')
# plt.figure(figsize=(WID, HIG))
# plt.rc('font', size=10)
# plt.rc('axes', titlesize=10)
# plt.rc('axes', labelsize=12)
# plt.rc('xtick', labelsize=10)
# plt.rc('ytick', labelsize=10)
# plt.rc('legend', fontsize=10)


# plt.title(f'Shot #{shot}', loc='right')
# plt.contourf(time_dlx_flt, dlx_sights, dlx_flt, args.depth, cmap='inferno')
# plt.plot(time_dlx_flt, xpr_dlx)
# plt.xlabel(X_LABEL)
# plt.ylabel(Y_LABEL)
# plt.colorbar(label=C_LABEL)


# Show plot or save
plt.tight_layout(pad=0.1)
plt.show()

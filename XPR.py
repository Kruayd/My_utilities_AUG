#!/usr/bin/python3

# Python code for X-point radiator analysis made by:
# - Luca Cinnirella
#
# and modified/updated by:
#

# Last update: 09.03.2022

# IMPORTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import sys
from optparse import OptionParser, OptionGroup
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import style
import numpy as np
import scipy.signal as sg
import scipy.optimize as opt
import pandas as pd
import aug_sfutils as sf



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
# Median filter
def median_filter(f_s_bol, time_window, time_bol, bol):
    window = int(f_s_bol * time_window)
    window += (window+1) % 2                            # to next odd
    step_size = int(np.ceil(window/3))                  # at least 1/3
    # indexes = []

    # OLD CODE
    # Create a indexes array for multiple slicing.
    # To know how many samples we can get out of the initial array we have to
    # look at how many last points of a window we can fit in it.
    # The range(window-1, time_bol.size, step_size) takes care of it and,
    # togheter with the for cycle, it returns the index (j) of the last element
    # of the selected window. The array of indexes can be easily constructed as
    # np.arange(j+1-window,j+1).
    # If we define i=j+1 then the range shall be modified as
    # range(window, time_bol.size + 1, step_size).
    #
    # for i in range(window, time_bol.size + 1, step_size):
    #     indexes.append(np.arange(i-window,i))
    # indexes = np.array(indexes)

    # FASTER AND BETTER NEW CODE
    # In order to properly slice the bolometer signal, as it will be shown in
    # the next paragraph, we need an indexes array built like this:
    # indexes = [[first        window]
    #            [second       window]
    #                     ...
    #            [quotient+1th window]]
    # quotient is the number of window end points that fit within the given
    # time array minus 1. It is simply evaluated by subtracting window-1 (since
    # we are counting end points we want to start from the end point of the
    # first window, though this won't be considered in anyway by the integer
    # division operation) form the size of time_bol and, then, by integer
    # dividing step_size.
    # When tiling, we want the resulting array to contain all the windows that
    # can fit within time_bol, therefore we need to add just the first window
    # (quotient + 1) which was excluded by the integer division.
    # At this poit (np.tile + transpose) our indexes array looks like this:
    # indexes = [[-window -window+1 ... -1]
    #            [-window -window+1 ... -1]
    #                               ...
    #            [-window -window+1 ... -1]]
    # To differentiate bitween windows we just add the value of window + order
    # of window*step_size
    quotient = (time_bol.size + 1 - window) // step_size
    indexes = (np.tile(np.arange(-window, 0), (quotient+1, 1)).transpose() +
               (window+np.arange(0, quotient+1)*step_size)).transpose()

    # Apply median operation on the sliced bol array
    # The bol[:,indexes] operation returns a 3D array with the size on the
    # second axis equals to window. By applying the median operation we get
    # back to a 2D array with zeroth axis lenght equals to bol.shape[0] and the
    # first axis decimated.
    # The new time array is obtained by taking from each row of the indexes
    # array the middle point and then slicing time_bol
    bol_filt = np.median(bol[:, indexes], axis=2)
    time_bol_filt = time_bol[indexes[:, int((window-1)/2)]]
    return time_bol_filt, bol_filt

# Find nearest index
def find_nearest_index(array, value, axis=None):
    return np.absolute(array-value).argmin(axis=axis)

# Re-slice
def re_slice(time_array, array, start, end):
    start_index = find_nearest_index(time_array, start)
    end_index = find_nearest_index(time_array, end)

    return time_array[start_index:end_index+1], array[:, start_index:end_index+1]

# Fitting function to apply over axis via numpy
def fit_gauss_time(array):
    # array elements from 0 to 4 included are to be fit
    # array elements from 5 to 9 included are weights
    p0 = [array[7], array[2], 1]
    try:
        coeff, _ = opt.curve_fit(gauss, array[:5], array[5:], p0=p0)
        return coeff[1]
    except:
        return np.average(array[:5], weights=array[5:])


# Define model function to be used in case of gaussian fit
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

# Real position of XPR
# Check notes!!!
def get_real_position(origin_1, m_star_1, m_ref_1, offset_1, coord_1,
                      origin_2, m_star_2, m_ref_2, offset_2, coord_2):
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
def find_peaks_2d(array, **kwargs):
    try:
        peaks, _ = sg.find_peaks(array, **kwargs)
        return peaks[0]
    except:
        return array.argmax()



# OPTIONS HANDLER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
parser = OptionParser(usage='Usage: %prog [mandatory] args [options] args',
                      add_help_option=False)
parser.add_option('-h', '--help',
                  action='help',
                  help='Show this help message')
parser.add_option('-w', '--window',
                  metavar='TIME_WINDOW',
                  action='store', type='float', dest='time_window',
                  default=50.,
                  help='Set the time window (in ms) length on which to ' +
                       'apply median filter.\nThe step size of the median ' +
                       'filter is also set to 1/3 of the given window ' +
                       'length. In this way a decimation is also applied ' +
                       'together with the filter (default is 50)')
parser.add_option('-e', '--equilibrium_diagnostic',
                  metavar='EQUILIBRIUM_DIAGNOSTIC',
                  action='store', type='str', dest='equ_diag', default='EQH',
                  help='Select which diagnostic is used for magnetic '+
                       'reconstruction (FPP, EQI or EQH, default is EQH)')
parser.add_option('-t', '--end_time',
                  metavar='END_time',
                  action='store', type='float', dest='end_time', default=12.,
                  help='Select the upper boundary of time interval to plot')
parser.add_option('-d', '--depth',
                  metavar='CONTOUR_DEPTH',
                  action='store', type='int', dest='depth', default=50,
                  help='Set matplotlib contourf depth to CONTOUR_DEPTH ' +
                       '(default is 50)')
parser.add_option('-f', '--frames-per-second',
                  metavar='FRAMES_PER_SECOND',
                  action='store', type='float', dest='fps', default=30,
                  help='Set matplotlib animation fps (default is 30)')
parser.add_option('-l', '--linear_detrend',
                  action='store_true', dest='detrend', default=False,
                  help='Use linear detrend from scipy in order to remove ' +
                       'the baseline radiation from DDC bolometer ' +
                       '(non-deterministic)\nDEPRECATED!!!')
parser.add_option('-g', '--gaussian_fit',
                  action='store_true', dest='gaussian', default=False,
                  help='Use gaussian fit from scipy in order to get the ' +
                       'non-discrete X-point radiator position ' +
                       '(non-deterministic)')
parser.add_option('-p', '--print_to_csv',
                  metavar='XPR_START_TIME',
                  action='store', type='float', dest='xpr_start_time',
                  help='Output a XPR_position.csv file containing data ' +
                       'about the x-point radiator position, the time base ' +
                       'and a variable that tells whether or not the xpr is ' +
                       'already established')
mandatory = OptionGroup(parser,
                        'Mandatory args',
                        'These args are mandatory!!!')
mandatory.add_option('-s', '--shot',
                     metavar='SHOT_NUMBER',
                     action='store', type='int', dest='shot',
                     help='Execute the code for shot #SHOT_NUMBER')
parser.add_option_group(mandatory)
(options, args) = parser.parse_args()
if not(options.shot):
    parser.print_help()
    sys.exit('No shot was provided')
if options.equ_diag == 'FPP':
    fpg_diag = 'FPG'
elif options.equ_diag == 'EQI':
    fpg_diag = 'GQI'
elif options.equ_diag == 'EQH':
    fpg_diag = 'GQH'
else:
    parser.print_help()
    sys.exit('Please, use one of the standard diagnostics')

shot = options.shot
time_window = options.time_window
equ_diag = options.equ_diag
end_time = options.end_time
f_interval = int(1000/options.fps)
detrend = options.detrend
gaussian = options.gaussian



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
equ = sf.EQU(shot, diag=equ_diag)
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
    sys.exit('Error while loading DTN')

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
    dlx_arr = []
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
    ddc_arr = []
    for i in range(0, 16):
        signal = 'S2L1A'+str(i).zfill(2)
        ddc_arr.append(xvs.getobject(signal, cal=True))
    ddc = np.array(ddc_arr)
    time_ddc = xvs.gettimebase('S2L1A00')
else:
    sys.exit('Error while laoding XVS')

print('Querying: done')



# SIGNAL PROCESSING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DLX bolometer
# Select only active line of sights
dlx_sights = np.arange(1, 13) * dlx_par['active'][:12]
dlx_sights = dlx_sights[dlx_sights != 0]
dlx = dlx[dlx_sights - 1]

# Remove offsets
dlx = (dlx.transpose() - np.mean(dlx[:, :10000], 1)).transpose()

# Apply median median_filter
time_dlx_filt, dlx_filt = median_filter(F_S_BOL, time_window, time_dlx, dlx)

# DDC bolometer
# Select only active line of sights
ddc_sights = np.arange(17, 33) * ddc_par['active'][16:32]
ddc_sights = ddc_sights[ddc_sights != 0]
ddc = ddc[ddc_sights - 17]

# Remove offsets
ddc = (ddc.transpose() - np.mean(ddc[:, :10000], 1)).transpose()

# Apply median median_filter
time_ddc_filt, ddc_filt = median_filter(F_S_BOL, time_window, time_ddc, ddc)

print('Processing: done')



# UNIVERSAL TIME SELECTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get maximum common time interval boundaries
start = max(time_Te_ld[0], time_Ne_ld[0], time_dlx_filt[0], time_ddc_filt[0])
end = min(end_time, time_Te_ld[-1], time_Ne_ld[-1], time_dlx_filt[-1], time_ddc_filt[-1])

# Get indexes of boundaries for ddc_filt
start_index_ddc = find_nearest_index(time_ddc_filt, start)
end_index_ddc = find_nearest_index(time_ddc_filt, end)

# Get indexes of boundaries for dlx_filt
start_index_dlx = find_nearest_index(time_dlx_filt, start)
end_index_dlx = find_nearest_index(time_dlx_filt, end)

# Get indexes of boundaries for Te_ld
start_index_Te_ld = find_nearest_index(time_Te_ld, start)
end_index_Te_ld = find_nearest_index(time_Te_ld, end)

# Get indexes of boundaries for Ne_ld
start_index_Ne_ld = find_nearest_index(time_Ne_ld, start)
end_index_Ne_ld = find_nearest_index(time_Ne_ld, end)

# Slice ddc_filt and relative time
ddc_filt = ddc_filt[:, start_index_ddc:end_index_ddc+1]
time_ddc_filt = time_ddc_filt[start_index_ddc:end_index_ddc+1]

# Slice dlx_filt and relative time
dlx_filt = dlx_filt[:, start_index_dlx:end_index_dlx+1]
time_dlx_filt = time_dlx_filt[start_index_dlx:end_index_dlx+1]

# Slice Te_ld and relative time
Te_ld = Te_ld[:, start_index_Te_ld:end_index_Te_ld+1]
time_Te_ld = time_Te_ld[start_index_Te_ld:end_index_Te_ld+1]

# Slice Ne_ld and relative time
Ne_ld = Ne_ld[:, start_index_Ne_ld:end_index_Ne_ld+1]
time_Ne_ld = time_Ne_ld[start_index_Ne_ld:end_index_Ne_ld+1]

print('Time selection: done')



# SLICING AND PROCESSING FOR DIFFERENT OPERATIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Remove first line of sight from DLX bolometer (usually it is just  a
# reflection of the actual radiator) and bring everityhing above 0 (necessary
# for weighted average)
dlx_pos_keep = (dlx_sights != 1)
dlx_pos = dlx_filt[dlx_pos_keep] - dlx_filt[dlx_pos_keep].min(axis=0)
dlx_pos_sights = dlx_sights[dlx_pos_keep]

# Remove sights 17, 18, 31, 32 from DDC bolometer (usually they are just
# radiation from the separatrix) and bring everityhing above 0 (necessary for
# weighted average)
ddc_pos_keep = ((ddc_sights != 32) & (ddc_sights != 31) & (ddc_sights != 18) &
                (ddc_sights != 17))
ddc_pos = ddc_filt[ddc_pos_keep]
ddc_pos_sights = ddc_sights[ddc_pos_keep]
# OLD CODE LEFT FOR REFERENCE
# if detrend:
#     print('Using scipy detrend non-deterministic method')
#     ddc_pos = sg.detrend(ddc_pos, axis=0)
# else:
#     print('Using deterministic method (check comments for more info)')
#     # The code evaluates the line coefficients between the two points
#     # ddc_pos_sights[-1] and ddc_pos_sights[0], then it evaluates the
#     # linear offset and removes it
#     m_off = (ddc_pos[-1] - ddc_pos[0])/(ddc_pos_sights[-1] -
#     ddc_pos_sights[0])
#     c_off = ddc_pos[-1] - m_off*ddc_pos_sights[-1]
#     ddc_pos = ddc_pos - (np.outer(ddc_pos_sights, m_off) + c_off)
# # Everything is brought above 0 (necessary for weighted average)
ddc_pos -= ddc_pos.min(axis=0)

# The XPR position is firstly approximated by queryng for the index of the
# peaks in the bolometer signals
xpr_dlx_nearest_index = np.apply_along_axis(find_peaks_2d, 0, dlx_pos,
                                            distance=dlx_pos.shape[0]*0.66)
xpr_ddc_nearest_index = np.apply_along_axis(find_peaks_2d, 0, ddc_pos,
                                            distance=ddc_pos.shape[0]*0.66)

# In order to get a better position we can try to evaluate the weighted average
# (deterministic) or do a gaussian fit (non deterministic) but, to do so we
# need our coordinate arrays (dlx/ddc_pos_sights) to have the same dimension of
# the signal arrays. Therefore, they need to be extended through time:
dlx_pos_sights_m = np.tile(dlx_pos_sights, (time_dlx_filt.size, 1)).transpose()
ddc_pos_sights_m = np.tile(ddc_pos_sights, (time_ddc_filt.size, 1)).transpose()

# The weighted avarage (or the fitting) will be done on 5 points centered
# around the maximum. Hence, to prevent any slicing error, we extend the
# coordinates and the signal arrays by two rows to the top and to the bottom.
# The extensions are generated in such a manner that they will be ignored both
# by the weighted average and the gaussian fit DLX extension
# DLX extension
dlx_pos_sights_ext = np.zeros((2, time_dlx_filt.size), dtype=int)
dlx_pos_sights_m = np.append(dlx_pos_sights_ext + dlx_pos_sights_m[0:2] -2,
                             dlx_pos_sights_m, axis=0)
dlx_pos_sights_m = np.append(dlx_pos_sights_m, dlx_pos_sights_ext +
                             dlx_pos_sights_m[-2:] +2, axis=0)

dlx_pos_ext = np.zeros((2, time_dlx_filt.size))
dlx_pos = np.append(dlx_pos_ext, dlx_pos, axis=0)
dlx_pos = np.append(dlx_pos, dlx_pos_ext, axis=0)

# DDC extension
ddc_pos_sights_ext = np.zeros((2, time_ddc_filt.size), dtype=int)
ddc_pos_sights_m = np.append(ddc_pos_sights_ext + ddc_pos_sights_m[0:2] -2,
                             ddc_pos_sights_m, axis=0)
ddc_pos_sights_m = np.append(ddc_pos_sights_m, ddc_pos_sights_ext +
                             ddc_pos_sights_m[-2:] +2, axis=0)

ddc_pos_ext = np.zeros((2, time_ddc_filt.size))
ddc_pos = np.append(ddc_pos_ext, ddc_pos, axis=0)
ddc_pos = np.append(ddc_pos, ddc_pos_ext, axis=0)

# We can finally do some advanced numpy slicing in order to get the desired 5
# points arrays.
# Just remember that we have add 2 more rows at the beginning of the arrays so
# every index should be shifted up of 2 positions
# (xpr_dlx_nearest_index -2 -----> xpr_dlx_nearest_index
#  xpr_dlx_nearest_index -1 -----> xpr_dlx_nearest_index +1
#  and so on...)
# DLX slicing
dlx_row_slicer = np.array([xpr_dlx_nearest_index,
                           xpr_dlx_nearest_index +1,
                           xpr_dlx_nearest_index +2,
                           xpr_dlx_nearest_index +3,
                           xpr_dlx_nearest_index +4])
dlx_column_slicer = np.arange(0, time_dlx_filt.size)
dlx_pos_sliced = dlx_pos[dlx_row_slicer, dlx_column_slicer]
dlx_pos_sights_m = dlx_pos_sights_m[dlx_row_slicer, dlx_column_slicer]

# DDC slicing
ddc_row_slicer = np.array([xpr_ddc_nearest_index,
                           xpr_ddc_nearest_index +1,
                           xpr_ddc_nearest_index +2,
                           xpr_ddc_nearest_index +3,
                           xpr_ddc_nearest_index +4])
ddc_column_slicer = np.arange(0, time_ddc_filt.size)
ddc_pos_sliced = ddc_pos[ddc_row_slicer, ddc_column_slicer]
ddc_pos_sights_m = ddc_pos_sights_m[ddc_row_slicer, ddc_column_slicer]

# Fit a gaussian or re-evaluate the weighted average
if gaussian:
    print('Using non-deterministic gaussian fit method')
    xpr_dlx = np.apply_along_axis(fit_gauss_time, 0,
                                  np.append(dlx_pos_sights_m,
                                            dlx_pos_sliced, axis=0))
    xpr_ddc = np.apply_along_axis(fit_gauss_time, 0,
                                  np.append(ddc_pos_sights_m,
                                            ddc_pos_sliced, axis=0))
else:
    print('Using weighted average method')
    xpr_dlx = np.average(dlx_pos_sights_m, axis=0, weights=dlx_pos_sliced)
    xpr_ddc = np.average(ddc_pos_sights_m, axis=0, weights=ddc_pos_sliced)

# Convert from bolometers coordinates to real coordinates
xpr_x, xpr_y = get_real_position(ORGIN_DLX, M_STAR_DLX, M_REF_DLX, OFFSET_DLX,
                                 xpr_dlx, ORGIN_DDC, M_STAR_DDC, M_REF_DDC,
                                 OFFSET_DDC, xpr_ddc)

print('Further processing: done')



# PRINTING FILE (OPTIONAL)
if options.xpr_start_time:
    df_data = np.r_['0,2',
                    xpr_x,
                    xpr_y,
                    time_dlx_filt,
                    np.where(time_dlx_filt > options.xpr_start_time, 1, 0)]
    df = pd.DataFrame(df_data, index=['R (m)', 'z (m)', 't (s)', 'is XPR ' +
                                      '(bool)'])
    df.to_csv('XPR_position.csv', header=False)




# PLOTTING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# General settings
style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
px = 1/plt.rcParams['figure.dpi']  # from pixel to inches
fig = plt.figure(figsize=(1600*px, 1000*px))
plt.suptitle(f'SHOT #{shot}', fontsize=32, fontweight='bold')
frames = time_dlx_filt.size

# Subplots
ax1 = plt.subplot2grid((5, 3), (0, 0), rowspan=1, colspan=2)
ax2 = plt.subplot2grid((5, 3), (1, 2), rowspan=3, colspan=1, aspect='equal')
ax1_2 = plt.subplot2grid((5, 3), (1, 0), rowspan=1, colspan=2)
ax1_3 = plt.subplot2grid((5, 3), (2, 0), rowspan=1, colspan=2)
axtext = plt.subplot2grid((5, 3), (0, 2), rowspan=1, colspan=1)
ax3 = plt.subplot2grid((5, 3), (3, 0), rowspan=1, colspan=1)
ax4 = plt.subplot2grid((5, 3), (3, 1), rowspan=1, colspan=1)
ax5 = plt.subplot2grid((5, 3), (4, 0), rowspan=1, colspan=1)
ax6 = plt.subplot2grid((5, 3), (4, 1), rowspan=1, colspan=1)

# Te_ld subplot
ax1.plot(time_Te_ld, Te_ld[9], label='10th core')
ax1.plot(time_Te_ld, Te_ld[10], label='11th core')
ax1.plot(time_Te_ld, Te_ld[11], label='12th core')
ax1.set_title(f'DTS electron temperature nearby X-point')
ax1.set_xlabel('s')
ax1.set_ylabel(Te_ld.phys_unit)
ax1.legend()

# Ne_ld subplot
ax1_2.plot(time_Ne_ld, Ne_ld[9], label='10th core')
ax1_2.plot(time_Ne_ld, Ne_ld[10], label='11th core')
ax1_2.plot(time_Ne_ld, Ne_ld[11], label='12th core')
ax1_2.set_title(f'DTS electron density nearby X-point')
ax1_2.set_xlabel('s')
ax1_2.set_ylabel(Ne_ld.phys_unit)
ax1_2.legend()

# Ne_ld subplot
ax1_3.plot(time_Te_ld, Te_ld[9] * Ne_ld[9], label='10th core')
ax1_3.plot(time_Te_ld, Te_ld[10] * Ne_ld[10], label='11th core')
ax1_3.plot(time_Te_ld, Te_ld[11] * Ne_ld[11], label='12th core')
ax1_3.set_title(f'Evaluated electron pressure nearby X-point')
ax1_3.set_xlabel('s')
ax1_3.set_ylabel('eV/m^3')
ax1_3.legend()

# X-point radiator position subplot
# TO ANIMATE BEGIN
sep_image, = ax2.plot([], [], '-', color=colors[0])
xpr_image, = ax2.plot([], [], 'o', color=colors[0])
r_sep, z_sep = sf.rho2rz(equ, 1, t_in=time_dlx_filt, coord_in='rho_pol')
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
             '-', color=colors[1])
for i in range(0, ddc_start_R.size):
    ax2.plot([ddc_start_R[i], ddc_end_R[i]], [ddc_start_Z[i], ddc_end_Z[i]],
             '-', color=colors[1])
# Vessel drawing
for gc in gc_d.values():
    ax2.plot(gc.r, gc.z, '-', color=colors[1])
# Settings
ax2.set_title(f'X-point radiator position')
ax2.set_xlabel('R')
ax2.set_ylabel('z')
ax2.set_xlim(1.1, 1.8)
ax2.set_ylim(-1.3, -0.6)

# DLX subplot
cont_1 = ax3.contourf(time_dlx_filt, dlx_sights, dlx_filt, options.depth,
                      cmap='inferno')
ax3.plot(time_dlx_filt, xpr_dlx, color=colors[1])
ax3.set_title(f'DLX detected radiation')
ax3.set_xlabel('s')
ax3.set_ylabel('sight')
fig.colorbar(cont_1, ax=ax3)

# DDC subplot
cont_2 = ax4.contourf(time_ddc_filt, ddc_sights, ddc_filt, options.depth,
                      cmap='inferno')
ax4.plot(time_ddc_filt, xpr_ddc, color=colors[1])
ax4.set_title(f'DDC detected radiation')
ax4.set_xlabel('s')
ax4.set_ylabel('sight')
fig.colorbar(cont_2, ax=ax4)

# Evolution of DLX signal subplot
# TO ANIMATE BEGIN
dlx_image, = ax5.plot([], [])
dlx_peak_image = ax5.axvline(color=colors[0])
# TO ANIMATE END
ax5.set_title(f'DLX signal evolution')
ax5.set_xlabel('sight')
ax5.set_ylabel('W $m^{-2}$')
ax5.set_xlim(dlx_sights[0]-0.5, dlx_sights[-1]+0.5)
ax5.set_ylim(dlx_filt.min(), dlx_filt.max())

# Evolution of DDC signal subplot
# TO ANIMATE BEGIN
ddc_image, = ax6.plot([], [])
ddc_peak_image = ax6.axvline(color=colors[0])
# TO ANIMATE END
ax6.set_title(f'DDC signal evolution')
ax6.set_xlabel('sight')
ax6.set_ylabel('W $m^{-2}$')
ax6.set_xlim(ddc_sights[0]-0.5, ddc_sights[-1]+0.5)
ax6.set_ylim(ddc_filt.min(), ddc_filt.max())

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
    dlx_image.set_data(dlx_sights, dlx_filt[..., frame])
    dlx_peak_image.set_xdata(xpr_dlx[frame])
    ddc_image.set_data(ddc_sights, ddc_filt[..., frame])
    ddc_peak_image.set_xdata(xpr_ddc[frame])
    timestamp.set_text(f'Time:  {time_dlx_filt[frame]:.3f} s')
    return sep_image, xpr_image, dlx_image, dlx_peak_image, ddc_image, ddc_peak_image, timestamp

ani = FuncAnimation(fig, update_ani, frames=frames, interval=f_interval,
                    blit=True)

# Show plot or save
plt.tight_layout()
plt.show()

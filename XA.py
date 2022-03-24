#!/usr/bin/python3

# Python code for X-point radiator access parameter analysis made by:
# - Luca Cinnirella
#
# and modified/updated by:
#

# Last update: 08.03.2022

# IMPORTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import sys
from optparse import OptionParser, OptionGroup
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import scipy.ndimage as ndm
import aug_sfutils as sf



# FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Find nearest index
def find_nearest_index(array, value, axis=None):
    return np.absolute(array-value).argmin(axis=axis)

# Select shortest array
def shortest_array(array_list):
    sizes = np.array([array.size for array in array_list])
    return array_list[sizes.argmin()]

# Find nearest index for given reference and test arrays
def find_nearest_multiple_index(ref_array, test_array):
    # For a given reference array and a test scalar, finding the nearest index
    # is quite simple: evaluate the distance of the scalar from each array
    # element (|array - constant|) and then query for the argument of the
    # minimum (the index of the closest element to the given scalar).
    # For two arrays we should cycle through the test array elements but for
    # very large arrays it would take too much time. We can improve on speed by
    # cloning the reference array along an additional dimension (making new
    # columns that are clones of the original array) and then subtracting from
    # each row the test array (done automatically by numpy). Finally we query
    # for the argument of the minimum along each column
    #
    # e.g.
    # REF_ARRAY = [0 1 2 ... n]
    # (the subsequent step is necessary since numpy is row major)
    # we get to REF_ARRAY = ⎡[0]⎤
    #                       ⎢[1]⎥
    #                       ⎢[2]⎥
    #                       ⎢...⎥
    #                       ⎣[n]⎦
    # and finally to REF_ARRAY = ⎡[0 0 0 ... 0]⎤
    #                            ⎢[1 1 1 ... 1]⎥
    #                            ⎢[2 2 2 ... 2]⎥
    #                            ⎢[... ... ...]⎥
    #                            ⎣[n n n ... n]⎦
    idx = np.abs(np.tile(np.expand_dims(ref_array, -1), test_array.size) -
                 test_array).argmin(0)
    # The function returns also an indexes of indexes array
    index_idx = np.arange(idx.size)
    return idx, index_idx



# OPTIONS HANDLER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
parser = OptionParser(usage='Usage: %prog [mandatory] args [options] arg',
                      add_help_option=False)
parser.add_option('-h', '--help',
                  action='help',
                  help='Show this help message')
parser.add_option('-e', '--equilibrium_diagnostic',
                  metavar='EQUILIBRIUM_DIAGNOSTIC',
                  action='store', type='str', dest='equ_diag', default='EQH',
                  help='Select which diagnostic is used for magnetic ' +
                       'reconstruction (FPP, EQI or EQH, default is EQH)')
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
equ_diag = options.equ_diag



# QUERYING SHOTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get shot-files relative to Function Parametrization
# and equilibrium
fpg = sf.SFREAD(shot, fpg_diag)
equ = sf.EQU(shot, diag=equ_diag)
# bpt = sf.SFREAD(shot, 'BPT', exp='DAVIDP')
iob = sf.SFREAD(shot, 'IOB')
ida = sf.SFREAD(shot, 'IDA')



# QUERYING SIGNALS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get calibrated time traces for q95, position of the magnetic axis, position
# of the lower x-point, power through separatrix, F01 divertor manometer,
# upstream temperature and upstream density
if fpg.status:
    q_95 = fpg.getobject('q95', cal=True).astype(np.double)
    time_q_95 = fpg.gettimebase('q95')
    r_magax = fpg.getobject('Rmag', cal=True)
    z_magax = fpg.getobject('Zmag', cal=True)
    time_magax = fpg.gettimebase('Zmag')
    r_xp = fpg.getobject('Rxpu', cal=True)
    z_xp = fpg.getobject('Zxpu', cal=True)
    time_xp = fpg.gettimebase('Zxpu')
else:
    sys.exit('Error while loading ' + fpg_diag)

# if bpt.status:
#     p_sep = bpt.getobject('Pr_sep', cal=True)
#     p_sep_min = bpt.getobject('Pr_sep-', cal=True)
#     p_sep_plus = bpt.getobject('Pr_sep+', cal=True)
#     p_sep_X = bpt.getobject('Pr_sepX', cal=True)
#     p_sep_X_min = bpt.getobject('Pr_sepX-', cal=True)
#     p_sep_X_plus = bpt.getobject('Pr_sepX+', cal=True)
# else:
#     sys.exit('Error while loading Pierre David\'s BPT')

if iob.status:
    n_0 = iob.getobject('F01', cal=True).astype(np.double)
    time_n_0 = iob.gettimebase('F01')
else:
    sys.exit('Error while loading IOB')

if ida.status:
    n_u_profile = ida.getobject('ne', cal=True)
    time_n_u = ida.gettimebase('ne')
    area_n_u = ida.getareabase('ne')
    T_u_profile = ida.getobject('Te', cal=True)
    time_T_u = ida.gettimebase('Te')
    area_T_u = ida.getareabase('Te')

print('Querying: done')



# UNIVERSAL TIME SELECTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get maximum common time interval boundaries
start = max(time_q_95[0], time_magax[0], time_xp[0], time_n_0[0], time_n_u[0],
            time_T_u[0], 2)
end = min(time_q_95[-1], time_magax[-1], time_xp[-1], time_n_0[-1],
          time_n_u[-1], time_T_u[-1], 8)

# Get indexes of boundaries for q95
start_index_q_95 = find_nearest_index(time_q_95, start)
end_index_q_95 = find_nearest_index(time_q_95, end)

# Get indexes of boundaries for magnetic axis
start_index_magax = find_nearest_index(time_magax, start)
end_index_magax = find_nearest_index(time_magax, end)

# Get indexes of boundaries for X-point
start_index_xp = find_nearest_index(time_xp, start)
end_index_xp = find_nearest_index(time_xp, end)

# Get indexes of boundaries for neutral density
start_index_n_0 = find_nearest_index(time_n_0, start)
end_index_n_0 = find_nearest_index(time_n_0, end)

# Get indexes of boundaries for upstream density
start_index_n_u = find_nearest_index(time_n_u, start)
end_index_n_u = find_nearest_index(time_n_u, end)

# Get indexes of boundaries for upstream temperature
start_index_T_u = find_nearest_index(time_T_u, start)
end_index_T_u = find_nearest_index(time_T_u, end)

# Slice q95 and relative time
q_95 = q_95[start_index_q_95:end_index_q_95+1]
time_q_95 = time_q_95[start_index_q_95:end_index_q_95+1]

# Slice magnetic axis and relative time
r_magax = r_magax[start_index_magax:end_index_magax+1]
z_magax = z_magax[start_index_magax:end_index_magax+1]
time_magax = time_magax[start_index_magax:end_index_magax+1]

# Slice X-point and relative time
r_xp = r_xp[start_index_xp:end_index_xp+1]
z_xp = z_xp[start_index_xp:end_index_xp+1]
time_xp = time_xp[start_index_xp:end_index_xp+1]

# Slice neutral density and relative time
n_0 = n_0[start_index_n_0:end_index_n_0+1]
time_n_0 = time_n_0[start_index_n_0:end_index_n_0+1]

# Slice upstream density and relative time
n_u_profile = n_u_profile[:, start_index_n_u:end_index_n_u+1]
time_n_u = time_n_u[start_index_n_u:end_index_n_u+1]
area_n_u = area_n_u[:, start_index_n_u:end_index_n_u+1]

# Slice upstream temperature and relative time
T_u_profile = T_u_profile[:, start_index_T_u:end_index_T_u+1]
time_T_u = time_T_u[start_index_T_u:end_index_T_u+1]
area_T_u = area_T_u[:, start_index_T_u:end_index_T_u+1]

# Get the time base with the lowest sampling rate
time_base = shortest_array([time_q_95, time_magax, time_xp, time_n_0, time_n_u,
                            time_T_u])

# Since we need to downsample the longest array to the same size of the
# shortest array we must before apply a gaussian blur in order to avoid
# aliasing. The sigma of the blur is chosen in such a way that the distance
# between 2 longest_array elements corresponding to adjacent shortest_array
# elements must be equal to 3 times sigma.
#
# e.g.
# time_shortest_array:      [0.0]               [0.4]               [0.8]
# time_longest_array:       [0.0][0.1][0.2][0.3][0.4][0.5][0.6][0.7][0.8][0.9]
# time_shortest_array.size: 3
# time_longest_array.size:  10
# In this case 3 times sigma is equal to 4, which is the distance between 4th
# and 0th elements of longest_array (or, equivalently, time_longest array), but
# can be evaluated in a more general way by the following expression:
# 3 * sigma = np.ceil(time_longest_array.size / time_shortest_array.size)
# therefore we have:
# sigma = np.ceil(time_longest_array.size / time_shortest_array.size) / 3
# In the case in which time_longest_array.size is equal to
# time_shortest_array.size, a sigma of 0.33 prevent the Gaussian filter from
# blurring the signal (since all the effects of the filter are dampened already
# at the immediately preceding and succeeding points

# Resample q95
sigma_q_95 = np.ceil(time_q_95.size / time_base.size) / 3
q_95 = ndm.gaussian_filter(q_95, sigma=sigma_q_95)
index_q_95, _ = find_nearest_multiple_index(time_q_95, time_base)
time_q_95 = time_q_95[index_q_95]
q_95 = q_95[index_q_95]

# Resample magnetic axis
sigma_magax = np.ceil(time_magax.size / time_base.size) / 3
r_magax = ndm.gaussian_filter(r_magax, sigma=sigma_magax)
z_magax = ndm.gaussian_filter(z_magax, sigma=sigma_magax)
index_magax, _ = find_nearest_multiple_index(time_magax, time_base)
time_magax = time_magax[index_magax]
r_magax = r_magax[index_magax]
z_magax = z_magax[index_magax]

# Resample X-point
sigma_xp = np.ceil(time_xp.size / time_base.size) / 3
r_xp = ndm.gaussian_filter(r_xp, sigma=sigma_xp)
z_xp = ndm.gaussian_filter(z_xp, sigma=sigma_xp)
index_xp, _ = find_nearest_multiple_index(time_xp, time_base)
time_xp = time_xp[index_xp]
r_xp = r_xp[index_xp]
z_xp = z_xp[index_xp]

# Resample neutral density
sigma_n_0 = np.ceil(time_n_0.size / time_base.size) / 3
n_0 = ndm.gaussian_filter(n_0, sigma=sigma_n_0)
index_n_0, _ = find_nearest_multiple_index(time_n_0, time_base)
time_n_0 = time_n_0[index_n_0]
n_0 = n_0[index_n_0]

# Resample upstream density
sigma_n_u = (0, np.ceil(time_n_u.size / time_base.size) / 3)
n_u_profile = ndm.gaussian_filter(n_u_profile, sigma=sigma_n_u)
index_n_u, _ = find_nearest_multiple_index(time_n_u, time_base)
time_n_u = time_n_u[index_n_u]
n_u_profile = n_u_profile[:, index_n_u]
area_n_u = area_n_u[:, index_n_u]

# Resample upstream temperature
sigma_T_u = (0, np.ceil(time_T_u.size / time_base.size) / 3)
T_u_profile = ndm.gaussian_filter(T_u_profile, sigma=sigma_T_u)
index_T_u, _ = find_nearest_multiple_index(time_T_u, time_base)
time_T_u = time_T_u[index_T_u]
T_u_profile = T_u_profile[:, index_T_u]
area_T_u = area_T_u[:, index_T_u]

print('Time selection done')



# MAJOR RADIUS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# By definition the major radius is equal to the R coordinate of the major
# radius
R_0 = r_magax.copy().astype(np.double)



# MINOR RADIUS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The minor radius is evaluated as the distance between the magnetic axis and
# the separatrix at the midplane
magax = np.array([r_magax, z_magax])
r_m100_int, z_m100_int = sf.rhoTheta2rz(equ, 1, theta_in=0, t_in=time_magax,
                                        coord_in='rho_pol')
m100_intersection = np.array([r_m100_int.flatten(), z_m100_int.flatten()])
a = np.linalg.norm(m100_intersection - magax, axis=0).astype(np.double)



# FLUX EXPANSION RATIO
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get coordinates of the intersection between the ρ95 surface and the line
# connecting the magnetic axis with the lower x-point. Then, evaluate the
# distance between the intersection and the lower x-poin itself
xp = np.array([r_xp, z_xp])
theta = np.arctan2(xp[1]-magax[1], xp[0]-magax[0])
r_l95_int = []
z_l95_int = []

for i in range(0, time_xp.size):
    r_l95_temp, z_l95_temp = sf.rhoTheta2rz(equ, 0.95, theta_in=theta[i],
                                            t_in=time_xp[i],
                                            coord_in='rho_pol')
    r_l95_int.append(r_l95_temp.flatten()[0])
    z_l95_int.append(z_l95_temp.flatten()[0])
l95_intersection = np.array([r_l95_int, z_l95_int])
xp_expansion = np.linalg.norm(l95_intersection - xp, axis=0).astype(np.double)

# Get coordinates of the intersection between the mid-plane and the ρ100 and
# ρ95 surfaces. Then, evaluate the distance between the two
r_m95_int, z_m95_int = sf.rhoTheta2rz(equ, 0.95, theta_in=0, t_in=time_xp, coord_in='rho_pol')
m95_intersection = np.array([r_m95_int.flatten(), z_m95_int.flatten()])
mid_expansion = np.linalg.norm(m100_intersection - m95_intersection, axis=0).astype(np.double)

# The flux expansion ratio should be the ratio between two areas enveloping the
# same magnetic lines but, due to the particular geometry,
expansion_ratio = xp_expansion / mid_expansion



# POWER RADIATED INSIDE THE SEPARATRIX
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Nothing to do



# NEUTRAL DENSITY AT THE X-POINT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Nothing to do



# UPSTREAM DENSITY
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We want to select the value at the separatrix
n_u = n_u_profile[find_nearest_index(area_n_u, 1, axis=0),
                  np.arange(n_u_profile.shape[-1])].astype(np.double)



# UPSTREAM TEMPERATURE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We want to select the value at the separatrix
T_u = T_u_profile[find_nearest_index(area_T_u, 1, axis=0),
                  np.arange(T_u_profile.shape[-1])].astype(np.double)



# EVALUATION OF XA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
X_A = R_0**2 * q_95**2 * expansion_ratio * n_u * n_0 / (a * T_u**2.5)



# PLOTTING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# General settings
style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
px = 1/plt.rcParams['figure.dpi']  # from pixel to inches
fig = plt.figure(figsize=(1600*px, 1000*px))
plt.suptitle(f'SHOT #{shot}', fontsize=32, fontweight='bold')

# Subplots
ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=1, colspan=1)
ax2 = ax1.twinx()
ax3 = plt.subplot2grid((5, 1), (1, 0), rowspan=1, colspan=1)
ax4 = ax3.twinx()
ax5 = plt.subplot2grid((5, 1), (2, 0), rowspan=1, colspan=1)
ax6 = ax5.twinx()
ax7 = plt.subplot2grid((5, 1), (3, 0), rowspan=1, colspan=1)
ax8 = plt.subplot2grid((5, 1), (4, 0), rowspan=1, colspan=1)

# R_0 subplot
ax1.plot(time_magax, R_0)
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Major radius (m)', color=colors[0])

# a subplot
ax2.plot(time_magax, a, color=colors[1])
ax2.set_ylabel('Minor radius (m)', color=colors[1])

# expansion_ratio subplot
ax3.plot(time_magax, expansion_ratio)
ax3.set_xlabel('time (s)')
ax3.set_ylabel('Expansion ratio', color=colors[0])

# q_95 subplot
ax4.plot(time_q_95, q_95, color=colors[1])
ax4.set_ylabel('q95', color=colors[1])

# n_0 subplot
ax5.plot(time_n_0, n_0)
ax5.set_xlabel('time (s)')
ax5.set_ylabel('Neutral density ($m^{-3}$)', color=colors[0])

# n_u subplot
ax6.plot(time_n_u, n_u, color=colors[1])
ax6.set_ylabel('Upstream density ($m^{-3}$)', color=colors[1])

# T_u subplot
ax7.plot(time_T_u, T_u, color=colors[4])
ax7.set_title(f'Upstream temperature')
ax7.set_xlabel('time (s)')
ax7.set_ylabel('eV')

# X_A subplot
ax8.plot(time_base, X_A, color=colors[2])
ax8.set_title(f'Access parameter')
ax8.set_xlabel('s')

# Show plot or save
plt.tight_layout()
plt.show()

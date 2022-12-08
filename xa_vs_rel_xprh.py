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
import argparse
import pandas as pd
import numpy as np
import aug_sfutils as sf
import scipy.ndimage as ndm
import matplotlib.pyplot as plt
from matplotlib import style

import sig_proc as sgpr
import manual_calibrators as manc


# OPTIONS HANDLER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
parser = argparse.ArgumentParser(description='Plot time traces of edge ' +
                                 'temperature profile')
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
args = parser.parse_args()

if args.mag_equ_diag == 'FPP':
    FPG_DIAG = 'FPG'
elif args.mag_equ_diag == 'EQI':
    FPG_DIAG = 'GQI'
elif args.mag_equ_diag == 'EQH':
    FPG_DIAG = 'GQH'

time_window = args.time_window / 1000
mag_equ_diag = args.mag_equ_diag


# PLOTTING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# General settings
style.use('bmh')
plt.rc('font', size=16)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
px = 1/plt.rcParams['figure.dpi']  # from pixel to inches
fig = plt.figure(figsize=(1600*px, 1000*px))
plt.suptitle('Correlation of X-point radiator access parameter\nand X-point ' +
             'radiator height during XPR cold core phase', fontsize=32,
             fontweight='bold')

# Subplots
ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

ax1.set_xlabel('$X_A$')
ax1.set_ylabel('1 - $\\rho_{pol}$')


for shot in [40365, 40366, 41157, 41158]:
    # COLD CORE XPR TIME BOUNDARIES
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    df_cld_time = pd.read_csv('./csvs/XPR_cld_times.csv', index_col=0).loc[shot]

    cld_start = np.atleast_1d(df_cld_time['start'])
    cld_end = np.atleast_1d(df_cld_time['end'])


    # READING SHOT TIME BOUNDARIES
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    df_shot_time = pd.read_csv('./csvs/equilibria.csv', index_col=0).loc[shot]

    shot_start = df_shot_time['start']
    shot_end = df_shot_time['end']


    # READING XPR POSITION DATA
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    df_pos_xpr = pd.read_csv('./csvs/XPR_position/' + str(shot) +
                             '_XPR_position_gaussian.csv', index_col=0)
    time_xpr = df_pos_xpr.index.values
    R_xpr = df_pos_xpr['R (m)'].values
    z_xpr = df_pos_xpr['z (m)'].values
    rho_xpr = df_pos_xpr['rho pol'].values
    good_pos_xpr = df_pos_xpr['detection with scipy.signal.find_peaks ' +
                              '(bool)'].values


    # QUERYING SHOTS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get shot-files relative to function parametrization, equilibrium , integrated
    # data analysis and ionization manometers
    fpg = sf.SFREAD(shot, FPG_DIAG)
    equ = sf.EQU(shot, diag=mag_equ_diag)
    ida = sf.SFREAD(shot, 'IDA')
    ioa = manc.SFIOCF01(shot, sensitivity=3.)
    iob = manc.SFIOCF01(shot)
    ioc = manc.SFIOCF01(shot, sensitivity=2.)
    tot = sf.SFREAD(shot, 'TOT')
    bpt = sf.SFREAD(shot, 'BPT', exp='DAVIDP')


    # QUERYING SIGNALS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get calibrated time traces for position of the magnetic axis, minor
    # horizontal plasma radius, q95, position of the lower x-point, upstream
    # temperature and density, F01 divertor manometer and upstream temperature and
    # density from two point model reconstruction (by Thomas Eich)
    if fpg.status:
        r_magax = fpg.getobject('Rmag', cal=True).astype(np.double)
        z_magax = fpg.getobject('Zmag', cal=True).astype(np.double)
        a = fpg.getobject('ahor', cal=True).astype(np.double)
        q_95 = -fpg.getobject('q95', cal=True).astype(np.double)
        r_xp = fpg.getobject('Rxpu', cal=True).astype(np.double)
        z_xp = fpg.getobject('Zxpu', cal=True).astype(np.double)
        time_fpg = fpg.gettimebase('Zmag')
    else:
        sys.exit('Error while loading ' + FPG_DIAG)

    if ida.status:
        T_u = ida.getobject('Te', cal=True).astype(np.double)
        n_u = ida.getobject('ne', cal=True).astype(np.double)
        area_ida = ida.getareabase('Te')
        time_ida = ida.gettimebase('Te')
        # Choosing separatrix data
        ida_idx = sgpr.find_nearest_index(area_ida, 1.0, axis=0)
        T_u = T_u[ida_idx, np.arange(time_ida.size)]
        n_u = n_u[ida_idx, np.arange(time_ida.size)]
    else:
        sys.exit('Error while loading IDA')

    if ioa.status and iob.status and ioc.status:
        n_0_l = ioa.getobject('F01').astype(np.double)
        n_0 = iob.getobject('F01').astype(np.double)
        n_0_u = ioc.getobject('F01').astype(np.double)
        time_iob = iob.gettimebase()
        iob_par = iob.getparset('CONVERT')
        iob_2_n = iob_par['density']
        n_0_l = n_0_l * iob_2_n
        n_0 = n_0 * iob_2_n
        n_0_u = n_0_u * iob_2_n
    else:
        sys.exit('Error while loading IOB')

    print('\n\n')
    print('Querying: done')


    # UNIVERSAL TIME SELECTION
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get maximum common time interval boundaries
    start = max(time_fpg[0], time_ida[0], time_iob[0], time_xpr[0], shot_start)
    end = min(time_fpg[-1], time_ida[-1], time_iob[-1], time_xpr[-1], shot_end)

    # Get indexes of boundaries for XPR
    start_index_xpr = sgpr.find_nearest_index(time_xpr, start)
    end_index_xpr = sgpr.find_nearest_index(time_xpr, end)

    # Get indexes of boundaries for FPG
    start_index_fpg = sgpr.find_nearest_index(time_fpg, start)
    end_index_fpg = sgpr.find_nearest_index(time_fpg, end)

    # Get indexes of boundaries for IDA
    start_index_ida = sgpr.find_nearest_index(time_ida, start)
    end_index_ida = sgpr.find_nearest_index(time_ida, end)

    # Get indexes of boundaries for IOB
    start_index_iob = sgpr.find_nearest_index(time_iob, start)
    end_index_iob = sgpr.find_nearest_index(time_iob, end)

    # Slice XPR quantities and relative time
    R_xpr = R_xpr[..., start_index_xpr:end_index_xpr + 1]
    z_xpr = z_xpr[..., start_index_xpr:end_index_xpr + 1]
    rho_xpr = rho_xpr[..., start_index_xpr:end_index_xpr + 1]
    good_pos_xpr = good_pos_xpr[..., start_index_xpr:end_index_xpr + 1]
    time_xpr = time_xpr[..., start_index_xpr:end_index_xpr + 1]

    # Slice FPG quantities and relative time
    r_magax = r_magax[..., start_index_fpg:end_index_fpg + 1]
    z_magax = z_magax[..., start_index_fpg:end_index_fpg + 1]
    a = a[..., start_index_fpg:end_index_fpg + 1]
    q_95 = q_95[..., start_index_fpg:end_index_fpg + 1]
    r_xp = r_xp[..., start_index_fpg:end_index_fpg + 1]
    z_xp = z_xp[..., start_index_fpg:end_index_fpg + 1]
    time_fpg = time_fpg[..., start_index_fpg:end_index_fpg + 1]

    # Slice IDA quantities and relative time
    T_u = T_u[..., start_index_ida:end_index_ida + 1]
    n_u = n_u[..., start_index_ida:end_index_ida + 1]
    time_ida = time_ida[..., start_index_ida:end_index_ida + 1]

    # Slice IOB quantities and relative time
    n_0_l = n_0_l[..., start_index_iob:end_index_iob + 1]
    n_0 = n_0[..., start_index_iob:end_index_iob + 1]
    n_0_u = n_0_u[..., start_index_iob:end_index_iob + 1]
    time_iob = time_iob[..., start_index_iob:end_index_iob + 1]

    print('Time selection done')


    # MEDIAN FILTERING
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Evaluate sampling rate for median filter function (sampling rates in the
    # middle of a time trace are more reliable than averages)
    f_s_xpr = 1 / (time_xpr[int(time_xpr.size / 2) + 1] -
                   time_xpr[int(time_xpr.size / 2)])
    f_s_fpg = 1 / (time_fpg[int(time_fpg.size / 2) + 1] -
                   time_fpg[int(time_fpg.size / 2)])
    f_s_ida = 1 / (time_ida[int(time_ida.size / 2) + 1] -
                   time_ida[int(time_ida.size / 2)])
    f_s_iob = 1 / (time_iob[int(time_iob.size / 2) + 1] -
                   time_iob[int(time_iob.size / 2)])

    # Apply median filters on IDA quantities
    _, T_u = sgpr.median_filter(f_s_ida, time_window, time_ida, T_u)
    time_ida, n_u = sgpr.median_filter(f_s_ida, time_window, time_ida, n_u)

    # Apply median filters on IOB quantities
    _, n_0_l = sgpr.median_filter(f_s_iob, time_window, time_iob, n_0_l)
    _, n_0 = sgpr.median_filter(f_s_iob, time_window, time_iob, n_0)
    time_iob, n_0_u = sgpr.median_filter(f_s_iob, time_window, time_iob, n_0_u)

    # Re-evaluate sampling rate for filtered quantities
    f_s_ida = 1 / (time_ida[int(time_ida.size / 2) + 1] -
                   time_ida[int(time_ida.size / 2)])
    f_s_iob = 1 / (time_iob[int(time_iob.size / 2) + 1] -
                   time_iob[int(time_iob.size / 2)])

    print('Median filters applied')


    # FLUX EXPANSION RATIO
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get coordinates of the intersection between the ρ95 surface and the line
    # connecting the magnetic axis with the lower x-point. Then, evaluate the
    # distance between the intersection and the lower x-point itself
    magax = np.array([r_magax, z_magax])
    xp = np.array([r_xp, z_xp])
    theta = np.arctan2(xp[1]-magax[1], xp[0]-magax[0])
    r_l95_int = list()
    z_l95_int = list()

    for index, time in enumerate(time_fpg):
        r_l95_temp, z_l95_temp = sf.rhoTheta2rz(equ, 0.95, theta_in=theta[index],
                                                t_in=time,
                                                coord_in='rho_pol')
        r_l95_int.append(r_l95_temp.flatten()[0])
        z_l95_int.append(z_l95_temp.flatten()[0])
    l95_intersection = np.array([r_l95_int, z_l95_int])
    xp_expansion = np.linalg.norm(l95_intersection - xp, axis=0).astype(np.double)

    # Get coordinates of the intersection between the mid-plane and the ρ100 and
    # ρ95 surfaces. Then, evaluate the distance between the two
    r_m100_int, z_m100_int = sf.rhoTheta2rz(equ, 1, theta_in=0, t_in=time_fpg,
                                            coord_in='rho_pol')
    m100_intersection = np.array([r_m100_int.flatten(), z_m100_int.flatten()])
    r_m95_int, z_m95_int = sf.rhoTheta2rz(equ, 0.95, theta_in=0,
                                          t_in=time_fpg,
                                          coord_in='rho_pol')
    m95_intersection = np.array([r_m95_int.flatten(), z_m95_int.flatten()])
    mid_expansion = np.linalg.norm(m100_intersection - m95_intersection,
                                   axis=0).astype(np.double)

    # The flux expansion ratio should be the ratio between two areas enveloping the
    # same magnetic lines but, due to the thoroidal symmetry, it can be reduced to
    # the ratio of distances between the magnetic surfaces
    f_exp = xp_expansion / mid_expansion
    # We must remove spikes due to bad magnetic reconstruction (usually flux
    # expansion is expected to be between 0 and 20)
    correction_indexes = (~np.asarray((f_exp < 20) &
                                      (f_exp > 0))).nonzero()[0]
    f_exp[correction_indexes] = (f_exp[correction_indexes + 1] +
                                 f_exp[correction_indexes - 1]) / 2

    print('Flux expansion evaluated')


    # SPECIFIC TIME POINTS SELECTION AND FILTERING FOR X_A WITH T_u
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Query for shortest time array. It will be used to evaluate sampling rate for
    # gaussian filters sigma computation and for resampling
    time_wT = sgpr.shortest_array([time_xpr, time_fpg, time_ida, time_iob])
    f_s_wT = 1 / (time_wT[int(time_wT.size / 2) + 1] -
                  time_wT[int(time_wT.size / 2)])

    # Apply gaussian filter to XPR quantities
    z_xpr_wT = ndm.gaussian_filter(z_xpr, f_s_xpr / (2 * f_s_wT))
    rho_xpr_wT = ndm.gaussian_filter(rho_xpr, f_s_xpr / (2 * f_s_wT))

    # Apply gaussian filter to FPG quantities
    R_0_wT = ndm.gaussian_filter(r_magax, f_s_fpg / (2 * f_s_wT))
    a_wT = ndm.gaussian_filter(a, f_s_fpg / (2 * f_s_wT))
    q_95_wT = ndm.gaussian_filter(q_95, f_s_fpg / (2 * f_s_wT))
    z_xp_wT = ndm.gaussian_filter(z_xp, f_s_fpg / (2 * f_s_wT))
    f_exp_wT = ndm.gaussian_filter(f_exp, f_s_fpg / (2 * f_s_wT))

    # Apply gaussian filter to IDA quantities
    T_u_wT = ndm.gaussian_filter(T_u, f_s_ida / (2 * f_s_wT))
    n_u_wT = ndm.gaussian_filter(n_u, f_s_ida / (2 * f_s_wT))

    # Apply gaussian filter to IOB quantities
    n_0_lwT = ndm.gaussian_filter(n_0_l, f_s_iob / (2 * f_s_wT))
    n_0_wT = ndm.gaussian_filter(n_0, f_s_iob / (2 * f_s_wT))
    n_0_uwT = ndm.gaussian_filter(n_0_u, f_s_iob / (2 * f_s_wT))

    # Re-slice XPR quantities
    xpr_indexes_wT, _ = sgpr.find_nearest_multiple_index(time_xpr, time_wT)
    z_xpr_wT = z_xpr_wT[xpr_indexes_wT]
    rho_xpr_wT = rho_xpr_wT[xpr_indexes_wT]
    time_xpr_wT = time_xpr[xpr_indexes_wT]

    # Re-slice FPG quantities
    fpg_indexes_wT, _ = sgpr.find_nearest_multiple_index(time_fpg, time_wT)
    R_0_wT = R_0_wT[fpg_indexes_wT]
    a_wT = a_wT[fpg_indexes_wT]
    q_95_wT = q_95_wT[fpg_indexes_wT]
    z_xp_wT = z_xp_wT[fpg_indexes_wT]
    f_exp_wT = f_exp_wT[fpg_indexes_wT]
    time_fpg_wT = time_fpg[fpg_indexes_wT]

    # Re-slice IDA quantities
    ida_indexes_wT, _ = sgpr.find_nearest_multiple_index(time_ida, time_wT)
    T_u_wT = T_u_wT[ida_indexes_wT]
    n_u_wT = n_u_wT[ida_indexes_wT]
    time_ida_wT = time_ida[ida_indexes_wT]

    # Re-slice IOB quantities
    iob_indexes_wT, _ = sgpr.find_nearest_multiple_index(time_iob, time_wT)
    n_0_lwT = n_0_lwT[iob_indexes_wT]
    n_0_wT = n_0_wT[iob_indexes_wT]
    n_0_uwT = n_0_uwT[iob_indexes_wT]
    time_iob_wT = time_iob[iob_indexes_wT]

    print('Downsampling executed')


    # EVALUATION OF X_A WITH T_u
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    X_A_lwT = R_0_wT**2 * q_95_wT**2 * f_exp_wT * n_u_wT * \
                n_0_lwT / (a_wT * T_u_wT**(5/2))
    X_A_wT = R_0_wT**2 * q_95_wT**2 * f_exp_wT * n_u_wT * \
                n_0_wT / (a_wT * T_u_wT**(5/2))
    X_A_uwT = R_0_wT**2 * q_95_wT**2 * f_exp_wT * n_u_wT * \
                n_0_uwT / (a_wT * T_u_wT**(5/2))

    print('X_A evaluated with T_u')

    # EVALUATION OF XPR RELATIVE HEIGHT WITH T_u DOWNSAMPLING
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    z_xpr_rel_wT = z_xpr_wT - z_xp_wT


    # SLICING ONLY FOR COLD CORE XPR PHASES
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Slice X_A
    cld_start_XA_wT_idx, _ = sgpr.find_nearest_multiple_index(time_wT, cld_start)
    cld_end_XA_wT_idx, _ = sgpr.find_nearest_multiple_index(time_wT, cld_end)
    cld_XA_wT_idx = np.array([idx for start, end in zip(cld_start_XA_wT_idx,
                                                        cld_end_XA_wT_idx) for idx
                              in range(start, end + 1)])
    cld_X_A_lwT = X_A_lwT[cld_XA_wT_idx]
    cld_X_A_wT = X_A_wT[cld_XA_wT_idx]
    cld_X_A_uwT = X_A_uwT[cld_XA_wT_idx]

    # Slice xpr
    cld_start_xpr_wT_idx, _ = sgpr.find_nearest_multiple_index(time_xpr_wT,
                                                               cld_start)
    cld_end_xpr_wT_idx, _ = sgpr.find_nearest_multiple_index(time_xpr_wT,
                                                             cld_end)
    cld_xpr_wT_idx = np.array([idx for start, end in zip(cld_start_xpr_wT_idx,
                                                         cld_end_xpr_wT_idx)
                               for idx in range(start, end + 1)])
    cld_z_xpr_rel_wT = z_xpr_rel_wT[cld_xpr_wT_idx]
    cld_rho_xpr_wT = rho_xpr_wT[cld_xpr_wT_idx]

    # PLOT
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ax1.errorbar(cld_X_A_wT, 1-cld_rho_xpr_wT, xerr=[cld_X_A_wT - cld_X_A_lwT,
                                                     cld_X_A_uwT - cld_X_A_wT],
                 fmt='o', label=f'{shot}')


# SHOW PLOT OR SAVE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.legend()
plt.tight_layout()
plt.show()

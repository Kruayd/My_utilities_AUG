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
import calibrators as cal


# GLOBAL VARIABLES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot sizes
WID = 2*5.512
HIG = WID / 1.618


# OPTIONS HANDLER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
parser = argparse.ArgumentParser(description='Plot time traces of edge ' +
                                 'temperature profile')
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

mag_equ_diag = args.mag_equ_diag


# PLOTTING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# General settings
style.use('bmh')
plt.figure(figsize=(WID, HIG))
plt.rc('font', size=10)
plt.rc('axes', titlesize=10)
plt.rc('axes', labelsize=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('legend', fontsize=10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Subplots
ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

ax1.set_xlabel('$X_A$')
ax1.set_ylabel('1 - $\\rho_{pol}$')


for shot in [40365, 40366, 41158]:
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
                             '_XPR_position.csv', index_col=0)
    time_xpr = df_pos_xpr.index.values
    rho_xpr = df_pos_xpr['rho pol'].values


    OVERRIDE = True

    # QUERYING SHOTS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get shot-files relative to function parametrization, equilibrium , integrated
    # data analysis and ionization manometers
    fpg = sf.SFREAD(shot, FPG_DIAG)
    equ = sf.EQU(shot, diag=mag_equ_diag)
    ida = sf.SFREAD(shot, 'IDA')
    ioc = cal.SFIOCF01(shot, interpolate=OVERRIDE)
    ioc_low = cal.SFIOCF01(shot, sensitivity=2.86, interpolate=OVERRIDE)
    ioc_upp = cal.SFIOCF01(shot, sensitivity=2.17, interpolate=OVERRIDE)
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
        T_u_matrix = ida.getobject('Te', cal=True).astype(np.double)
        T_u_low_matrix = ida.getobject('Te_lo', cal=True).astype(np.double)
        T_u_upp_matrix = ida.getobject('Te_up', cal=True).astype(np.double)
        n_u_matrix = ida.getobject('ne', cal=True).astype(np.double)
        n_u_low_matrix = ida.getobject('ne_lo', cal=True).astype(np.double)
        n_u_upp_matrix = ida.getobject('ne_up', cal=True).astype(np.double)
        area_ida = ida.getareabase('Te')
        time_ida = ida.gettimebase('Te')
        # Choosing separatrix data
        ida_idx = sgpr.find_nearest_index(area_ida, 1.0, axis=0)
        T_u = T_u_matrix[ida_idx, np.arange(time_ida.size)]
        T_u_low = T_u_low_matrix[ida_idx, np.arange(time_ida.size)]
        T_u_upp = T_u_upp_matrix[ida_idx, np.arange(time_ida.size)]
        n_u = n_u_matrix[ida_idx, np.arange(time_ida.size)]
        n_u_low = n_u_low_matrix[ida_idx, np.arange(time_ida.size)]
        n_u_upp = n_u_upp_matrix[ida_idx, np.arange(time_ida.size)]
    else:
        sys.exit('Error while loading IDA')

    if ioc.status and ioc_low.status and ioc_upp.status:
        n_0 = ioc.getobject('F01').astype(np.double)
        n_0_low = ioc_low.getobject('F_lower').astype(np.double)
        n_0_upp = ioc_upp.getobject('F_upper').astype(np.double)
        time_ioc = ioc.gettimebase()
        ioc_par = ioc.getparset('CONVERT')
        ioc_2_n = ioc_par['density']
        n_0 = n_0 * ioc_2_n
        n_0_low = n_0_low * ioc_2_n
        n_0_upp = n_0_upp * ioc_2_n
    else:
        sys.exit('Error while loading IOC')

    if tot.status:
        P_tot = tot.getobject('P_TOT', cal=True).astype(np.double)
        time_tot = tot.gettimebase('P_TOT')
    else:
        sys.exit('Error while loading TOT')

    if bpt.status:
        P_rad = bpt.getobject('Pr_sepX', cal=True).astype(np.double)
        P_rad_low = bpt.getobject('Pr_sepX-', cal=True).astype(np.double)
        P_rad_upp = bpt.getobject('Pr_sepX+', cal=True).astype(np.double)
        time_bpt = bpt.gettimebase('Pr_sepX')
    else:
        sys.exit('Error while loading BPT')

    print('\n\n')
    print('Querying: done')


    # UNIVERSAL TIME SELECTION
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get maximum common time interval boundaries
    start = max(time_fpg[0], time_ida[0], time_ioc[0], time_tot[0],
                time_bpt[0], time_xpr[0], shot_start)
    end = min(time_fpg[-1], time_ida[-1], time_ioc[-1], time_tot[-1],
              time_bpt[-1], time_xpr[-1], shot_end)

    # Get indexes of boundaries for XPR
    start_index_xpr = sgpr.find_nearest_index(time_xpr, start)
    end_index_xpr = sgpr.find_nearest_index(time_xpr, end)

    # Get indexes of boundaries for FPG
    start_index_fpg = sgpr.find_nearest_index(time_fpg, start)
    end_index_fpg = sgpr.find_nearest_index(time_fpg, end)

    # Get indexes of boundaries for IDA
    start_index_ida = sgpr.find_nearest_index(time_ida, start)
    end_index_ida = sgpr.find_nearest_index(time_ida, end)

    # Get indexes of boundaries for IOC
    start_index_ioc = sgpr.find_nearest_index(time_ioc, start)
    end_index_ioc = sgpr.find_nearest_index(time_ioc, end)

    # Get indexes of boundaries for TOT
    start_index_tot = sgpr.find_nearest_index(time_tot, start)
    end_index_tot = sgpr.find_nearest_index(time_tot, end)

    # Get indexes of boundaries for BPT
    start_index_bpt = sgpr.find_nearest_index(time_bpt, start)
    end_index_bpt = sgpr.find_nearest_index(time_bpt, end)

    # Slice XPR quantities and relative time
    rho_xpr = rho_xpr[..., start_index_xpr:end_index_xpr + 1]
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
    T_u_low = T_u_low[..., start_index_ida:end_index_ida + 1]
    T_u_upp = T_u_upp[..., start_index_ida:end_index_ida + 1]
    n_u = n_u[..., start_index_ida:end_index_ida + 1]
    n_u_low = n_u_low[..., start_index_ida:end_index_ida + 1]
    n_u_upp = n_u_upp[..., start_index_ida:end_index_ida + 1]
    time_ida = time_ida[..., start_index_ida:end_index_ida + 1]

    # Slice IOB quantities and relative time
    n_0 = n_0[..., start_index_ioc:end_index_ioc + 1]
    n_0_low = n_0_low[..., start_index_ioc:end_index_ioc + 1]
    n_0_upp = n_0_upp[..., start_index_ioc:end_index_ioc + 1]
    time_ioc = time_ioc[..., start_index_ioc:end_index_ioc + 1]

    # Slice TOT quantities and relative time
    P_tot = P_tot[..., start_index_tot:end_index_tot + 1]
    time_tot = time_tot[..., start_index_tot:end_index_tot + 1]

    # Slice BPT quantities and relative time
    P_rad = P_rad[..., start_index_bpt:end_index_bpt + 1]
    P_rad_low = P_rad_low[..., start_index_bpt:end_index_bpt + 1]
    P_rad_upp = P_rad_upp[..., start_index_bpt:end_index_bpt + 1]
    time_bpt = time_bpt[..., start_index_bpt:end_index_bpt + 1]

    print('Time selection done')


    # SAMPLING RATES
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Evaluate sampling rate for subsequent downsampling (sampling rates in the
    # middle of a time trace are more reliable than averages)
    f_s_xpr = 1 / (time_xpr[int(time_xpr.size / 2) + 1] -
                   time_xpr[int(time_xpr.size / 2)])
    f_s_fpg = 1 / (time_fpg[int(time_fpg.size / 2) + 1] -
                   time_fpg[int(time_fpg.size / 2)])
    f_s_ida = 1 / (time_ida[int(time_ida.size / 2) + 1] -
                   time_ida[int(time_ida.size / 2)])
    f_s_ioc = 1 / (time_ioc[int(time_ioc.size / 2) + 1] -
                   time_ioc[int(time_ioc.size / 2)])
    f_s_tot = 1 / (time_tot[int(time_tot.size / 2) + 1] -
                   time_tot[int(time_tot.size / 2)])
    f_s_bpt = 1 / (time_bpt[int(time_bpt.size / 2) + 1] -
                   time_bpt[int(time_bpt.size / 2)])

    print('Sampling rates evaluated')


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
    time_wT = sgpr.shortest_array([time_xpr, time_fpg, time_ida, time_ioc])
    f_s_wT = 1 / (time_wT[int(time_wT.size / 2) + 1] -
                  time_wT[int(time_wT.size / 2)])

    # Apply gaussian filter to XPR quantities
    rho_xpr_wT = ndm.gaussian_filter(rho_xpr, f_s_xpr / (2 * f_s_wT))

    # Apply gaussian filter to FPG quantities
    R_0_wT = ndm.gaussian_filter(r_magax, f_s_fpg / (2 * f_s_wT))
    a_wT = ndm.gaussian_filter(a, f_s_fpg / (2 * f_s_wT))
    q_95_wT = ndm.gaussian_filter(q_95, f_s_fpg / (2 * f_s_wT))
    f_exp_wT = ndm.gaussian_filter(f_exp, f_s_fpg / (2 * f_s_wT))

    # Apply gaussian filter to IDA quantities
    T_u_wT = ndm.gaussian_filter(T_u, f_s_ida / (2 * f_s_wT))
    T_u_low_wT = ndm.gaussian_filter(T_u_low, f_s_ida / (2 * f_s_wT))
    T_u_upp_wT = ndm.gaussian_filter(T_u_upp, f_s_ida / (2 * f_s_wT))
    n_u_wT = ndm.gaussian_filter(n_u, f_s_ida / (2 * f_s_wT))
    n_u_low_wT = ndm.gaussian_filter(n_u_low, f_s_ida / (2 * f_s_wT))
    n_u_upp_wT = ndm.gaussian_filter(n_u_upp, f_s_ida / (2 * f_s_wT))

    # Apply gaussian filter to IOB quantities
    n_0_wT = ndm.gaussian_filter(n_0, f_s_ioc / (2 * f_s_wT))
    n_0_low_wT = ndm.gaussian_filter(n_0_low, f_s_ioc / (2 * f_s_wT))
    n_0_upp_wT = ndm.gaussian_filter(n_0_upp, f_s_ioc / (2 * f_s_wT))

    # Re-slice XPR quantities
    xpr_indexes_wT, _ = sgpr.find_nearest_multiple_index(time_xpr, time_wT)
    rho_xpr_wT = rho_xpr_wT[xpr_indexes_wT]
    time_xpr_wT = time_xpr[xpr_indexes_wT]

    # Re-slice FPG quantities
    fpg_indexes_wT, _ = sgpr.find_nearest_multiple_index(time_fpg, time_wT)
    R_0_wT = R_0_wT[fpg_indexes_wT]
    a_wT = a_wT[fpg_indexes_wT]
    q_95_wT = q_95_wT[fpg_indexes_wT]
    f_exp_wT = f_exp_wT[fpg_indexes_wT]
    time_fpg_wT = time_fpg[fpg_indexes_wT]

    # Re-slice IDA quantities
    ida_indexes_wT, _ = sgpr.find_nearest_multiple_index(time_ida, time_wT)
    T_u_wT = T_u_wT[ida_indexes_wT]
    T_u_low_wT = T_u_low_wT[ida_indexes_wT]
    T_u_upp_wT = T_u_upp_wT[ida_indexes_wT]
    n_u_wT = n_u_wT[ida_indexes_wT]
    n_u_low_wT = n_u_low_wT[ida_indexes_wT]
    n_u_upp_wT = n_u_upp_wT[ida_indexes_wT]
    time_ida_wT = time_ida[ida_indexes_wT]

    # Re-slice IOB quantities
    ioc_indexes_wT, _ = sgpr.find_nearest_multiple_index(time_ioc, time_wT)
    n_0_wT = n_0_wT[ioc_indexes_wT]
    n_0_low_wT = n_0_low_wT[ioc_indexes_wT]
    n_0_upp_wT = n_0_upp_wT[ioc_indexes_wT]
    time_ioc_wT = time_ioc[ioc_indexes_wT]

    print('Downsampling executed')


    # EVALUATION OF X_A WITH T_u
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    X_A_wT = R_0_wT**2 * q_95_wT**2 * f_exp_wT * n_u_wT * \
            n_0_wT / (a_wT * T_u_wT**(5/2))
    X_A_low_wT = R_0_wT**2 * q_95_wT**2 * f_exp_wT * n_u_low_wT * \
            n_0_low_wT / (a_wT * T_u_upp_wT**(5/2))
    X_A_upp_wT = R_0_wT**2 * q_95_wT**2 * f_exp_wT * n_u_upp_wT * \
            n_0_upp_wT / (a_wT * T_u_low_wT**(5/2))

    print('X_A evaluated with T_u')


    # SLICING ONLY FOR COLD CORE XPR PHASES
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Slice X_A
    cld_start_XA_wT_idx, _ = sgpr.find_nearest_multiple_index(time_wT, cld_start)
    cld_end_XA_wT_idx, _ = sgpr.find_nearest_multiple_index(time_wT, cld_end)
    cld_XA_wT_idx = np.array([idx for start, end in zip(cld_start_XA_wT_idx,
                                                        cld_end_XA_wT_idx) for idx
                              in range(start, end + 1)])
    cld_X_A_lwT = X_A_low_wT[cld_XA_wT_idx]
    cld_X_A_wT = X_A_wT[cld_XA_wT_idx]
    cld_X_A_uwT = X_A_upp_wT[cld_XA_wT_idx]

    # Slice xpr
    cld_start_xpr_wT_idx, _ = sgpr.find_nearest_multiple_index(time_xpr_wT,
                                                               cld_start)
    cld_end_xpr_wT_idx, _ = sgpr.find_nearest_multiple_index(time_xpr_wT,
                                                             cld_end)
    cld_xpr_wT_idx = np.array([idx for start, end in zip(cld_start_xpr_wT_idx,
                                                         cld_end_xpr_wT_idx)
                               for idx in range(start, end + 1)])
    cld_rho_xpr_wT = rho_xpr_wT[cld_xpr_wT_idx]

    # PLOT
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ax1.errorbar(cld_X_A_wT, 1-cld_rho_xpr_wT,
                 fmt='o', label=f'{shot}')


# SHOW PLOT OR SAVE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.legend()
plt.tight_layout()
plt.show()

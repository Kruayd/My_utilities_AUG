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
import argparse
import pandas as pd
import numpy as np
import aug_sfutils as sf
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.widgets import Slider

import sig_proc as sgpr


# OPTIONS HANDLER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
parser = argparse.ArgumentParser(description='Plot time traces of edge ' +
                                 'temperature profile')
parser.add_argument('shot', metavar='SHOT_NUMBER', type=int,
                    help='Execute the code for shot #SHOT_NUMBER')
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

shot = args.shot
mag_equ_diag = args.mag_equ_diag


# READING SHOT TIME BOUNDARIES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
df_shot_time = pd.read_csv('./csvs/equilibria.csv', index_col=0).loc[shot]

shot_start = df_shot_time['start']
shot_end = df_shot_time['end']


# QUERYING SHOTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get shot-files relative to Charge exchange diagnostic and equilibrium
ces = sf.SFREAD(shot, 'CES')
equ = sf.EQU(shot, diag=mag_equ_diag)


# QUERYING SIGNALS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get calibrated time traces for impurity density of CES
if ces.status:
    n_imp = ces.getobject('nimp', cal=True).transpose()[:24]
    R_ces = ces.getobject('R')[:24]
    z_ces = ces.getobject('z')[:24]
    time_ces = ces.gettimebase('nimp')
    area_ces = list()
    for index, time in enumerate(time_ces):
        area_ces.append(sf.rz2rho(equ, R_ces[..., index], z_ces[..., index],
                                  t_in=time, coord_out='rho_pol').flatten())
    area_ces = np.array(area_ces).transpose()
else:
    sys.exit('Error while laoding CES')


# # General settings
# style.use('ggplot')
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# px = 1/plt.rcParams['figure.dpi']  # from pixel to inches
# fig0 = plt.figure(0, figsize=(1600*px, 1000*px))
# plt.suptitle(f'SHOT #{shot}', fontsize=32, fontweight='bold')
# 
# # Subplots
# ax1 = plt.subplot2grid((25, 3), (0, 0), rowspan=19, colspan=3)
# ax2 = plt.subplot2grid((25, 3), (22, 0), rowspan=1, colspan=2)
# ax3 = plt.subplot2grid((25, 3), (20, 2), rowspan=5, colspan=1)
# 
# # Main plot
# initial_index = int(time_ces.size / 2)
# ax1.set_xlim(0, 1.1)
# ax1.set_ylim(0, 1.1*n_imp.max())
# ax1.set_xlabel(r'$\rho_{pol}$')
# ax1.set_ylabel('atoms m$^{-3}$')
# profile, = ax1.plot(area_ces[..., initial_index], n_imp[..., initial_index])
# 
# # Text plot
# time_stamp = ax3.text(0.5, 0.5, f'Time: {time_ces[initial_index]:.3f} s',
#                       fontsize=32, fontweight='bold', ha='center', va='center',
#                       transform=ax3.transAxes)
# ax3.set_facecolor('white')
# ax3.get_xaxis().set_visible(False)
# ax3.get_yaxis().set_visible(False)
# 
# # Slider
# t_slider = Slider(ax=ax2, label='time index', valmin=0, valmax=time_ces.size-1,
#                   valinit=initial_index)
# 
# 
# def update_slider(val):
#     profile.set_data(area_ces[..., int(t_slider.val)], n_imp[...,
#                                                              int(t_slider.val)])
#     time_stamp.set_text(f'Time: {time_ces[int(t_slider.val)]:.3f} s')
# 
# 
# t_slider.on_changed(update_slider)
# 
# 
# # Other plot
# fig1 = plt.figure(1, figsize=(1600*px, 1000*px))
# plt.suptitle(f'SHOT #{shot}', fontsize=32, fontweight='bold')

# Slicing at rho_pol 0.2, 0.5 and 0.8
ces_index_core = sgpr.find_nearest_index(area_ces, 0.2, axis=0)
ces_index_medi = sgpr.find_nearest_index(area_ces, 0.5, axis=0)
ces_index_edge = sgpr.find_nearest_index(area_ces, 0.8, axis=0)
n_imp_core = n_imp[ces_index_core, np.arange(time_ces.size)]
n_imp_medi = n_imp[ces_index_medi, np.arange(time_ces.size)]
n_imp_edge = n_imp[ces_index_edge, np.arange(time_ces.size)]

# # Subplots
# ax4 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
# ax4.plot(time_ces, n_imp_core, label=r'at $\rho_{pol}=0.2$')
# ax4.plot(time_ces, n_imp_medi, label=r'at $\rho_{pol}=0.5$')
# ax4.plot(time_ces, n_imp_edge, label=r'at $\rho_{pol}=0.8$')
# ax4.set_title(r'Impurities time traces at different $\rho_{pol}$')
# ax4.set_xlabel('s')
# ax4.set_ylabel('atoms m$^{-3}$')
# ax4.legend()
# 
# # Show plot or save
# plt.tight_layout()
# plt.show()

WID = 5.512
HIG = WID / 1.618
X_LABEL = r'Time (s)'
Y_LABEL = r'Nitrogen impurity density (m$^{-3}$)'


plt.style.use('bmh')
plt.figure(figsize=(WID, HIG))
plt.rc('font', size=10)
plt.rc('axes', titlesize=10)
plt.rc('axes', labelsize=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('legend', fontsize=10)


plt.title(f'Shot #{shot}', loc='right')
plt.plot(time_ces, n_imp_core)
plt.plot(time_ces, n_imp_medi)
plt.plot(time_ces, n_imp_edge)
plt.xlabel(X_LABEL)
plt.ylabel(Y_LABEL)


# Show plot or save
plt.tight_layout(pad=0.1)
plt.show()

#!/usr/bin/python3


import numpy as np
import matplotlib.pyplot as plt


# GLOBAL VARIABLES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot sizes
WID = 5.512
HIG = 3*WID / 1.618


index = [1, 2, 3]
labels = ['40365', '40366', '41158']

open_xa = np.array([0.2e39, 0.4e39, 1.0e39])
close_xa = np.array([0.6e39, 0.5e39, 2.0e39])
high_xa = np.array([1.0e39, 2.5e39, 6.0e39])
low_xa = np.array([0.1e39, 0.1e39, 0.4e39])

open_xae = np.array([2.5e45, 1.2e45, 2.0e45])
close_xae = np.array([3.3e45, 1.4e45, 3.0e45])
high_xae = np.array([4.0e45, 1.8e45, 3.4e45])
low_xae = np.array([2.0e45, 0.8e45, 1.7e45])

open_xar = np.array([1.5e38, 1.3e38, 1.0e38])
close_xar = np.array([1.6e38, 1.5e38, 1.2e38])
high_xar = np.array([1.7e38, 1.9e38, 1.3e38])
low_xar = np.array([1.4e38, 1.0e38, 0.9e38])


# PLOTTING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# General settings
plt.style.use('bmh')
plt.figure(figsize=(WID, HIG))
plt.rc('font', size=10)
plt.rc('axes', titlesize=10)
plt.rc('axes', labelsize=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('legend', fontsize=10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


width = .3
width2 = .01


# Subplots
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=1, colspan=1)
ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=1, colspan=1)
ax3 = plt.subplot2grid((3, 1), (2, 0), rowspan=1, colspan=1)

# Access parameter sublpot with T_u
ax1.bar(index, close_xa-open_xa, width, bottom=open_xa, color=colors[0])
ax1.bar(index, high_xa-close_xa, width2, bottom=close_xa, color=colors[0])
ax1.bar(index, low_xa-open_xa, width2, bottom=open_xa, color=colors[0])
ax1.set_xlabel('Shot')
ax1.set_ylabel(r'$X_A$')
ax1.set_xticks(index, labels)

# Access parameter sublpot with P_sep
ax2.bar(index, close_xae-open_xae, width, bottom=open_xae, color=colors[0])
ax2.bar(index, high_xae-close_xae, width2, bottom=close_xae, color=colors[0])
ax2.bar(index, low_xae-open_xae, width2, bottom=open_xae, color=colors[0])
ax2.set_xlabel('Shot')
ax2.set_ylabel(r'$X_{A,eng}$')
ax2.set_xticks(index, labels)

# Access parameter sublpot with reconstructed parameters
ax3.bar(index, close_xar-open_xar, width, bottom=open_xar, color=colors[0])
ax3.bar(index, high_xar-close_xar, width2, bottom=close_xar, color=colors[0])
ax3.bar(index, low_xar-open_xar, width2, bottom=open_xar, color=colors[0])
ax3.set_xlabel('Shot')
ax3.set_ylabel(r'$X_{A,tpm}$')
ax3.set_xticks(index, labels)

# Show plot or save
plt.tight_layout()
plt.show()

#!/usr/bin/python3


import numpy as np
import matplotlib.pyplot as plt


# GLOBAL VARIABLES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot sizes
WID = 5.512
HIG = 3*WID / 1.618


index = [1, 2, 3, 4]
labels = ['40365', '40366', '41137', '41158']

val_xa = np.array([1e38, 4e36, 0.9e36, 0.4e37])
upp_xa = np.array([1.5e38, 8e36, 2.7e36, 1e37])
upp_xa = upp_xa - val_xa
low_xa = np.array([0.5e38, 2e36, 0.4e36, 0.2e37])
low_xa = val_xa - low_xa

val_xae = np.array([1.8e45, 1.2e43, 4e43, 4.0e43])
upp_xae = np.array([2.0e45, 1.9e43, 6e43, 6.0e43])
upp_xae = upp_xae - val_xae
low_xae = np.array([1.5e45, 0.8e43, 2e43, 2.0e43])
low_xae = val_xae - low_xae

val_xar = np.array([1.27e38, 1.0e37, 0.7e37, 1.0e37])
upp_xar = np.array([1.30e38, 1.3e37, 1.0e37, 1.3e37])
upp_xar = upp_xar - val_xar
low_xar = np.array([1.24e38, 0.7e37, 0.6e37, 0.8e37])
low_xar = val_xar - low_xar


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


# Subplots
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=1, colspan=1)
ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=1, colspan=1)
ax3 = plt.subplot2grid((3, 1), (2, 0), rowspan=1, colspan=1)

# Access parameter sublpot with T_u
ax1.errorbar(index, val_xa, yerr=(low_xa, upp_xa), fmt='o')
ax1.set_xlabel('Shot')
ax1.set_ylabel(r'$X_A$')
ax1.set_xticks(index, labels)

# Access parameter sublpot with P_sep
ax2.errorbar(index, val_xae, yerr=(low_xae, upp_xae), fmt='o')
ax2.set_xlabel('Shot')
ax2.set_ylabel(r'$X_{A,eng}$')
ax2.set_xticks(index, labels)

# Access parameter sublpot with reconstructed parameters
ax3.errorbar(index, val_xar, yerr=(low_xar, upp_xar), fmt='o')
ax3.set_xlabel('Shot')
ax3.set_ylabel(r'$X_{A,tpm}$')
ax3.set_xticks(index, labels)

# Show plot or save
plt.tight_layout()
plt.show()

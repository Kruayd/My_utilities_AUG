import sys
import pandas as pd
import numpy as np
import aug_sfutils as sf
import matplotlib.pyplot as plt

import sig_proc as sgpr
import manual_calibrators as manc


WID = 5.512
HIG = WID / 1.618
DEPTH = 50
X_LABEL = r'Time (s)'
Y_LABEL = r'Neutral density (m$^{-3}$)'

SHOT = 40365


df_shot_time = pd.read_csv('./csvs/equilibria.csv', index_col=0).loc[SHOT]

shot_start = df_shot_time['start']
shot_end = df_shot_time['end']


ioc_raw = sf.SFREAD(SHOT, 'IOC')

ioc_low = manc.SFIOCF01(SHOT, sensitivity=3.)
ioc = manc.SFIOCF01(SHOT)
ioc_upp = manc.SFIOCF01(SHOT, sensitivity=2.)


if ioc_raw.status:
    raw_n = ioc_raw.getobject('F01', cal=True).astype(np.double)
    time_raw = ioc_raw.gettimebase('F01').astype(np.double)
    raw_n_upp = ioc_raw.getobject('F_upper', cal=True)
    raw_n_upp = raw_n_upp[..., 0].astype(np.double)
    raw_n_low = ioc_raw.getobject('F_lower', cal=True)
    raw_n_low = raw_n_low[..., 0].astype(np.double)
    ioc_par = ioc_raw.getparset('CONVERT')
    ioc_2_n = ioc_par['density']
    raw_n_low = raw_n_low * ioc_2_n
    raw_n = raw_n * ioc_2_n
    raw_n_upp = raw_n_upp * ioc_2_n
else:
    sys.exit('Error while loading IOC')

if ioc_low.status and ioc.status and ioc_upp.status:
    n_0_l = ioc_low.getobject('F_lower').astype(np.double)
    n_0 = ioc.getobject('F01').astype(np.double)
    n_0_u = ioc_upp.getobject('F_upper').astype(np.double)
    time_ioc = ioc.gettimebase()
    n_0_l = n_0_l * ioc_2_n
    n_0 = n_0 * ioc_2_n
    n_0_u = n_0_u * ioc_2_n
else:
    sys.exit('Error while loading IOB')


start_index_ioc = sgpr.find_nearest_index(time_ioc, shot_start)
end_index_ioc = sgpr.find_nearest_index(time_ioc, shot_end)


raw_n_low = raw_n_low[..., start_index_ioc:end_index_ioc + 1]
raw_n = raw_n[..., start_index_ioc:end_index_ioc + 1]
raw_n_upp = raw_n_upp[..., start_index_ioc:end_index_ioc + 1]
time_raw = time_raw[..., start_index_ioc:end_index_ioc + 1]
n_0_l = n_0_l[..., start_index_ioc:end_index_ioc + 1]
n_0 = n_0[..., start_index_ioc:end_index_ioc + 1]
n_0_u = n_0_u[..., start_index_ioc:end_index_ioc + 1]
time_ioc = time_ioc[..., start_index_ioc:end_index_ioc + 1]


plt.style.use('bmh')
plt.figure(figsize=(WID, HIG))
plt.rc('font', size=10)
plt.rc('axes', titlesize=10)
plt.rc('axes', labelsize=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('legend', fontsize=10)


# DLX subplot
plt.title(f'Shot #{SHOT}', loc='right')
plt.plot(time_raw, raw_n)
plt.fill_between(time_raw, raw_n_low, raw_n_upp, alpha=0.2)
plt.plot(time_ioc, n_0)
plt.fill_between(time_ioc, n_0_l, n_0_u, alpha=0.2)
plt.xlabel(X_LABEL)
plt.ylabel(Y_LABEL)

plt.tight_layout(pad=0.1)
plt.show()

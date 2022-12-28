import sys
import pandas as pd
import numpy as np
import aug_sfutils as sf
import matplotlib.pyplot as plt

import sig_proc as sgpr
import calibrators as cal


WID = 5.512
HIG = 2*WID / 1.618
DEPTH = 50
X_LABEL = r'Time (s)'
Y_LABEL = r'Neutral density (m$^{-3}$)'

SHOT = 40365


df_shot_time = pd.read_csv('./csvs/equilibria.csv', index_col=0).loc[SHOT]

shot_start = df_shot_time['start']
shot_end = df_shot_time['end']


uvs = sf.SFREAD(SHOT, 'UVS')
ioc_raw = sf.SFREAD(SHOT, 'IOC')
ioc = cal.SFIOCF01(SHOT)


if uvs.status:
    d_puff = uvs.getobject('D_tot', cal=True).astype(np.double)
    n_puff = uvs.getobject('N_tot', cal=True).astype(np.double)
    time_uvs = uvs.gettimebase('D_tot').astype(np.double)
else:
    sys.exit('Error while loading UVS')

if ioc_raw.status:
    raw_n = ioc_raw.getobject('F01', cal=True).astype(np.double)
    time_raw = ioc_raw.gettimebase('F01').astype(np.double)
    ioc_par = ioc_raw.getparset('CONVERT')
    ioc_2_n = ioc_par['density']
    raw_n = raw_n * ioc_2_n
else:
    sys.exit('Error while loading IOC')

if ioc.status:
    background = ioc.bg_fit
    confidence = ioc.bg_std
    n_0 = ioc.getobject('F01').astype(np.double)
    time_ioc = ioc.gettimebase()
    background = background * ioc_2_n
    confidence = confidence * ioc_2_n
    n_0 = n_0 * ioc_2_n
else:
    sys.exit('Error while loading calibrated IOC')


start_index_uvs = sgpr.find_nearest_index(time_uvs, shot_start)
end_index_uvs = sgpr.find_nearest_index(time_uvs, shot_end)

start_index_ioc = sgpr.find_nearest_index(time_ioc, shot_start)
end_index_ioc = sgpr.find_nearest_index(time_ioc, shot_end)


d_puff = d_puff[..., start_index_uvs:end_index_uvs + 1]
n_puff = n_puff[..., start_index_uvs:end_index_uvs + 1]
time_uvs = time_uvs[..., start_index_uvs:end_index_uvs + 1]

raw_n = raw_n[..., start_index_ioc:end_index_ioc + 1]
time_raw = time_raw[..., start_index_ioc:end_index_ioc + 1]
background = background[..., start_index_ioc:end_index_ioc + 1]
confidence = confidence[..., start_index_ioc:end_index_ioc + 1]
n_0 = n_0[..., start_index_ioc:end_index_ioc + 1]
time_ioc = time_ioc[..., start_index_ioc:end_index_ioc + 1]


plt.style.use('bmh')
plt.figure(figsize=(WID, HIG))
plt.rc('font', size=10)
plt.rc('axes', titlesize=10)
plt.rc('axes', labelsize=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('legend', fontsize=10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


ax1 = plt.subplot2grid((2, 1), (0, 0), rowspan=1, colspan=1)
ax2 = plt.subplot2grid((2, 1), (1, 0), rowspan=1, colspan=1)


ax1.set_title(f'Shot #{SHOT}', loc='right')
ax1.plot(time_uvs, d_puff, '--')
ax1.plot(time_uvs, n_puff)
ax1.set_xlabel(X_LABEL)
ax1.set_ylabel(r'Gas puff (electrons s$^{-1}$)')

ax2.plot(time_raw, raw_n)
ax2.plot(time_ioc, background, '--')
ax2.fill_between(time_ioc, 0, background + 2*confidence, color=colors[1],
                 alpha=0.2)
ax2.plot(time_ioc, n_0, '-.')
ax2.set_xlabel(X_LABEL)
ax2.set_ylabel(Y_LABEL)


plt.tight_layout(pad=0.1)
plt.show()

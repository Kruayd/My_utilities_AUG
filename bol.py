import sys
import numpy as np
import pandas as pd
import aug_sfutils as sf
import matplotlib.pyplot as plt

import sig_proc as sgpr


WID = 5.512
HIG = 2 * WID / 1.618
DEPTH = 50
X_LABEL = r'Time (s)'
Y_LABEL = r'DLX line of sight'
C_LABEL = r'Radiation flux (W m$^{-2}$)'

SHOT = 40366
TIME_WINDOW = 50.
F_S_BOL = 500.


df_shot_time = pd.read_csv('./csvs/equilibria.csv', index_col=0).loc[SHOT]

shot_start = df_shot_time['start']
shot_end = df_shot_time['end']


blc = sf.SFREAD(sf.previousshot('BLC', SHOT),
                'BLC')
xvt = sf.SFREAD(SHOT, 'XVT')


if blc.status:
    dlx_par = blc.getparset('DLX')
else:
    sys.exit('Error while loading BLC')

if xvt.status:
    dlx_arr = list()
    for i in range(0, 12):
        signal = 'S4L0A'+str(i).zfill(2)
        dlx_arr.append(xvt.getobject(signal, cal=True))
    dlx = np.array(dlx_arr)*100
    time_dlx = xvt.gettimebase('S4L0A00')
else:
    sys.exit('Error while laoding XVT')


dlx_sights = np.arange(1, 13) * dlx_par['active'][:12]
dlx_sights = dlx_sights[dlx_sights != 0]
dlx = dlx[dlx_sights - 1]

dlx = (dlx.transpose() - np.mean(dlx[:, :10000], 1)).transpose()


start = max(time_dlx[0], shot_start)
end = min(time_dlx[-1], shot_end)

start_index_dlx = sgpr.find_nearest_index(time_dlx, start)
end_index_dlx = sgpr.find_nearest_index(time_dlx, end)

dlx = dlx[:, start_index_dlx:end_index_dlx+1]
time_dlx = time_dlx[start_index_dlx:end_index_dlx+1]


time_dlx_flt, dlx_flt = sgpr.median_filter(F_S_BOL, TIME_WINDOW, time_dlx, dlx)


start_index_dlx_zoom = sgpr.find_nearest_index(time_dlx, 4.469)
end_index_dlx_zoom = sgpr.find_nearest_index(time_dlx, 4.49)

dlx_zoom = dlx[:, start_index_dlx_zoom:end_index_dlx_zoom+1]
time_dlx_zoom = time_dlx[start_index_dlx_zoom:end_index_dlx_zoom+1]

start_index_flt_zoom = sgpr.find_nearest_index(time_dlx_flt, 4.469)
end_index_flt_zoom = sgpr.find_nearest_index(time_dlx_flt, 4.49)

flt_zoom = dlx_flt[:, start_index_flt_zoom:end_index_flt_zoom+1]
time_flt_zoom = time_dlx_flt[start_index_flt_zoom:end_index_flt_zoom+1]


plt.style.use('bmh')
plt.figure(figsize=(WID, HIG))
plt.rc('font', size=10)
plt.rc('axes', titlesize=10)
plt.rc('axes', labelsize=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('legend', fontsize=10)


ax1 = plt.subplot2grid((2, 1), (0, 0), rowspan=1, colspan=1)
ax2 = plt.subplot2grid((2, 1), (1, 0), rowspan=1, colspan=1)


# DLX subplot
# plt.title(f'Shot #{SHOT}', loc='right')
ax1.set_title(f'Shot #{SHOT}', loc='right')
ax1.plot(time_dlx, dlx[5])
ax1.plot(time_dlx_flt, dlx_flt[5], '--')
ax1.set_xlabel(X_LABEL)
ax1.set_ylabel(C_LABEL)
ax2.plot(time_dlx_zoom, dlx_zoom[5])
ax2.plot(time_flt_zoom, flt_zoom[5], '--')
ax2.set_xlabel(X_LABEL)
ax2.set_ylabel(C_LABEL)
# plt.contourf(time_dlx, dlx_sights, dlx, DEPTH, cmap='inferno')
# plt.xlabel(X_LABEL)
# plt.ylabel(Y_LABEL)
# plt.colorbar(label=C_LABEL)

plt.tight_layout(pad=0.1)
plt.show()

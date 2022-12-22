import sys
import aug_sfutils as sf
import matplotlib.pyplot as plt

import sig_proc as sgpr


WID = 5.512
HIG = 2 * WID / 1.618
X_LABEL = r'Time (s)'

SHOT = 41137


xvs = sf.SFREAD(SHOT, 'XVS')


if xvs.status:
    ddc = xvs.getobject('S2L0A15', cal=True)
    time_ddc = xvs.gettimebase('S2L0A15')
else:
    sys.exit('Error while laoding XVS')


start = max(time_ddc[0], 3)
end = min(time_ddc[-1], 8)

start_index_ddc = sgpr.find_nearest_index(time_ddc, start)
end_index_ddc = sgpr.find_nearest_index(time_ddc, end)

ddc = ddc[start_index_ddc:end_index_ddc+1]
time_ddc = time_ddc[start_index_ddc:end_index_ddc+1]

start_index_ddc_zoom = sgpr.find_nearest_index(time_ddc, 7.05)
end_index_ddc_zoom = sgpr.find_nearest_index(time_ddc, 7.25)

ddc_zoom = ddc[start_index_ddc_zoom:end_index_ddc_zoom+1]
time_ddc_zoom = time_ddc[start_index_ddc_zoom:end_index_ddc_zoom+1]


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
ax1.plot(time_ddc, ddc)
ax1.set_xlabel(X_LABEL)
ax1.set_ylabel(r'Radiation flux (W m$^{-2}$)')

ax2.plot(time_ddc_zoom, ddc_zoom)
ax2.set_xlabel(X_LABEL)
ax2.set_ylabel(r'Radiation flux (W m$^{-2}$)')

plt.tight_layout(pad=0.1)
plt.show()

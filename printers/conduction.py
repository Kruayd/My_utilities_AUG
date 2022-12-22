import sys
import numpy as np
import aug_sfutils as sf
import matplotlib.pyplot as plt

import sig_proc as sgpr


WID = 5.512
HIG = 2 * WID / 1.618
X_LABEL = r'Time (s)'

SHOT = 38773


ida = sf.SFREAD(SHOT, 'IDA')
dtn = sf.SFREAD(SHOT, 'DTN')


if ida.status:
    T_u_matrix = ida.getobject('Te', cal=True)
    area_ida = ida.getareabase('Te')
    time_ida = ida.gettimebase('Te')
    ida_idx = sgpr.find_nearest_index(area_ida, 1.0, axis=0)
    T_u = T_u_matrix[ida_idx, np.arange(time_ida.size)]
    T_u_low = T_u_matrix[ida_idx + 2, np.arange(time_ida.size)]
    T_u_upp = T_u_matrix[ida_idx - 1, np.arange(time_ida.size)]
else:
    sys.exit('Error while laoding IDA')

if dtn.status:
    Te = dtn.getobject('Te_ld', cal=True)
    Te = Te.transpose()
    time_Te = dtn.gettimebase('Te_ld')
else:
    sys.exit('Error while laoding DTN')


start = max(time_ida[0], time_Te[0])
end = min(time_ida[-1], time_Te[-1])

start_index_ida = sgpr.find_nearest_index(time_ida, start)
end_index_ida = sgpr.find_nearest_index(time_ida, end)

start_index_Te = sgpr.find_nearest_index(time_Te, start)
end_index_Te = sgpr.find_nearest_index(time_Te, end)

T_u = T_u[start_index_ida:end_index_ida+1]
T_u_low = T_u_low[start_index_ida:end_index_ida+1]
T_u_upp = T_u_upp[start_index_ida:end_index_ida+1]
time_ida = time_ida[start_index_ida:end_index_ida+1]

Te = Te[..., start_index_Te:end_index_Te+1]
time_Te = time_Te[start_index_Te:end_index_Te+1]


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
ax1.plot(time_ida, T_u)
ax1.fill_between(time_ida, T_u_low, T_u_upp, alpha=0.2)
ax1.set_xlabel(X_LABEL)
ax1.set_ylabel(r'Electron temperature (eV)')

ax2.plot(time_Te, Te[9])
ax2.plot(time_Te, Te[10])
ax2.plot(time_Te, Te[11])
ax2.set_xlabel(X_LABEL)
ax2.set_ylabel(r'Electron temperature (eV)')

plt.tight_layout(pad=0.1)
plt.show()

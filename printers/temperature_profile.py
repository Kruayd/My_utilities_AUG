import sys
import numpy as np
import aug_sfutils as sf
import matplotlib.pyplot as plt


WID = 5.512
HIG = WID / 1.618
X_LABEL = r'$\rho_{pol}$'
Y_LABEL = r'Electron temperature (ev)'

SHOT = 41137


ida = sf.SFREAD(SHOT, 'IDA')


if ida.status:
    T_u = ida.getobject('Te', cal=True).astype(np.double)
    area_ida = ida.getareabase('Te')
    time_ida = ida.gettimebase('Te')
    # Choosing data in the middle of the shot
    T_u = T_u[..., int(time_ida.size / 2)]
    area_ida = area_ida[..., int(time_ida.size / 2)]
    time_ida = time_ida[..., int(time_ida.size / 2)]
else:
    sys.exit('Error while loading IDA')


plt.style.use('bmh')
plt.figure(figsize=(WID, HIG))
plt.rc('font', size=10)
plt.rc('axes', titlesize=10)
plt.rc('axes', labelsize=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('legend', fontsize=10)


# DLX subplot
plt.xlim([0.89, 1.11])
plt.title(f't={time_ida} s, Shot #{SHOT}', loc='right')
plt.plot(area_ida, T_u, marker='+')
plt.xlabel(X_LABEL)
plt.ylabel(Y_LABEL)

plt.tight_layout(pad=0.1)
plt.show()

import sys
import numpy as np
import aug_sfutils as sf
import matplotlib.pyplot as plt


WID = 5.512
HIG = WID / 1.618
X_LABEL = r'Time (s)'
Y_LABEL = r'Radiated power (W)'

SHOT = 41158


bpt = sf.SFREAD(SHOT, 'BPT', exp='DAVIDP')


if bpt.status:
    P_rad = bpt.getobject('Pr_sep', cal=True).astype(np.double)
    P_rad_low = bpt.getobject('Pr_sep-', cal=True).astype(np.double)
    P_rad_upp = bpt.getobject('Pr_sep+', cal=True).astype(np.double)
    P_radX = bpt.getobject('Pr_sepX', cal=True).astype(np.double)
    P_radX_low = bpt.getobject('Pr_sepX-', cal=True).astype(np.double)
    P_radX_upp = bpt.getobject('Pr_sepX+', cal=True).astype(np.double)
    time_bpt = bpt.gettimebase('Pr_sepX')
else:
    sys.exit('Error while loading BPT')


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
plt.plot(time_bpt, P_rad, '-.')
plt.fill_between(time_bpt, P_rad_low, P_rad_upp, alpha=0.2)
plt.plot(time_bpt, P_radX, '--')
plt.fill_between(time_bpt, P_radX_low, P_radX_upp, alpha=0.2)
plt.xlabel(X_LABEL)
plt.ylabel(Y_LABEL)

plt.tight_layout(pad=0.1)
plt.show()

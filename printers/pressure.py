import sys
import numpy as np
import aug_sfutils as sf
import matplotlib.pyplot as plt

import manual_calibrators as manc


WID = 5.512
HIG = WID / 1.618
DEPTH = 50
X_LABEL = r'Time (s)'
Y_LABEL = r'Neutral density (m$^{-3}$)'

SHOT = 40365


ioc_raw = sf.SFREAD(SHOT, 'IOC')

ioc_low = manc.SFIOCF01(SHOT, sensitivity=3.)
ioc = manc.SFIOCF01(SHOT)
ioc_upp = manc.SFIOCF01(SHOT, sensitivity=2.)


if ioc_raw.status:
    raw_f = ioc_raw.getobject('F01', cal=True).astype(np.double)
    time_raw = ioc_raw.gettimebase('F01').astype(np.double)
    raw_f_upp = ioc_raw.getobject('F_upper', cal=True)
    raw_f_upp = raw_f_upp[..., 0].astype(np.double)
    raw_f_low = ioc_raw.getobject('F_lower', cal=True)
    raw_f_low = raw_f_low[..., 0].astype(np.double)
    ioc_par = ioc_raw.getparset('CONVERT')
    ioc_2_n = ioc_par['density']
else:
    sys.exit('Error while loading IOC')

if ioc_low.status and ioc.status and ioc_upp.status:
    n_0_l = ioc_low.getobject('F01').astype(np.double)
    n_0 = ioc.getobject('F01').astype(np.double)
    n_0_u = ioc_upp.getobject('F01').astype(np.double)
    time_ioc = ioc.gettimebase()
    n_0_l = n_0_l * ioc_2_n
    n_0 = n_0 * ioc_2_n
    n_0_u = n_0_u * ioc_2_n
else:
    sys.exit('Error while loading IOB')


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
plt.plot(time_ioc, n_0_l)
plt.plot(time_ioc, n_0)
plt.plot(time_ioc, n_0_u)
plt.xlabel(X_LABEL)
plt.ylabel(Y_LABEL)

plt.tight_layout(pad=0.1)
plt.show()

import sys
import numpy as np
import aug_sfutils as sf
import matplotlib.pyplot as plt


WID = 5.512
HIG = WID / 1.618
DEPTH = 50
X_LABEL = r'Time (s)'
Y_LABEL = r'flux expansion between the upstream and the X-point'

SHOT = 41158
FPG_DIAG = 'GQH'
MAG_EQU_DIAG = 'EQH'


fpg = sf.SFREAD(SHOT, FPG_DIAG)
equ = sf.EQU(SHOT, diag=MAG_EQU_DIAG)


if fpg.status:
    r_magax = fpg.getobject('Rmag', cal=True).astype(np.double)
    z_magax = fpg.getobject('Zmag', cal=True).astype(np.double)
    r_xp = fpg.getobject('Rxpu', cal=True).astype(np.double)
    z_xp = fpg.getobject('Zxpu', cal=True).astype(np.double)
    time_fpg = fpg.gettimebase('Zmag')
else:
    sys.exit('Error while loading ' + FPG_DIAG)


magax = np.array([r_magax, z_magax])
xp = np.array([r_xp, z_xp])
theta = np.arctan2(xp[1]-magax[1], xp[0]-magax[0])
r_l95_int = list()
z_l95_int = list()

for index, time in enumerate(time_fpg):
    r_l95_temp, z_l95_temp = sf.rhoTheta2rz(equ, 0.95, theta_in=theta[index],
                                            t_in=time,
                                            coord_in='rho_pol')
    r_l95_int.append(r_l95_temp.flatten()[0])
    z_l95_int.append(z_l95_temp.flatten()[0])

l95_intersection = np.array([r_l95_int, z_l95_int])
xp_expansion = np.linalg.norm(l95_intersection - xp, axis=0).astype(np.double)

r_m100_int, z_m100_int = sf.rhoTheta2rz(equ, 1, theta_in=0, t_in=time_fpg,
                                        coord_in='rho_pol')
m100_intersection = np.array([r_m100_int.flatten(), z_m100_int.flatten()])
r_m95_int, z_m95_int = sf.rhoTheta2rz(equ, 0.95, theta_in=0,
                                      t_in=time_fpg,
                                      coord_in='rho_pol')
m95_intersection = np.array([r_m95_int.flatten(), z_m95_int.flatten()])
mid_expansion = np.linalg.norm(m100_intersection - m95_intersection,
                               axis=0).astype(np.double)

f_exp = xp_expansion / mid_expansion


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
plt.plot(time_fpg, f_exp)
plt.xlabel(X_LABEL)
plt.ylabel(Y_LABEL)

plt.tight_layout(pad=0.1)
plt.show()

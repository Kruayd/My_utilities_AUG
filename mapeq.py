# Python code for equilibrium mapping made by:
# - Giovanni Tardini
#
# and modified/updated by:
# - Luca Cinnirella
#

# Last update: 13.12.2021

# IMPORTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import scipy.ndimage as ndm



# GLOBAL VARIABLES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dictionary that converts from coord string to flux array
coord2flux = {
               'rho_pol'    :   eqm.pfl,
               'Psi'        :   eqm.pfl,
               'rho_tor'    :   eqm.tfl,
               'rho_V'      :   eqm.vol,
               'r_V'        :   eqm.vol,
             }



# FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Find nearest index for given reference and test arrays
def get_nearest_index(ref_array, test_array):
    # For a given reference array and a test scalar,
    # finding the nearest index is quite simple:
    # evaluate the distance of the scalar from each array
    # element (|array - constant|) and then query for the
    # argument of the minimum (the index of the closest
    # element to the given scalar).
    # For two arrays we should cycle through the test array
    # elements but for very large arrays it would take too
    # much time. We can improve on speed by cloning the
    # reference array along an additional dimension
    # (making new columns that are clones of the original
    # array) and then subtracting from each row the test
    # array (done automatically by numpy). Finally we query
    # for the argument of the minimum along each column
    #
    # e.g.
    # REF_ARRAY = [0 1 2 ... n]
    # (the subsequent step is necessary since numpy is
    # row major)
    # we get to REF_ARRAY = ⎡[0]⎤
    #                       ⎢[1]⎥
    #                       ⎢[2]⎥
    #                       ⎢...⎥
    #                       ⎣[n]⎦
    # and finally to REF_ARRAY = ⎡[0 0 0 ... 0]⎤
    #                            ⎢[1 1 1 ... 1]⎥
    #                            ⎢[2 2 2 ... 2]⎥
    #                            ⎢[... ... ...]⎥
    #                            ⎣[n n n ... n]⎦
    idx = np.abs(np.tile(np.expand_dims(ref_array, -1), test_array.size) - test_array).argmin(0)
    # The function returns also an indexes of indexes array
    index_idx = np.arange(idx.size)
    return idx, index_idx


# Mapping from/to rho_pol, rho_tor, r_V, rho_V, Psi, r_a
# r_V is the STRAHL-like radial coordinate
#
# Input
# ----------
# eqm: equilibrium object
# t_in : float or 1darray or None
#       time (eqm.time if None)
# rho_in : float, ndarray
#       radial coordinates, 1D (time constant) or 2D+ (time variable) of size (nt,nx,...)
# coord_in:  str ['rho_pol', 'rho_tor' ,'rho_V', 'r_V', 'Psi','r_a']
#       input coordinate label
# coord_out: str ['rho_pol', 'rho_tor' ,'rho_V', 'r_V', 'Psi','r_a']
#       output coordinate label
# extrapolate: bool
#       extrapolate rho_tor, r_V outside the separatrix
#
# Output
# -------
# rho : 2d+ array (nt, nr, ...)
# converted radial coordinate
def rho2rho(eqm, rho_in, t_in=None, coord_in='rho_pol', coord_out='rho_tor', extrapolate=False):
    if t_in is None:
        t_in = eqm.time

    # Making everything at least 1D (even if scalar) simplify all the
    # subsequent operations
    rho_in = np.atleast_1d(rho_in)
    t_in = np.atleast_1d(t_in)

    # We have to check if the last axis of rho_in is as long as
    # t_in. If it is not the case, make the given tensor (i.e. nD arrays)
    # constant through time by extending along a new dimension and
    # repeatedly cloning the original tensor.
    if rho_in.shape[-1] != t_in.size:
        rho_in = np.tile(np.expand_dims(rho_in, -1), t_in.size)

    # Trivial case
    if coord_out == coord_in:
        return rho_in

    matrix_in = dictionary[coord_in]
    matrix_out = dictionary[coord_out]

    rho_in_matrix = normalize(matrix_in)
    rho_out_matrix = normalize(matrix_out)

    splinefunction = opt.UnivariateSpline_like(rho_in_matrix, rho_out_matrix, k=4, s=5e-3,ext=3)

    rho_out = splinefunction(rho_in, time)

    return rho_out



    # mag_out, sep_out = np.interp([PSI0[i], PSIX[i]],  PFL[i], label_out[i])
    # if lbl_in != lbl_out:
    #    mag_in, sep_in = np.interp([PSI0[i], PSIX[i]],  PFL[i], label_in[i])
    # else:
    #    mag_in = mag_out
    #    sep_in = sep_out
    #
    # if (abs(sep_out - mag_out) < 1e-4) or (abs(sep_in - mag_in) < 1e-4): #corrupted timepoint
    #    continue
    #
    # Poloidal flux PSI is the x-coordinate
    # AT time i:
    # Evaluate label_out and label_in at the magnetic axis and the separatrix through
    # linear interpolation (used for normalizing)
    # [PSI0, PSIX] (poloidal flux of magax and poloidal flux of the separatrix)
    # are the coordinates of the point at which to evaluate interpolation
    # PFL is poloidal flux (coordinate) and label_in/label_out are functions of PFL
    # (connected through the same grid)
    #
    #
    #
    # rho_out = (label_out[i] - mag_out)/(sep_out - mag_out)
    # rho_in  = (label_in [i] - mag_in )/(sep_in  - mag_in )
    #
    # rho_out[(rho_out > 1) | (rho_out < 0)] = 0  #remove rounding errors
    # rho_in[ (rho_in  > 1) | (rho_in  < 0)] = 0
    #
    # rho_out = np.r_[np.sqrt(rho_out), 1]
    # rho_in  = np.r_[np.sqrt(rho_in ), 1]
    #
    # ind = (rho_out==0) | (rho_in==0)
    # rho_out, rho_in = rho_out[~ind], rho_in[~ind]
    #
    # First normalize labels_in/out and rename them rho_in/out
    # correct for errors by setting to 0 everythin that is above 1 or below 0
    # numpy r_ generates array by concatenation. This leads to an array that is
    # the square root of rho_in/out with 1 appended at the end.
    # Finally we take only the elements that are non-zero
    #
    #
    #
    # sortind = np.unique(rho_in, return_index=True)[1]
    # w = np.ones_like(sortind)*rho_in[sortind]
    # w = np.r_[w[1]/2, w[1:], 1e3]
    # ratio = rho_out[sortind]/rho_in[sortind]
    # rho_in = np.r_[0, rho_in[sortind]]
    # ratio = np.r_[ratio[0], ratio]
    #
    # sortind in this case return the sorted unique array and
    # the index array from which the sorted one was constructed.
    # The [1] index select the letter array.
    # w is basically the sorted unique array but a bit modified.
    # It will be used in the UnivariateSpline.
    # We need a ratio (still a bit modified) between rho_out
    # and rho_in for the UnivariateSpline
    # we also prepend 0 to rho_in
    #
    #
    #
    # s = UnivariateSpline(rho_in, ratio, w=w, k=4, s=5e-3,ext=3)  #BUG s = 5e-3 can be sometimes too much, sometimes not enought :(
    #
    # First position is the independent variable, second is the dependent one,
    # w are the weights, s is the smoothing factor (number of knots)
    # and ext=3 returns boundary values for elements not in the interval
    # defined by the knot sequence



# Equilibrium mapping routine, map from R,Z -> rho (pol,tor,r_V,...)
# Fast for a large number of points
#
# Input
# ----------
# eqm: equilibrium object
# t_in : float or 1darray or None
#       time (eqm.time if None)
# r_in : ndarray
#       R coordinates
#       1D (time constant) or 2D+ (time variable) of size (...,nx,ny,nz,len(t_in))
# z_in : ndarray
#       Z coordinates
#       1D (time constant) or 2D+ (time variable) of size (...,nx,ny,nz,len(t_in))
# coord_out: str
#       mapped coordinates - rho_pol,  rho_tor, r_V, rho_V, Psi
# extrapolate: bool
#       extrapolate coordinates (like rho_tor) for values larger than 1
#
# Output
# -------
# rho : 2D+ array (...,nx,ny,nz,len(t_in))
# Magnetic flux coordinates of the points
def rz2rho(eqm, r_in, z_in, t_in=None, coord_out='rho_pol', extrapolate=True):
    if t_in is None:
        t_in = eqm.time

    # Making everything at least 1D (even if scalar) simplify all the
    # subsequent operations
    r_in = np.atleast_1d(r_in)
    z_in = np.atleast_1d(z_in)
    t_in = np.atleast_1d(t_in)

    # Step size of Rmesh and Zmesh
    dr = (eqm.Rmesh[-1] - eqm.Rmesh[0])/(len(eqm.Rmesh) - 1)
    dz = (eqm.Zmesh[-1] - eqm.Zmesh[0])/(len(eqm.Zmesh) - 1)

    # We have to check if the last axis of r_in and z_in is as long as
    # t_in. If it is not the case, make the given tensors (i.e. nD arrays)
    # constant through time by extending along a new dimension and
    # repeatedly cloning the original tensor.
    if r_in.shape[-1] != t_in.size:
        r_in = np.tile(np.expand_dims(r_in, -1), t_in.size)
    if z_in.shape[-1] != t_in.size:
        z_in = np.tile(np.expand_dims(z_in, -1), t_in.size)
    # r_in and z_in must have a 1 to 1 correspondence and, therefore,
    # must have the same shape
    if r_in.shape != z_in.shape:
        raise Exception(f'Not equal shape of z_in {z_in.shape} and r_in {r_in.shape}')

    t_idx, t_index_idx =  get_nearest_index(eqm.time, tarr)
    # r_idx and z_idx contain positions in terms of grid points
    # coordinates. The values are not discrete but continous and
    # removed
    # the offset relative to the mesh origin coordinates must be
    r_idx = (r_in - eqm.Rmesh[0])/dr
    z_idx = (z_in - eqm.Zmesh[0])/dz
    # In order to use ndm.map_coordinates with eqm.pfm (poloidal flux matrix which
    # is 3D: 2-space + 1-time), we must provide an array of indexes (position_idx)
    # with first axis of lenght equals to 3 (i.e. 3 coordinates for 3 dimension).
    # The first coordinate of position_idx (or eqm.pfm) is of course r_idx,
    # the second z_idx and the third an array based on t_idx with same shape as
    # r_idx and z_idx. Since there's a 1 to 1 correspondence between each element
    # of r_idx and z_idx last axis and time, we can just tile t_idx so that the
    # resulting array has the same shape of r_idx and z_idx and every
    # array[i, j, ..., k, :] is a copy of t_idx.
    # This can be easily achieved by
    # np.tile(t_idx, r_idx.shape[:-1]+(1,))
    #
    # r_idx.shape[:-1]+(1,) simply takes the shape of r_idx up to the last element
    # (excluded) and then concatenates 1 to it
    position_idx = np.array([r_idx, z_idx, np.tile(t_idx, r_idx.shape[:-1]+(1,))])

    Psi = ndm.map_coordinates(eqm.pfm, position_idx, mode='nearest', order=2, prefilter=True)
    # evaluate rho with rho2rho
    # return rho

'''
Collection of personal signal processing utilities made by:
    - Luca Cinnirella

Last update: 05.10.2022
'''


# IMPORTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from numbers import Number
from typing import Union, Optional, List
import numpy as np
# from numba import njit


# TYPE ALIASES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Numeric = Union[Number, np.ndarray]


# FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def find_nearest_index(array: np.ndarray,
                       value: Number, axis: Optional[int] = None) -> Numeric:
    '''
    Find index of closest element to a test value in a numpy array

    Input
    ----------
    array : ndarray
        array to be tested
    value : number_like
        test scalar
    axis : int, optional
        evaluation axis of np.argmin

    Output
    ----------
    index_array : int or ndarray of ints
        index or array of indexes where value is closer to the elements of
        array
    '''
    return np.absolute(array-value).argmin(axis=axis)


def shortest_array(array_list: List[np.ndarray]) -> np.ndarray:
    '''
    Find in a given list of arrays, the shortest one

    Input
    ----------
    array_list : list of ndarray
        array list of which the shortest has to be found

    Output
    ----------
    shortest_array: ndarray
        the shortest array in the given list
    '''
    sizes = np.array([array.size for array in array_list])
    return array_list[sizes.argmin()]


# Find nearest index for a given test arrays
def find_nearest_multiple_index(array: np.ndarray,
                                test_array: np) -> (np.ndarray, np.ndarray):
    '''
    Find indexes of closest elements to a test array in a numpy array

    Input
    ----------
    array : ndarray
        array to be tested
    test_array : 1D ndarray
        test values

    Output
    ----------
    index_array : ndarray of ints
        array of indexes where test_array elements are closer to the elements
        of array
    index_index : ndarray of ints
        indexes of indexes
    '''
    # For a given array and a test scalar, finding the nearest index is quite
    # simple: evaluate the distance of the scalar from each array element
    # (|array - constant|) and then query for the argument of the minimum (the
    # index of the closest element to the given scalar). For two arrays we
    # should cycle through the test array elements but for very large arrays it
    # would take too much time. We can improve on speed by cloning the
    # tested array along an additional dimension (making new columns that are
    # clones of the original array) and then subtracting from each row the test
    # array (done automatically by numpy). Finally we query for the argument of
    # the minimum along each column.
    #
    # e.g.
    # ARRAY = [0 1 2 ... n]
    # (the subsequent step is necessary since numpy is row major)
    # we get to ARRAY = ⎡[0]⎤
    #                   ⎢[1]⎥
    #                   ⎢[2]⎥
    #                   ⎢...⎥
    #                   ⎣[n]⎦
    # and finally to ARRAY = ⎡[0 0 0 ... 0]⎤
    #                        ⎢[1 1 1 ... 1]⎥
    #                        ⎢[2 2 2 ... 2]⎥
    #                        ⎢[... ... ...]⎥
    #                        ⎣[n n n ... n]⎦
    idx = np.abs(np.tile(np.expand_dims(array, -1), test_array.size) -
                 test_array).argmin(0)
    # The function returns also an indexes of indexes array
    index_idx = np.arange(idx.size)
    return idx, index_idx


# Sampling rate
def sampling_rate(time_array: np.ndarray) -> float:
    '''
    Get sampling rate from the middle of a time array

    Input
    ----------
    time_array : array_like
        array of sampling time-points

    Output
    ----------
    f_s : float
        sampling frequency of the measurement
    '''
    f_s = 1 / (time_array[int(time_array.size / 2) + 1] -
               time_array[int(time_array.size / 2)])
    return f_s


# Median filter
def median_filter(f_s_diag: float, time_window: float,
                  time_diag: np.ndarray, diag: np.ndarray,
                  time_points: Optional[np.ndarray] = None) -> (np.ndarray,
                                                                np.ndarray):
    '''
    Median filter and downsample the given signal

    Input
    ----------
    f_s_diag : float
        sampling rate of the diagnostic
    time_window : float
        size of the time window on which to apply the median filter
    time_diag : ndarray
        time array of the diagnostic
    diag : ndarray
        diagnostic signal
    time_points : ndarray, optional
        requested specific time points

    Output
    ----------
    time_diag_filt : ndarray
        sampled down time array
    diag_filt : ndarray
        filtered and sampled down signal
    '''
    if time_points is None:
        # NOTE: window, step_size and quotient are not index values but sizes,
        # therefore they must be strictly grater than 0 in order to let the
        # code work properly
        window = int(f_s_diag * time_window)
        window += (window+1) % 2                            # to next odd
        step_size = int(np.ceil(window/3))                  # at least 1/3

        # In order to properly slice the diagnostic signal, as it will be shown
        # in the next paragraph, we need an indexes array built like this:
        # indexes = [[first        window]
        #            [second       window]
        #                     ...
        #            [quotient+1th window]]
        # quotient is the number of window end points that fit within the given
        # time array minus 1. It is simply evaluated by subtracting window-1
        # (since we are counting end points we want to start from the end point
        # of the first window, though this won't be considered in anyway by the
        # integer division operation) form the size of time_diag and, then, by
        # integer dividing step_size.
        # When tiling, we want the resulting array to contain all the windows
        # that can fit within time_diag, therefore we need to add just the
        # first window (quotient + 1) which was excluded by the integer
        # division.
        # At this point (np.tile) our indexes array looks like this:
        # indexes = [[-window -window+1 ... -1]
        #            [-window -window+1 ... -1]
        #                               ...
        #            [-window -window+1 ... -1]]
        # To differentiate between windows we just add the value of
        # window + order of window*step_size but. In order to do so, arrays
        # must have last dimensions of the same sizes, therefore, we have to
        # transpose the indexes array before adding the offsets array.
        # Finally we can transpose the indexes array again so that we preserve
        # the time on last dimension convention
        quotient = (time_diag.size - window) // step_size
        indexes = (np.tile(np.arange(-window, 0),
                           (quotient+1, 1)).transpose() +
                   (window+np.arange(0, quotient+1)*step_size)).transpose()
        # Time indexes are selected as the central column of the indexes array
        time_indexes = indexes[..., int((window-1)/2)]
    else:
        # The procedure here is similar to the preceeding case but instead of
        # an indexes array similar to
        # indexes = [[-window -window+1 ... -1]
        #            [-window -window+1 ... -1]
        #                               ...
        #            [-window -window+1 ... -1]]
        # this one looks like:
        # indexes = [[-window/2 -window/2+1 ... window/2]
        #            [-window/2 -window/2+1 ... window/2]
        #                                   ...
        #            [-window/2 -window/2+1 ... window/2]]
        # The offsets array is directly given by time_indexes which is derived
        # by looking for the closest indexes of time_points in time_diag
        half_window = int(f_s_diag * time_window / 2)
        time_indexes, _ = find_nearest_multiple_index(time_diag, time_points)
        indexes = (np.tile(np.arange(-half_window, half_window+1),
                           time_points.shape + (1,)).transpose() +
                   time_indexes).transpose()

    # Apply median operation on the sliced diag array
    # The diag[..., indexes] operation returns a 3D array with the size on the
    # second axis equals to window. By applying the median operation we get
    # back to a 2D array with zeroth axis lenght equals to diag.shape[0] and
    # the first axis decimated.
    # The new time array is obtained by taking from each row of the indexes
    # array the middle point and then slicing time_diag
    diag_filt = np.median(diag[..., indexes], axis=-1)
    time_diag_filt = time_diag[time_indexes]
    return time_diag_filt, diag_filt


# @njit
# def n_ranges_nb(starts, ends):
#     master_range = np.arange(np.max(ends) + 1)
#     out_length = (ends - starts).sum()
#     out = np.zeros(out_length)
#     out_start, out_end = 0, 0
#     for start, end in zip(starts, ends):
#         out_end += end - start
#         out[out_start:out_end] = master_range[start:end]
#         out_start = out_end
#     return out

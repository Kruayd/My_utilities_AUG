import sys
import numpy as np
import pandas as pd
import aug_sfutils as sf
import scipy.interpolate as inter

import sig_proc as sgpr


MAN_N_CAL = 2.5
MAN_CSV_CAL = './csvs/IOC_calibrator.csv'


class SFIOCF01():

    def __init__(self, shot: int, sensitivity=MAN_N_CAL, interpolate=False):
        self.sfobject = sf.SFREAD(shot, 'IOC')

        self.status = self.sfobject.status

        if self.status:
            self.flux = self.sfobject.getobject('F01',
                                                cal=True).astype(np.double)
            self.time = self.sfobject.gettimebase('F01').astype(np.double)
            self.f_upper = self.sfobject.getobject('F_upper', cal=True)
            self.f_upper = self.f_upper[..., 0].astype(np.double)
            self.f_lower = self.sfobject.getobject('F_lower', cal=True)
            self.f_lower = self.f_lower[..., 0].astype(np.double)
        else:
            sys.exit('Error while loading IOC')

        if interpolate:
            iob = sf.SFREAD(shot, 'IOB')
            if iob.status:
                f_iob = iob.getobject('F01', cal=True).astype(np.double)
                time_iob = iob.gettimebase('F01').astype(np.double)
            else:
                sys.exit('Error while loading IOB')

            self.flux = self.interpolate_nan(time_iob, f_iob, self.flux)

        df_cal = pd.read_csv(MAN_CSV_CAL, index_col=0).loc[shot]
        cal_start = np.atleast_1d(df_cal['start'])
        cal_end = np.atleast_1d(df_cal['end'])
        cal_bg = np.atleast_1d(df_cal['background'])[0]
        cal_bg_end = np.atleast_1d(df_cal['end background'])[0]
        cal_type = np.atleast_1d(df_cal['type'])[0]
        cal_start_idx, _ = sgpr.find_nearest_multiple_index(self.time,
                                                            cal_start)
        cal_end_idx, _ = sgpr.find_nearest_multiple_index(self.time, cal_end)
        cal_idx = np.array([idx for start, end in zip(cal_start_idx,
                                                      cal_end_idx) for idx in
                            range(start, end + 1)])
        cal_time = self.time[cal_idx]
        cal_sample = self.flux[cal_idx]
        cal_sample_upper = self.f_upper[cal_idx]
        cal_sample_lower = self.f_lower[cal_idx]

        if cal_type == 'spline':
            self.spline_calibrate(cal_bg, cal_time, cal_sample,
                                  cal_sample_upper, cal_sample_lower,
                                  sensitivity)
        elif cal_type == 'constant':
            self.constant_calibrate(cal_bg, cal_bg_end, cal_sample,
                                    cal_sample_upper, cal_sample_lower,
                                    sensitivity)
        else:
            sys.exit('Calibration method not yet implemented')

    def getobject(self, name: str):
        if name == 'F01':
            return self.flux
        if name == 'F_upper':
            return self.f_upper
        if name == 'F_lower':
            return self.f_lower
        sys.exit(f'no such object as {name}')

    def gettimebase(self):
        return self.time

    def getparset(self, parameter: str):
        return self.sfobject.getparset(parameter)

    def interpolate_nan(self, reference_time, reference, signal):
        time_index, _ = sgpr.find_nearest_multiple_index(reference_time,
                                                      self.time)
        print(np.all(np.diff(time_index) >= 0))
        reference_time = reference_time[time_index]
        reference = reference[time_index]

        # IOC = a * IOB + c
        # we want to find a and c

        unc_sig = np.r_['1,2,0', 



        a_nan = (np.diff(signal) / np.diff(reference))
        a_nan[np.abs(a_nan) == np.inf] = np.nan
        for i in range(100):
            exclude = (np.abs(a_nan - a_nan.mean()) > 2*a_nan.std())
            a_nan[exclude] = np.nan
        c_nan = ((signal[1:] + signal[:-1]) / 2 -
                 a_nan * (reference[1:] + reference[:-1]) / 2)
        print(c_nan)
        time_int = (self.time[1:] + self.time[:-1])/2

        weights = ~np.isnan(a_nan)
        a_nan[~weights] = 0
        c_nan[~weights] = 0

        a_int = inter.UnivariateSpline(time_int, a_nan, w=weights, s=1e13)
        c_int = inter.UnivariateSpline(time_int, c_nan, w=weights, s=1e60)

        a_coeff = a_int(self.time)
        c_coeff = c_int(self.time)
        print(a_coeff)
        print(c_coeff)
        self.test = a_coeff

        return signal

    def spline_calibrate(self, cal_bg, cal_time, cal_sample, cal_sample_upper,
                         cal_sample_lower, sensitivity):
        weights = ~np.isnan(cal_sample)
        s_factor = ((weights * 4.1e21)**2).sum()
        cal_sample[~weights] = 0
        cal_sample_upper[~weights] = 0
        cal_sample_lower[~weights] = 0

        background = inter.UnivariateSpline(cal_time, cal_sample, w=weights,
                                            s=s_factor)
        self.bg_fit = background(self.time)
        std = np.sqrt(background.get_residual() / weights.sum())
        self.bg_std = np.full_like(self.time, std)
        print(std)

        background_upper = inter.UnivariateSpline(cal_time, cal_sample_upper,
                                                  w=weights, s=s_factor)
        std_upper = np.sqrt(background_upper.get_residual() / weights.sum())

        background_lower = inter.UnivariateSpline(cal_time, cal_sample_lower,
                                                  w=weights, s=s_factor)
        std_lower = np.sqrt(background_lower.get_residual() / weights.sum())

        if cal_bg == 'D':
            self.flux -= background(self.time) + 2*std
            self.f_upper -= background_upper(self.time) + 2*std_upper
            self.f_lower -= background_lower(self.time) + 2*std_lower

            self.flux = np.where(self.flux > 0, self.flux / sensitivity,
                                 self.flux)
            self.f_upper = np.where(self.f_upper > 0, self.f_upper /
                                    sensitivity, self.f_upper)
            self.f_lower = np.where(self.f_lower > 0, self.f_lower /
                                    sensitivity, self.f_lower)

            self.flux += background(self.time) + 2*std
            self.f_upper += background_upper(self.time) + 2*std_upper
            self.f_lower += background_lower(self.time) + 2*std_lower

        elif cal_bg == 'N':
            self.flux -= background(self.time) + 2*std
            self.f_upper -= background_upper(self.time) + 2*std_upper
            self.f_lower -= background_lower(self.time) + 2*std_lower

            self.flux = np.where(self.flux < 0, self.flux / sensitivity,
                                 self.flux)
            self.f_upper = np.where(self.f_upper < 0, self.f_upper /
                                    sensitivity, self.f_upper)
            self.f_lower = np.where(self.f_lower < 0, self.f_lower /
                                    sensitivity, self.f_lower)

            self.flux += (background(self.time) + 2*std) / sensitivity
            self.f_upper += (background_upper(self.time) + 2*std_upper) /\
                sensitivity
            self.f_lower += (background_lower(self.time) + 2*std_lower) /\
                sensitivity
        else:
            self.status = False
            sys.exit('Atomic species not yet implemented')

    def constant_calibrate(self, cal_bg, cal_bg_end, cal_sample,
                           cal_sample_upper, cal_sample_lower, sensitivity):
        background = np.where(self.time < cal_bg_end, cal_sample.mean(), 0)
        self.bg_fit = background
        std = np.where(self.time < cal_bg_end, cal_sample.std(), 0)
        self.bg_std = std

        background_upper = np.where(self.time < cal_bg_end,
                                    cal_sample_upper.mean(), 0)
        std_upper = np.where(self.time < cal_bg_end, cal_sample_upper.std(), 0)

        background_lower = np.where(self.time < cal_bg_end,
                                    cal_sample_lower.mean(), 0)
        std_lower = np.where(self.time < cal_bg_end, cal_sample_lower.std(), 0)

        if cal_bg == 'D':
            self.flux -= background + 2*std
            self.f_upper -= background_upper + 2*std_upper
            self.f_lower -= background_lower + 2*std_lower

            self.flux = np.where(self.flux > 0, self.flux / sensitivity,
                                 self.flux)
            self.f_upper = np.where(self.f_upper > 0, self.f_upper /
                                    sensitivity, self.f_upper)
            self.f_lower = np.where(self.f_lower > 0, self.f_lower /
                                    sensitivity, self.f_lower)

            self.flux += background + 2*std
            self.f_upper += background_upper + 2*std_upper
            self.f_lower += background_lower + 2*std_lower

        elif cal_bg == 'N':
            self.flux -= background + 2*std
            self.f_upper -= background_upper + 2*std_upper
            self.f_lower -= background_lower + 2*std_lower

            self.flux = np.where(self.flux < 0, self.flux / sensitivity,
                                 self.flux)
            self.f_upper = np.where(self.f_upper < 0, self.f_upper /
                                    sensitivity, self.f_upper)
            self.f_lower = np.where(self.f_lower < 0, self.f_lower /
                                    sensitivity, self.f_lower)

            self.flux += (background + 2*std) / sensitivity
            self.f_upper += (background_upper + 2*std_upper) / sensitivity
            self.f_lower += (background_lower + 2*std_lower) / sensitivity
        else:
            self.status = False
            sys.exit('Atomic species not yet implemented')

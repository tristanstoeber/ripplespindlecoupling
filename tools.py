from scipy.signal import butter, lfilter
import numpy as np
from scipy.signal import butter, lfilter, hilbert
import ripple_detection as rd
import pandas as pd
import pdb
from scipy.stats import zscore


def butter_bandpass(lowcut, highcut, fs, order=5):
    """ 
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """ 
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """ 
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """ 

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def segment_boolean_series(
        series,
        minimum_duration=False,
        maximum_duration=False,
        merge_duration=False):
    '''
    Modified from https://github.com/Eden-Kramer-Lab/ripple_detection/

    Returns a list of tuples where each tuple contains the start time of
     segement and end time of segment. It takes a boolean pandas series as
     input where the index is time.
     Parameters
     ----------
     series : pandas boolean Series (n_time,)
         Consecutive Trues define each segement.
     minimum_duration : float,
         Segments must be at least this duration to be included.
     maximum_duration : float,
         Segments must not exceed this duration to be included.
     merge_duration : float
         Segments with less distance are merged

     Returns
     -------
     segments : list of 2-element tuples
     '''
    
    start_times, end_times = rd.core._get_series_start_end_times(series)

    # check for distance between events
    if merge_duration:
        start_times, end_times = merge_when_close(start_times, end_times, merge_duration)

    # check for length of events
    evnts = []
    for strt, stp in zip(start_times, end_times):
        cond = True
        if minimum_duration:
            if stp < (strt + minimum_duration):
                cond=False
        if maximum_duration:
            if stp > (strt + maximum_duration):
                cond=False
        if cond:
            evnts.append((strt, stp))

    return evnts


def merge_when_close(strt, stp, d):
    """
    Merge events when stp_i-strt_i+1 < d

    """

    assert len(strt)==len(stp)
    assert np.array_equal(strt, np.sort(strt))
    assert np.array_equal(stp, np.sort(stp))
    
    dist = strt[1:]-stp[:-1]
    idx = np.arange(0, len(strt), 1)

    
    pos_merge = np.where(dist<d)[0]
    pos_stay_strt = np.array(list(set(idx)-set(pos_merge+1)))
    pos_stay_stp = np.array(list(set(idx)-set(pos_merge)))    
    strt_new = strt[pos_stay_strt]
    stp_new = stp[pos_stay_stp]

    return strt_new, stp_new
    
    
def evnt_pos(evnts, time):
    pos = np.searchsorted(time, np.array(evnts).flatten())
    pos_start = pos[::2]
    pos_stop = pos[1::2]
    pos = [(strt, stp) for strt, stp in  zip(pos_start, pos_stop)]
    return pos

def event_detection(
        data,
        freq_sampl,
        freq_min,
        freq_max,
        buttw_ord,
        dur_min,
        dur_max,
        dur_merge,
        ampl_thresh,
        return_magn):

    """
    Parameters
    ----------
    data : array_like, shape (n_time,)
    freq_sampl : float
        Number of samples per second.
    freq_min : int
        Bandpass filter minimal frequency in Hz
    freq_max : int
        Bandpass filter maximal frequency in Hz
    buttw_ord : int,
        Bandpass filter order,
    dur_min : float
        Minimal event duration in seconds
    dur_max : float
        Maximal event duration in seconds
    dur_merge : float
        Minimal duration between events in seconds    
    ampl_thresh : float
        Amplitude threshold in standard deviation
    return_magn : boolean
        Also return magnitude

    Returns
    ----------
    evts_ts 
    
    """
    # create time vector
    sampling_interval = 1./freq_sampl
    t_stop = len(data)*sampling_interval
    t = np.arange(0, t_stop, sampling_interval)
    assert len(data) == len(t)

    # bandpass filter
    data_fltd = butter_bandpass_filter(data, freq_min, freq_max, freq_sampl, order=buttw_ord)

    # Determine magnitude
    data_hilbert = hilbert(data_fltd)
    data_envlp = np.abs(data_hilbert)

    # standard deviation of magnitude
    data_zscore = zscore(data_envlp)

    # find data points exceeding std
    evts_bool = data_zscore > ampl_thresh

    # determine start/stop times
    is_above_threshold = pd.Series(evts_bool, index=t)
    evts_ts = segment_boolean_series(
        is_above_threshold,
        minimum_duration=dur_min,
        maximum_duration=dur_max,
        merge_duration=dur_merge)

    # TODO merge close events, repurpose rd.exclude_close_events
    
    if return_magn:
        # get peak amplitudes
        evnts_pos = evnt_pos(evts_ts, t)
        magn = []
        for strt, stp in evnts_pos:
            magn.append(data_envlp[strt:stp])
        
        return [evts_ts, magn]
    else:
        return evts_ts

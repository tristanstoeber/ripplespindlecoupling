from scipy.signal import butter, lfilter
import numpy as np
from scipy.signal import cheby1, lfilter, hilbert, sosfilt
import ripple_detection as rd
import pandas as pd
import pdb
from scipy.stats import zscore
import glob


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

def cheby1_bandpass_filter(data, cheby1_params):
    """
    Chebyshev type I bandpass filter

    Parameters
    ----------
    data : numpy.ndarray
        signal
    
    lowcut : int
        lower frequency edge in Hz
    highcut : int
        higher frequency edge in Hz
    fs_data : int
        sampling rate of data in Hz
    order : int
        order of filter
    ripple : float
        The maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number.

    Returns
    -------
    out : np.ndarray
    """
    
    sos = cheby1(**cheby1_params)
    data_filtrd = sosfilt(sos, data)
    

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
    evnts = np.array(evnts)
    pos_start = np.searchsorted(time, evnts[:, 0])
    pos_stop = np.searchsorted(time, evnts[:, 1])
    pos = [(strt, stp) for strt, stp in  zip(pos_start, pos_stop)]
    return pos

def get_evts_centers(evnts):
    return np.mean(evnts, axis=1)

def event_detection(
        data,
        freq_sampl,
        freq_min,
        freq_max,
        cheby1_ord,
        cheby1_ripple,
        dur_min,
        dur_max,
        dur_merge,
        ampl_thresh,
        return_info):

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
    cheby1_ord : int,
        Bandpass filter order
    cheby1_ripple : float
        see scipy.signal.cheby1
    dur_min : float
        Minimal event duration in seconds
    dur_max : float
        Maximal event duration in seconds
    dur_merge : float
        Minimal duration between events in seconds    
    ampl_thresh : float
        Amplitude threshold in standard deviation
    return_info : dict
        dict specifying return
        'evts_ts_strtstop' : bool
           start and stop of event
        'evts_ts_center' : bool
           event center
        'evts_ampl' : [-d_t0, d_t1] | True,
           amplitude of signal, in relation to center,
           if True within start stop
        'evts_data' : [-d_t0, d_t1] | None,
           raw signal, in relation to center,
           if True within start stop


    Returns
    ----------
    df : pandas.DataFrame
    
    """
    # create time vector
    sampling_interval = 1./freq_sampl
    t_stop = len(data)*sampling_interval
    t = np.arange(0, t_stop, sampling_interval)
    assert len(data) == len(t)

    # bandpass filter
    sos = cheby1(
        cheby1_ord,
        cheby1_ripple,
        [freq_min, freq_max],
        'bp',
        fs=freq_sampl,
        output='sos')
    data_fltd = sosfilt(sos, data)
    assert len(data)==len(data_fltd)

    # Determine magnitude
    data_hilbert = hilbert(data_fltd)
    data_envlp = np.abs(data_hilbert)
    assert len(data)==len(data_envlp)    

    # standard deviation of magnitude
    data_zscore = zscore(data_envlp)
    assert len(data)==len(data_zscore)
    
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

    # create return dataframe
    ret = {}
    keys = return_info.keys()
    
    if 'evts_ts_strtstop' in keys:
        ret['evts_ts_strtstop'] = evts_ts
    if 'evts_ts_center' in keys:
        ret['evts_ts_center'] = get_evts_centers(evts_ts)
    if 'evts_ampl' in keys:
        val = return_info['evts_ampl']
        if type(val) == list:
            assert len(val) == 2
            d_t0 = val[0]
            d_t1 = val[1]
        else:
            d_t0 = 0
            d_t1 = 0
            
        ar_evts_ts = np.array(evts_ts)
        ar_evts_ts[:, 0] += d_t0
        ar_evts_ts[:, 1] += d_t1
        
        evnts_pos = evnt_pos(ar_evts_ts, t)
        ampl = []
        ts_ampl = []
        for strt, stp in evnts_pos:
            ampl.append(data_envlp[strt:stp])
            ts_ampl.append(t[strt:stp])
            if stp-strt == 0:
                pdb.set_trace()
            
        ret['evts_ampl'] = ampl
        ret['evts_ampl_ts'] = ts_ampl
        
    if 'evts_data' in keys:
        val = return_info['evts_data']
        if type(val) == list:
            assert len(val) == 2
            d_t0 = val[0]
            d_t1 = val[1]
        else:
            d_t0 = 0
            d_t1 = 0
            
        ar_evts_ts = np.array(evts_ts)
        ar_evts_ts[:, 0] += d_t0
        ar_evts_ts[:, 1] += d_t1
        
        evnts_pos = evnt_pos(ar_evts_ts, t)
        datapts = []
        ts_datapts = []
        for strt, stp in evnts_pos:
            datapts.append(data[strt:stp])
            ts_datapts.append(t[strt:stp])

        ret['evts_data'] = datapts
        ret['evts_data_ts'] = ts_datapts

    df = pd.DataFrame.from_dict(ret)

    return df


def generate_fileinfo(path, re=None, columnnames=None):
    """
    Generate dataframe with filenames and extract infos
    
    Params
    ------
    path : str
       Path to search for files
    re : str
       Regular expression to use
    columnnames : dir
       To rename columns corresponding to groups in re
    """

    ls_fname = []
    for fname in glob.glob(path):
        ls_fname.append({'fname' : fname})
    
        df = pd.DataFrame({'fname': glob.glob(path)})

    if not re:
        sub_id = '\w+'
        sub_date = '\d{6}'
        sub_session = '\d+'

        re = (r'('+sub_id+
              ')_('+sub_date+
              ')_('+sub_session+
              ').set')

    df_extract = df['fname'].str.extract(re, expand=True)

    if not columnnames:
        columnnames = {
            0:'id',
            1:'date',
            2:'session'}
    
    df_extract = df_extract.rename(
        columns=columnnames)

    df = df.join(df_extract)

    return df


import numpy as np
from scipy.signal import cheby1, hilbert, sosfilt
import ripple_detection as rd
import pandas as pd
import pdb
from scipy.stats import zscore
import glob
import multiprocessing as mp
from pyxona import File


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
        t_excl_edge,
        return_info,
        ):

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
    t_excl_edge : float
        Exclude events with centers closer to edge than t_excl_edge
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
    assert len(data) == len(data_fltd)

    # Determine magnitude
    data_hilbert = hilbert(data_fltd)
    data_envlp = np.abs(data_hilbert)
    assert len(data) == len(data_envlp)

    # standard deviation of magnitude
    data_zscore = zscore(data_envlp)
    assert len(data) == len(data_zscore)

    # find data points exceeding std
    evts_bool = data_zscore > ampl_thresh

    # determine start/stop times
    is_above_threshold = pd.Series(evts_bool, index=t)
    evts_ts = segment_boolean_series(
        is_above_threshold,
        minimum_duration=dur_min,
        maximum_duration=dur_max,
        merge_duration=dur_merge)

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
    # exclude events with centers too close to edge
    if t_excl_edge:
        bool_valid = np.logical_and(
            df['evts_ts_center'] > t_excl_edge,
            df['evts_ts_center'] < t_stop-t_excl_edge)
        df = df[bool_valid].reset_index()
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
    for fname in glob.glob(path+'*.set'):
        ls_fname.append({'fname' : fname})
    
        df = pd.DataFrame({'fname': glob.glob(path+'*.set')})

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


def load_notes(path, fname_notes):
    df_notes = pd.read_excel(path+fname_notes, engine="odf")

    # fill missing values
    for c_name in ['animal', 'date', 'treatment']:
        df_notes[c_name] = df_notes[c_name].fillna(method='ffill')

    # rename animal header for consistency
    df_notes = df_notes.rename(columns={'animal': 'id'})

    # convert date and session to string for consistency
    df_notes['date'] = df_notes['date'].astype(int).astype(str)
    df_notes['session'] = df_notes['session'].astype(int).astype(str)

    return df_notes


def event_detection_mp(
        df_info, params_ripples, params_spindles, verbose=False, concat_results=True):
    """
    event detection with multiprocessing
    """
    
    results = {}
        
    pool = mp.Pool(mp.cpu_count()-1)

    try:
        for index, row in df_info.iterrows():
            results[index] = pool.apply_async(
                event_detection_wrapper,
                args=(row, params_ripples, params_spindles, verbose),
            )
    except:
        pdb.set_trace()
    pool.close()
    pool.join()
    
    ls_df_rppls = []
    ls_df_spndls = []

    for i, val in results.items():
        res_i = val.get()
        ls_df_rppls.append(res_i[0])
        ls_df_spndls.append(res_i[1])

    if concat_results:
        df_rppls = pd.concat(ls_df_rppls, ignore_index=True, sort=False)
        df_spndls = pd.concat(ls_df_spndls, ignore_index=True, sort=False)

        return df_rppls, df_spndls
    else:
        return ls_df_rppls, ls_df_spndls


def event_detection_wrapper(
        row, params_ripples, params_spindles, verbose=False):
    """
    Wrapper function around event detection to allow for multiprocessing
    """
    fname = row['fname']

    f = File(fname)
    lfp_hpc = f.analog_signals[0]
    lfp_ctx = f.analog_signals[1]

    df_rppls_i = event_detection(
        lfp_hpc.signal.magnitude,
        lfp_hpc.sample_rate.magnitude,
        **params_ripples)

    df_spndls_i = event_detection(
        lfp_ctx.signal.magnitude,
        lfp_ctx.sample_rate.magnitude,
        **params_spindles)

    assert params_ripples['t_excl_edge'] == params_spindles['t_excl_edge']
    # merge with df_fnames
    d = row.to_dict()
    df_fnames_i = pd.DataFrame({k: [v] for k, v in d.items()})
    df_fnames_i['tmp'] = 1

    # assign duration of recording
    duration = f.attrs['duration']
    # compensate for cutted edges
    duration = duration - 2*params_ripples['t_excl_edge']
    df_fnames_i['duration'] = duration

    df_rppls_i['tmp'] = 1
    df_rppls_i = pd.merge(df_fnames_i, df_rppls_i, on=['tmp'])
    df_rppls_i.drop('tmp', axis=1, inplace=True)

    df_spndls_i['tmp'] = 1
    df_spndls_i = pd.merge(df_fnames_i, df_spndls_i, on=['tmp'])
    df_spndls_i.drop('tmp', axis=1, inplace=True)
    if verbose:
        print(
            'Finished event detection for animal '+row['id']+
            ', date '+str(row['date'])+
            ', session '+str(row['session']))
        print('\n')
    return df_rppls_i, df_spndls_i


def get_events_with_zero_change(
        x, tol=10e-10, return_values=False, minimum_duration=0):
    """
    Returns list of start and ends of events where change smaller tol
    can be detected in relation to previous value.
    Values of these events can returned.
    
    Parameters
    ----------
    x : ndarray, shape (n,)
    tol : float
        Tolerance for considering values to not change
    return_values : boolean
    minimum_duration : int
        Minimal duration of events to be considered

    Returns
    ---------
    start_end: ndarray, shape (n_segments, 2)
    vals: ndarray, shape (n_segments,)

    """
    
    diff = np.diff(x, prepend=tol+1)  # prepend one value
    # to account for shorter diff result

    diff_bool = np.abs(diff) <= tol

    start_end = rd.core.segment_boolean_series(
        pd.Series(diff_bool), minimum_duration=minimum_duration)
    start_end = np.array(start_end)

    if return_values:
        vals = x[start_end[:, 0]]
        assert np.allclose(vals, x[start_end[:, 1]], tol)
        return start_end, vals
    else:
        return start_end


def detect_maxmin_reaches(x, vmin=None, vmax=None, tol=10e-10, len_seg_min=0):
    """
    Detect segments where signal hits the positive or negative end of the
    recording range
    
    Params
    ----------
    x : ndarray, shape (n,)
    vmin : u
        if None, then maximal signal of data will be taken
    vmax : maximal value to look for
        if None, then maximal signal of data will be taken
    tol : float
        tolerance for how close values subsequent can be before
        they will be considered constant
    len_seg_min : int
        Minimal length of segments

    Returns
    -------
    start_end: ndarray, shape (n_segments, 2)

    """
    if not vmin:
        vmin = np.min(x)
    if not vmax:
        vmax = np.max(x)

    # find values close to edge
    bool_close_vmin = np.abs(x - vmin) <= tol
    bool_close_vmax = np.abs(x - vmax) <= tol
    bool_close = np.logical_or(bool_close_vmin, bool_close_vmax)

    # detect segments
    start_end = rd.core.segment_boolean_series(
        pd.Series(bool_close), minimum_duration=len_seg_min)
    start_end = np.array(start_end)
    
    return start_end
    

def detect_maxmin_reaches_mp(
        df_info,
        params,
        verbose=True
        ):
    """
    noise detection with multiprocessing
    """

    results = {}

    pool = mp.Pool(mp.cpu_count()-1)

    for index, row in df_info.iterrows():
        results[index] = pool.apply_async(
            detect_maxmin_reaches_wrapper,
            args=(row, params, verbose),
        )
        
    pool.close()
    pool.join()

    ls_df_res = []
    for i, val in results.items():
        res_i = val.get()
        res_i = pd.DataFrame(res_i).transpose()
        ls_df_res.append(res_i)
        
    df_res = pd.concat(ls_df_res, ignore_index=True, sort=False)
    return df_res


def detect_maxmin_reaches_wrapper(
        row, params, verbose=True):
    """
    Wrapper function around event detection to allow for multiprocessing
    """
    fname = row['fname']

    f = File(fname)
    lfp = {
        'hpc': f.analog_signals[0].signal.magnitude,  # hippocampal signal
        'ctx': f.analog_signals[1].signal.magnitude,  # cortex signal
    }
    hpc_sampling_rate = f.analog_signals[0].sample_rate.magnitude
    ctx_sampling_rate = f.analog_signals[1].sample_rate.magnitude
    assert hpc_sampling_rate == ctx_sampling_rate
    sampling_interval = 1./hpc_sampling_rate
    t_stop = len(lfp['hpc'])*sampling_interval
    t = np.arange(0, t_stop, sampling_interval)
    
    for key, lfp_i in lfp.items():
        vmin = row[key + '_rec_min']
        vmax = row[key + '_rec_max']

        start_end = detect_maxmin_reaches(
            lfp_i, vmin=vmin, vmax=vmax, **params)
        
        # convert from pos to time
        if len(start_end) > 0:
            start_end_t = np.vstack([
                [t[a[0]], t[a[1]]]
                for a in start_end])
        else:
            start_end_t = None
            
        row[key + '_' + 'maxmin_reaches'] = start_end_t

    if verbose:
        print(
            'Finished detection of maxmin reaches for animal ' + row['id'] +
            ', date ' + str(row['date']) +
            ', session '+str(row['session']))
        print('\n')
        
    return row

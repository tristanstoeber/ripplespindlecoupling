params_ripples = {
    'freq_min': 100,
    'freq_max': 250,
    'cheby1_ord': 4,
    'cheby1_ripple': 0.1,
    'dur_min': 0.01,
    'dur_max': 1.,
    'dur_merge': 0.05,
#    'dur_merge': False,
    'ampl_thresh': 3.,
    't_excl_edge': 10,
    'return_info':
        {'evts_ts_strtstop': True,
         'evts_ts_center': True,
         'evts_ampl': [-4, 4],
         'evts_data': [-4, 4]}
}
params_spindles = {
    'freq_min': 12,
    'freq_max': 15,
    'cheby1_ord': 4,
    'cheby1_ripple': 0.1,
    'dur_min': 0.2,
    'dur_max': 2.,
    'dur_merge': 0.1,
    'ampl_thresh': 2,
    't_excl_edge': 10,
    'return_info':
        {'evts_ts_strtstop': True,
         'evts_ts_center': True,
         'evts_ampl': [-14, 14],
         'evts_data': [-4, 4],
        }
}

params_maxmin_reaches = {
    'len_seg_min': 2,
    'tol': 10e-5}


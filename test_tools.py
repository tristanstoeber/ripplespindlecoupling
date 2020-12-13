import pytest
import tools
import numpy as np
import pdb

def test_evnt_pos():
    evnts = [
        (1, 2),
        (4, 6)
    ]

    ts = np.arange(0, 9, 1)

    envts_pos = tools.evnt_pos(evnts, ts)
    assert envts_pos == evnts
    

def test_merge_when_close():
    strt = np.array([0., 1, 2, 3, ])
    stp = np.array([0.9, 1.9, 2.1, 3.1])
    d = 0.2

    strt_new, stp_new = tools.merge_when_close(strt, stp, d)

    strt_target = np.array([0., 3, ])
    stp_target = np.array([2.1, 3.1])

    assert np.array_equal(strt_new, strt_target)
    assert np.array_equal(stp_new, stp_target)    

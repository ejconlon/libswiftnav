#from swiftnav cimport plover
import numpy as np
cimport numpy as np
import pandas as pd
from libc.stdlib cimport malloc, free

#def to_index(gnss_signal_t s):
#  dd_N, string-band, string-satid

def convert_code(band):
    mapping = {
        '1': CODE_GPS_L1CA,
        '2': CODE_GPS_L2CM
    }
    assert band in mapping
    return mapping[band]

cdef class CPloverFilter:
    cdef filter_state state;

    def __init__(self):
        self.state = make_filter_state();

    def get_P(self):
        cdef int sats = self.state.num_sats
        cdef np.ndarray[double, ndim=2, mode="c"] P_flat = np.zeros([sats+2, sats+2])
        cdef np.ndarray[double, ndim=1, mode="c"] P_max = np.zeros([225])
        P_max[:] = self.state.P

        P_flat[:] = P_max.reshape((15, 15))[:sats+2, :sats+2]
        return P_flat

        #self.P = P_flat.reshape((sats+2, sats+2))

    def get_baseline(self):
        x, y, z = self.state.x[0], self.state.x[1], self.state.x[2]
        return pd.Series([x, y, z], index=['x', 'y', 'z'])

    def update(self, base_obs, sdiffs, receiver_ecef):
        # 0 filter_state = self.state
        # 1 measurements
        sats = len(sdiffs)
        #cdef np.ndarray[measurement, ndim=1, mode="c"] measurements = np.empty(sats)
        cdef measurement *measurements = <measurement *>malloc(sats*sizeof(measurement))
        #for i, sdiff in enumerate(list(sdiffs.iterrows()))
        #import ipdb
        for i in range(sdiffs.shape[0]):
            sdiff = sdiffs.ix[i]

            #ipdb.set_trace()

            measurements[i].pseudorange = sdiff['pseudorange']
            measurements[i].carrier_phase = sdiff['carrier_phase']
            measurements[i].snr = sdiff['signal_noise_ratio']
            measurements[i].sat_id.sat = int(sdiff['sat'])
            measurements[i].sat_id.code = convert_code(sdiff['band'])

        # 2 base_pos
        cdef np.ndarray[double, ndim=1, mode="c"] base_pos = np.zeros([3])
        base_pos[0] = receiver_ecef[0]
        base_pos[1] = receiver_ecef[1]
        base_pos[2] = receiver_ecef[2]

        # 3 sat_pos
        cdef np.ndarray[double, ndim=2, mode="c"] sat_pos = \
          np.atleast_2d(base_obs[['sat_x', 'sat_y', 'sat_z']].values.copy(order='c'))

        update_(sats, &self.state, &measurements[0], &base_pos[0], &sat_pos[0,0])
        free(measurements)
        #plover.update_()

def kalman_predict(
        np.ndarray[double, ndim=1, mode="c"] x,
        np.ndarray[double, ndim=2, mode="c"] P,
        np.ndarray[double, ndim=2, mode="c"] F,
        np.ndarray[double, ndim=2, mode="c"] Q):
    dim = len(x)
    cdef np.ndarray[double, ndim=1, mode="c"] x_new = np.zeros([dim])
    cdef np.ndarray[double, ndim=2, mode="c"] P_new = np.zeros([dim, dim])
    kalman_predict_(dim, &x[0], &P[0,0], &F[0,0], &Q[0,0], &x_new[0], &P_new[0,0])
    return (x_new, P_new)

def kalman_update(
        np.ndarray[double, ndim=1, mode="c"] x,
        np.ndarray[double, ndim=2, mode="c"] P,
        np.ndarray[double, ndim=1, mode="c"] y,
        np.ndarray[double, ndim=2, mode="c"] H,
        np.ndarray[double, ndim=2, mode="c"] R):
    xdim = len(x)
    dim = len(y)
    cdef np.ndarray[double, ndim=1, mode="c"] x_new = np.zeros([xdim])
    cdef np.ndarray[double, ndim=2, mode="c"] P_new = np.zeros([xdim, xdim])
    kalman_update_(xdim, dim, &x[0], &P[0,0], &y[0], &H[0,0], &R[0,0], &x_new[0], &P_new[0,0])
    return (x_new, P_new)

def inverse(np.ndarray[double, ndim=2, mode="c"] P):
    dim = len(P)
    cdef np.ndarray[double, ndim=2, mode="c"] new = np.zeros([dim, dim])
    matrix_inverse(dim, &P[0,0], &new[0,0])
    return new

def observation_model(self,
        rover_obs, base_obs,
        #np.ndarray[double, ndim=1, mode="c"] x,
        base_pos_tuple):

    sdiffs = self.get_single_diffs(rover_obs, base_obs, propagate_base=False)
    sats = len(sdiffs)
    cdef np.ndarray[double, ndim=1, mode="c"] pseudoranges = sdiffs['pseudorange'].values
    cdef np.ndarray[double, ndim=1, mode="c"] carrier_phases = sdiffs['carrier_phase'].values
    cdef np.ndarray[double, ndim=2, mode="c"] sat_pos = \
      np.atleast_2d(base_obs[['sat_x', 'sat_y', 'sat_z']].values.copy(order='c'))
    cdef np.ndarray[double, ndim=1, mode="c"] base_pos = np.zeros([3])
    base_pos[0] = base_pos_tuple[0]
    base_pos[1] = base_pos_tuple[1]
    base_pos[2] = base_pos_tuple[2]
    cdef np.ndarray[double, ndim=1, mode="c"] x = self.x.values

    cdef np.ndarray[double, ndim=1, mode="c"] y = np.zeros([2*sats-2])
    cdef np.ndarray[double, ndim=2, mode="c"] H = np.zeros([2*sats-2, sats+2])
    cdef np.ndarray[double, ndim=2, mode="c"] R = np.zeros([2*sats-2, 2*sats-2])
    observation_model_(sats, &pseudoranges[0], &carrier_phases[0], &x[0], &base_pos[0],
            &sat_pos[0,0], self.sig_cp, self.sig_pr, &y[0], &H[0,0], &R[0,0])

    return (y, H, R)

# make KF object wrapper for plover implementation
    # init?
    # structs in cython?
# construct arguments to update_
# test suite
#def update(  ):
#    pass
#    #filter_state ?
#    #measurements[sats] ?
#
#    #receiver_ecef[3]
#    #sat_positions[sats, 3]

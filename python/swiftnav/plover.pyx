import numpy as np
cimport numpy as np
import pandas as pd
from libc.stdlib cimport malloc, free
from gnss_analysis.io.common import create_sid

# TODO add other codes
def convert_code(band):
    mapping = {
        '1': CODE_GPS_L1CA,
        '2': CODE_GPS_L2CM
    }
    assert band in mapping
    return mapping[band]

# TODO add other codes
def from_code(band):
    mapping = {
        CODE_GPS_L1CA: '1',
        CODE_GPS_L2CM: '2',
    }
    assert band in mapping
    return mapping[band]

def sat_pad(sat):
    if sat < 10:
        return '0' + str(sat)
    else:
        return str(sat)

cdef class CPloverFilter:
    cdef filter_state state;

    def __init__(self):
        self.state = make_filter_state();

    def get_ref_id(self):
        # TODO return band too
        return create_sid('GPS', sat_pad(self.state.ref.sat), from_code(self.state.ref.code))
    def get_knowns(self):
        ref_tuples = [('reference_N',  '1', str(self.get_ref_id()))]
        ref_index = pd.MultiIndex.from_tuples(ref_tuples,
                                              names=('type', 'group', 'name'))
        knowns = pd.Series(0., index=ref_index)
        return knowns

    def change_reference_signal(self, int new_ref):
        cdef gnss_signal_t s
        s.sat = new_ref
        s.code = CODE_GPS_L1CA
        change_ref(&self.state, s)

    def get_state(self):
        cdef int sats = self.state.num_sats
        cdef np.ndarray[double, ndim=1, mode="c"] x_flat = np.zeros([sats+2])
        # stored in struct as flat array of maximum possible size
        max_size = MAX_FILTER_STATE_DIM
        cdef np.ndarray[double, ndim=1, mode="c"] x_max = np.zeros([max_size])
        x_max[:] = self.state.x

        x_flat[:] = x_max[:sats+2]
        return x_flat
    def get_covariance(self):
        cdef int sats = self.state.num_sats
        cdef np.ndarray[double, ndim=2, mode="c"] P_flat = np.zeros([sats+2, sats+2])
        # stored in struct as flat array of maximum possible size
        max_size = MAX_FILTER_STATE_DIM * MAX_FILTER_STATE_DIM
        cdef np.ndarray[double, ndim=1, mode="c"] P_max = np.zeros([max_size])
        P_max[:] = self.state.P

        P_flat[:] = P_max[:(sats+2)*(sats+2)].reshape((sats+2, sats+2))
        return P_flat

    def get_baseline(self):
        x, y, z = self.state.x[0], self.state.x[1], self.state.x[2]
        return pd.Series([x, y, z], index=['x', 'y', 'z'])

    def update(self, base_obs, sdiffs, receiver_ecef):
        # 0 filter_state = self.state
        # 1 measurements
        sats = len(sdiffs)
        cdef sdiff_t *sdiffs_converted = <sdiff_t *>malloc(sats*sizeof(sdiff_t))
        # sat_pos
        cdef np.ndarray[double, ndim=2, mode="c"] sat_pos = \
          np.atleast_2d(base_obs[['sat_x', 'sat_y', 'sat_z']].values.copy(order='c'))

        for i in range(sdiffs.shape[0]):
            sdiff = sdiffs.ix[i]

            sdiffs_converted[i].pseudorange = sdiff['pseudorange']
            sdiffs_converted[i].carrier_phase = sdiff['carrier_phase']
            sdiffs_converted[i].snr = sdiff['signal_noise_ratio']
            sdiffs_converted[i].sid.sat = int(sdiff['sat'])
            sdiffs_converted[i].sid.code = convert_code(sdiff['band'])
            sdiffs_converted[i].sat_pos[0] = sdiff['sat_x']
            sdiffs_converted[i].sat_pos[1] = sdiff['sat_y']
            sdiffs_converted[i].sat_pos[2] = sdiff['sat_z']

        # 2 base_pos
        cdef np.ndarray[double, ndim=1, mode="c"] base_pos = np.zeros([3])
        base_pos[0] = receiver_ecef[0]
        base_pos[1] = receiver_ecef[1]
        base_pos[2] = receiver_ecef[2]

        #cdef filter_state *s = <filter_state *>malloc(0)
        #free(s)
        #update_(sats, s, &sdiff_t[0], &base_pos[0], &sat_pos[0,0])
        update_(sats, &self.state, &sdiffs_converted[0], &base_pos[0])
        free(sdiffs_converted)


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

#def observation_model(self,
#        rover_obs, base_obs,
#        #np.ndarray[double, ndim=1, mode="c"] x,
#        base_pos_tuple):
#
#    sdiffs = self.get_single_diffs(rover_obs, base_obs, propagate_base=False)
#    sats = len(sdiffs)
#    cdef np.ndarray[double, ndim=1, mode="c"] pseudoranges = sdiffs['pseudorange'].values
#    cdef np.ndarray[double, ndim=1, mode="c"] carrier_phases = sdiffs['carrier_phase'].values
#    cdef np.ndarray[double, ndim=2, mode="c"] sat_pos = \
#      np.atleast_2d(base_obs[['sat_x', 'sat_y', 'sat_z']].values.copy(order='c'))
#    cdef np.ndarray[double, ndim=1, mode="c"] base_pos = np.zeros([3])
#    base_pos[0] = base_pos_tuple[0]
#    base_pos[1] = base_pos_tuple[1]
#    base_pos[2] = base_pos_tuple[2]
#    cdef np.ndarray[double, ndim=1, mode="c"] x = self.x.values
#
#    cdef np.ndarray[double, ndim=1, mode="c"] y = np.zeros([2*sats-2])
#    cdef np.ndarray[double, ndim=2, mode="c"] H = np.zeros([2*sats-2, sats+2])
#    cdef np.ndarray[double, ndim=2, mode="c"] R = np.zeros([2*sats-2, 2*sats-2])
#    observation_model_(sats, &pseudoranges[0], &carrier_phases[0], &x[0], &base_pos[0],
#            &sat_pos[0,0], self.sig_cp, self.sig_pr, &y[0], &H[0,0], &R[0,0])
#
#    return (y, H, R)

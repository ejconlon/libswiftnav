cimport libc.stdlib
from libcpp cimport bool
from prelude cimport (u16, s32)

cdef extern from "libswiftnav/plover/plover_lib.h":
  enum: MAX_FILTER_STATE_DIM

  ctypedef struct gnss_signal_t:
      u16 sat
      u16 code

  ctypedef struct filter_state:
      s32 num_sats
      gnss_signal_t ref
      gnss_signal_t sats [MAX_FILTER_STATE_DIM]
      bool invalid
      double x [MAX_FILTER_STATE_DIM]
      double P [MAX_FILTER_STATE_DIM * MAX_FILTER_STATE_DIM]
      double sigma

  ctypedef struct measurement:
      double pseudorange
      double carrier_phase
      double snr
      gnss_signal_t sat_id
      double sat_pos[3]

cdef extern from "libswiftnav/plover/plover_lib.h":
    filter_state make_filter_state ();
    void kalman_predict_ (const s32 dim, const double * x, const double * P, const double * F, const double * Q, double * x_new, double * P_new);
    void kalman_update_ (const s32 dim, const s32 dim, const double * x, const double * P, const double * y, const double * H, const double * R, double * x_new, double * P_new);

    s32 matrix_inverse (const s32 dim, const double * x, double * y);
    void print_all_ (const s32 dim, const double * innovation, const double * S, const double * Si, const double * K, const double * x_new, const double * P_new);
    void direct_observation_model_ (const s32 sats, const double * pseudoranges, const double * carrier_phases, const double * x, const double * base_pos, const double * sat_positions, const double sig_cp, const double sig_pr, double * y, double * H, double * R);
    void sigtest (gnss_signal_t * const x);

    void update_ (const s32 dim, filter_state * const state, const measurement * input_sdiffs, const double * receiver_ecef);

cdef extern from "libswiftnav/signal.h":
    enum:
      CODE_INVALID
      CODE_GPS_L1CA
      CODE_GPS_L2CM
      CODE_SBAS_L1CA
      CODE_COUNT

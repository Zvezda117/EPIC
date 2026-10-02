#pragma once
#ifdef __cplusplus
extern "C" {
#endif
/* Arrays are host C-contiguous: pos[N,3]/freq[C]/dirs[P,3]/center[3] doubles,
 * gains[C,N,2] and input[T,C,N,2] interleaved complex FP32 (real,imag).
 * Output[C,P,4] FP32 is mean(XX,YY,Re(XY*),Im(XY*)). */
void* epic_fx_create(int n, int channels, int pixels, int ntime, int device,
                     int remove_autocorr, const double* positions_m,
                     const double* frequencies_hz, const double* directions,
                     const double* phase_center, const void* gains_complex64);
int epic_fx_image(void* context, const void* voltages_complex64, float* output);
void epic_fx_destroy(void* context);
/* Error text is thread-local, valid until the next API call in that thread. */
const char* epic_fx_error(void);
#ifdef __cplusplus
}
#endif

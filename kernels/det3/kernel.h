#ifndef _DET3_KERNEL_H_
#define __DET3_KERNEL_H_

#include <immintrin.h>

#define SIMD_SIZE 8

// void kernel(float *Ax, float *Ay, float *Bx, float *By, float *Cx, float *Cy,
//             float *Dx, float *Dy, float *out) {
//   __m256 reg0 = _mm256_load_ps(Ax);
//   __m256 reg1 = _mm256_load_ps(Ay);
//   __m256 reg2 = _mm256_load_ps(Bx);
//   __m256 reg3 = _mm256_load_ps(By);
//   __m256 reg4 = _mm256_load_ps(Cx);
//   __m256 reg5 = _mm256_load_ps(Cy);
//   __m256 reg4 = _mm256_load_ps(Dx);
//   __m256 reg5 = _mm256_load_ps(Dy);

//   //__m256 reg6 =

//   _mm256_store_ps(out, TODO);
// }

#endif
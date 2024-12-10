#ifndef _DET3_KERNEL1_H_
#define __DET3_KERNEL1_H_

#include <immintrin.h>

static inline void kernel_sub
(
  float*     restrict Ax,
  float*     restrict Ay,
  float*     restrict Bx,
  float*     restrict By,
  float*     restrict Cx,
  float*     restrict Cy,
  float*     restrict Dx,
  float*     restrict Dy,
  __m256*      a,
  __m256*      b,
  __m256*      d,
  __m256*      e,
  __m256*      g,
  __m256*      h
){

  __m256 reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  __m256 reg8, reg9, reg10, reg11, reg12, reg13, reg14, reg15;

  reg0 = _mm256_load_ps(&Ax[0]);
  reg1 = _mm256_load_ps(&Ay[0]);
  reg2 = _mm256_load_ps(&Bx[0]);
  reg3 = _mm256_load_ps(&By[0]);
  reg4 = _mm256_load_ps(&Cx[0]);
  reg5 = _mm256_load_ps(&Cy[0]);
  reg6 = _mm256_load_ps(&Dx[0]);
  reg7 = _mm256_load_ps(&Dy[0]);

  reg8  = _mm256_load_ps(&Ax[SIMD_SIZE]);
  reg9  = _mm256_load_ps(&Ay[SIMD_SIZE]);
  reg10 = _mm256_load_ps(&Bx[SIMD_SIZE]);
  reg11 = _mm256_load_ps(&By[SIMD_SIZE]);
  reg12 = _mm256_load_ps(&Cx[SIMD_SIZE]);
  reg13 = _mm256_load_ps(&Cy[SIMD_SIZE]);
  reg14 = _mm256_load_ps(&Dx[SIMD_SIZE]);
  reg15 = _mm256_load_ps(&Dy[SIMD_SIZE]);

  __asm__ volatile("" ::: "memory");
  // Ax - Dx
  reg0 =   _mm256_sub_ps(reg0, reg6);
  reg8 =   _mm256_sub_ps(reg8, reg14);
  // Ay - Dy
  reg1 =   _mm256_sub_ps(reg1, reg7);
  reg9 =   _mm256_sub_ps(reg9, reg15);
  // Bx - Dx
  reg2 =   _mm256_sub_ps(reg2, reg6);
  reg10 =  _mm256_sub_ps(reg10, reg14);
  // By - Dy
  reg3 =   _mm256_sub_ps(reg3, reg7);
  reg11 =  _mm256_sub_ps(reg11, reg15);
  // Cx - Dx
  reg4 =   _mm256_sub_ps(reg4, reg6);
  reg12 =  _mm256_sub_ps(reg12, reg14);
  // Cy - Dy
  reg5 =   _mm256_sub_ps(reg5, reg7);
  reg13 =  _mm256_sub_ps(reg13, reg15);

  __asm__ volatile("" ::: "memory");

  // // Store a
  a[0] = reg0;
  a[1] = reg8;
  // // Store b
  b[0] = reg1;
  b[1] = reg9;
  // // Store d
  d[0] = reg2;
  d[1] = reg10;
  // // Store e
  e[0] = reg3;
  e[1] = reg11;
  // // Store g
  g[0] = reg4;
  g[1] = reg12;
  // // Store h
  h[0] = reg5;
  h[1] = reg13;
}

static inline void kernel_square_add
(
  __m256*      a,
  __m256*      b,
  __m256*      c,
  __m256*      d,
  __m256*      e,
  __m256*      f,
  __m256*      g,
  __m256*      h,
  __m256*      i
){
  __m256 reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  __m256 reg8, reg9, reg10, reg11, reg12, reg13, reg14, reg15;

  reg0 = a[0];
  reg1 = b[0];
  reg2 = d[0];
  reg3 = e[0];
  reg4 = g[0];
  reg5 = h[0];

  reg6 = a[1];
  reg7 = b[1];
  reg8 = d[1];
  reg9 = e[1];
  reg10 = g[1];
  reg11 = h[1];

  // a * a
  reg0 = _mm256_mul_ps(reg0, reg0);
  reg6 = _mm256_mul_ps(reg6, reg6);
  // b * b
  reg1 = _mm256_mul_ps(reg1, reg1);
  reg7 = _mm256_mul_ps(reg7, reg7);
  // d * d
  reg2 = _mm256_mul_ps(reg2, reg2);
  reg8 = _mm256_mul_ps(reg8, reg8);
  // e * e
  reg3 = _mm256_mul_ps(reg3, reg3);
  reg9 = _mm256_mul_ps(reg9, reg9);
  // g * g
  reg4 = _mm256_mul_ps(reg4, reg4);
  reg10 = _mm256_mul_ps(reg10, reg10);
  // h * h
  reg5 = _mm256_mul_ps(reg5, reg5);
  reg11 = _mm256_mul_ps(reg11, reg11);

  // c = a^2 + b^2
  reg0 = _mm256_add_ps(reg0, reg1);
  reg6 = _mm256_add_ps(reg6, reg7);
  // f = d^2 + e^2
  reg2 = _mm256_add_ps(reg2, reg3);
  reg8 = _mm256_add_ps(reg8, reg9);
  // i = g^2 + h^2
  reg4 = _mm256_add_ps(reg4, reg5);
  reg10 = _mm256_add_ps(reg10, reg11);

  __asm__ volatile("" ::: "memory");

  // Store c
  c[0] = reg0;
  c[1] = reg6;
  // Store f
  f[0] = reg2;
  f[1] = reg8;
  // Store i
  i[0] = reg4;
  i[1] = reg10;
}

static inline void kernel_det2
(
  __m256*      a,
  __m256*      b,
  __m256*      c,
  __m256*      d,
  __m256*      e,
  __m256*      f,
  __m256*      g,
  __m256*      h,
  __m256*      i,
  __m256*      det2_out1,
  __m256*      det2_out2,
  __m256*      det2_out3
){
  __m256 reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  __m256 reg8, reg9, reg10, reg11, reg12, reg13, reg14, reg15;

  reg0 = d[0];
  reg1 = e[0];
  reg2 = f[0];
  reg3 = g[0];
  reg4 = h[0];
  reg5 = i[0];

  reg6 = d[1];
  reg7 = e[1];
  reg8 = f[1];
  reg9 = g[1];
  reg10 = h[1];
  reg11 = i[1];

  // ei = e * i
  reg12 = _mm256_mul_ps(reg1, reg5);
  reg13 = _mm256_mul_ps(reg7, reg11);

  // eg = e * g
  reg14 = _mm256_mul_ps(reg1, reg3);
  reg15 = _mm256_mul_ps(reg7, reg9);
  // reg1 and reg7 now free

  // di = d * i
  reg1 = _mm256_mul_ps(reg0, reg5);
  reg7 = _mm256_mul_ps(reg6, reg11);
  // reg5, reg11 now free

  // fg = f * g
  reg5 = _mm256_mul_ps(reg2, reg3);
  reg11 = _mm256_mul_ps(reg8, reg9);
  // reg3, reg9 now free

  // dh = d*h
  reg3 = _mm256_mul_ps(reg0, reg4);
  reg9 = _mm256_mul_ps(reg6, reg10);
  // reg0, reg6 now free

  // fh = f*h
  reg0 = _mm256_mul_ps(reg2, reg4);
  reg6 = _mm256_mul_ps(reg8, reg10);
  // reg2, reg4, reg8, reg10 now free

  /*** Load a,b,c when I can ***/
  reg2 = a[1];
  reg4 = b[0];
  reg8 = c[0];
  reg10 = c[1];

  // ei - fh
  reg12 = _mm256_sub_ps(reg12, reg0);
  reg13 = _mm256_sub_ps(reg13, reg6);
  // reg0, reg6 now free

  /*** Load a,b,c when I can ***/
  reg0 = a[0];
  reg6 = b[1];

  // fg - di
  reg5 = _mm256_sub_ps(reg5, reg1);
  reg11 = _mm256_sub_ps(reg11, reg7);
  // reg1, reg7 now free

  // dh - eg
  reg3 = _mm256_sub_ps(reg3, reg14);
  reg9 = _mm256_sub_ps(reg9, reg15);
  // reg14, reg15 now free

  // a(ei-fh)
  reg0 = _mm256_mul_ps(reg0, reg12);
  reg2 = _mm256_mul_ps(reg2, reg13);

  // b(fg-di)
  reg4 = _mm256_mul_ps(reg4, reg5);
  reg6 = _mm256_mul_ps(reg6, reg11);

  // c(dh-eg)
  reg8 = _mm256_mul_ps(reg8, reg3);
  reg10 = _mm256_mul_ps(reg10, reg9);
  
  // Store det2_out1
  det2_out1[0] = reg0;
  det2_out1[1] = reg2;
  // Store det2_out2
  det2_out2[0] = reg4;
  det2_out2[1] = reg6;
  // Store det2_out3
  det2_out3[0] = reg8;
  det2_out3[1] = reg10;
}

void kernel
(
  float*     restrict Ax,
  float*     restrict Ay,
  float*     restrict Bx,
  float*     restrict By,
  float*     restrict Cx,
  float*     restrict Cy,
  float*     restrict Dx,
  float*     restrict Dy,
  float*     restrict det3_out
){
  
  __m256 a[2], b[2], c[2], d[2], e[2], f[2], g[2], h[2], i[2], det2_out1[2], det2_out2[2], det2_out3[2];
  __m256 reg0, reg1, reg2, reg3, reg4, reg5;
  int idx = 0;

  // for (int p = 0; p < (int)((NUM_ELEMS*SIMD_SIZE)/KERNEL_SIZE); p++){
  //   idx = KERNEL_SIZE*p;
    
  //   // part 1
  //   kernel_sub(&Ax[idx], &Ay[idx], &Bx[idx], &By[idx], &Cx[idx], &Cy[idx], &Dx[idx], &Dy[idx], \
  //             a, b, d, e, g, h);
  //   kernel_square_add(a, b, c, d, e, f, g, h, i);
  //   kernel_det2(a, b, c, d, e, f, g, h, i, \
  //              det2_out1, det2_out2, det2_out3);

  //   // part 2
  //   reg0 = det2_out1[0];
  //   reg1 = det2_out1[1];
  //   reg2 = det2_out2[0];
  //   reg3 = det2_out2[1];
  //   reg4 = det2_out3[0];
  //   reg5 = det2_out3[1];

  //   reg0 = _mm256_add_ps(reg0, reg2);
  //   reg1 = _mm256_add_ps(reg1, reg3);
  //   reg0 = _mm256_add_ps(reg0, reg4);
  //   reg1 = _mm256_add_ps(reg1, reg5);

  //   // Store det3_out
  //   _mm256_store_ps(&det3_out[idx], reg0);
  //   _mm256_store_ps(&det3_out[idx+SIMD_SIZE], reg1);
  // }

  // part 1
  kernel_sub(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, \
            a, b, d, e, g, h);
  kernel_square_add(a, b, c, d, e, f, g, h, i);
  kernel_det2(a, b, c, d, e, f, g, h, i, \
              det2_out1, det2_out2, det2_out3);

  // part 2
  reg0 = det2_out1[0];
  reg1 = det2_out1[1];
  reg2 = det2_out2[0];
  reg3 = det2_out2[1];
  reg4 = det2_out3[0];
  reg5 = det2_out3[1];

  reg0 = _mm256_add_ps(reg0, reg2);
  reg1 = _mm256_add_ps(reg1, reg3);
  reg0 = _mm256_add_ps(reg0, reg4);
  reg1 = _mm256_add_ps(reg1, reg5);

  // Store det3_out
  _mm256_store_ps(&det3_out[0], reg0);
  _mm256_store_ps(&det3_out[0+SIMD_SIZE], reg1);
}

#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

static inline void kernel_sub
(
  int               m,
  int               n,
  int               k,
  double*     restrict Ax,
  double*     restrict Ay,
  double*     restrict Bx,
  double*     restrict By,
  double*     restrict Cx,
  double*     restrict Cy,
  double*     restrict Dx,
  double*     restrict Dy,
  __m256d*      a,
  __m256d*      b,
  __m256d*      d,
  __m256d*      e,
  __m256d*      g,
  __m256d*      h
){

  __m256d reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  __m256d reg8, reg9, reg10, reg11, reg12, reg13, reg14, reg15;

  reg0 = _mm256_load_pd(&Ax[0]);
  reg1 = _mm256_load_pd(&Ay[0]);
  reg2 = _mm256_load_pd(&Bx[0]);
  reg3 = _mm256_load_pd(&By[0]);
  reg4 = _mm256_load_pd(&Cx[0]);
  reg5 = _mm256_load_pd(&Cy[0]);
  reg6 = _mm256_load_pd(&Dx[0]);
  reg7 = _mm256_load_pd(&Dy[0]);

  reg8  = _mm256_load_pd(&Ax[4]);
  reg9  = _mm256_load_pd(&Ay[4]);
  reg10 = _mm256_load_pd(&Bx[4]);
  reg11 = _mm256_load_pd(&By[4]);
  reg12 = _mm256_load_pd(&Cx[4]);
  reg13 = _mm256_load_pd(&Cy[4]);
  reg14 = _mm256_load_pd(&Dx[4]);
  reg15 = _mm256_load_pd(&Dy[4]);

  // Ax - Dx
  reg0 =   _mm256_sub_pd(reg0, reg6);
  reg8 =   _mm256_sub_pd(reg8, reg14);
  // Ay - Dy
  reg1 =   _mm256_sub_pd(reg1, reg7);
  reg9 =   _mm256_sub_pd(reg9, reg15);
  // Bx - Dx
  reg2 =   _mm256_sub_pd(reg2, reg6);
  reg10 =  _mm256_sub_pd(reg10, reg14);
  // By - Dy
  reg3 =   _mm256_sub_pd(reg3, reg7);
  reg11 =  _mm256_sub_pd(reg11, reg15);
  // Cx - Dx
  reg4 =   _mm256_sub_pd(reg4, reg6);
  reg12 =  _mm256_sub_pd(reg12, reg14);
  // Cy - Dy
  reg5 =   _mm256_sub_pd(reg5, reg7);
  reg13 =  _mm256_sub_pd(reg13, reg15);

  // // Store a
  // _mm256_store_pd(&a[0], reg0);
  // _mm256_store_pd(&a[4], reg8);
  a[0] = reg0;
  a[1] = reg8;
  // // Store b
  // _mm256_store_pd(&b[0], reg1);
  // _mm256_store_pd(&b[4], reg9);
  b[0] = reg1;
  b[1] = reg9;
  // // Store d
  // _mm256_store_pd(&d[0], reg2);
  // _mm256_store_pd(&d[4], reg10);
  d[0] = reg2;
  d[1] = reg10;
  // // Store e
  // _mm256_store_pd(&e[0], reg3);
  // _mm256_store_pd(&e[4], reg11);
  e[0] = reg3;
  e[1] = reg11;
  // // Store g
  // _mm256_store_pd(&g[0], reg4);
  // _mm256_store_pd(&g[4], reg12);
  g[0] = reg4;
  g[1] = reg12;
  // // Store h
  // _mm256_store_pd(&h[0], reg5);
  // _mm256_store_pd(&h[4], reg13);
  h[0] = reg5;
  h[1] = reg13;
}

static inline void kernel_square_add
(
  int               m,
  int               n,
  int               k,
  __m256d*      a,
  __m256d*      b,
  __m256d*      c,
  __m256d*      d,
  __m256d*      e,
  __m256d*      f,
  __m256d*      g,
  __m256d*      h,
  __m256d*      i
){
  __m256d reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  __m256d reg8, reg9, reg10, reg11, reg12, reg13, reg14, reg15;

  // reg0 = _mm256_load_pd(&a[0]);
  // reg1 = _mm256_load_pd(&b[0]);
  // reg2 = _mm256_load_pd(&d[0]);
  // reg3 = _mm256_load_pd(&e[0]);
  // reg4 = _mm256_load_pd(&g[0]);
  // reg5 = _mm256_load_pd(&h[0]);

  // reg6 = _mm256_load_pd(&a[4]);
  // reg7 = _mm256_load_pd(&b[4]);
  // reg8 = _mm256_load_pd(&d[4]);
  // reg9 = _mm256_load_pd(&e[4]);
  // reg10 = _mm256_load_pd(&g[4]);
  // reg11 = _mm256_load_pd(&h[4]);
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
  reg0 = _mm256_mul_pd(reg0, reg0);
  reg6 = _mm256_mul_pd(reg6, reg6);
  // b * b
  reg1 = _mm256_mul_pd(reg1, reg1);
  reg7 = _mm256_mul_pd(reg7, reg7);
  // d * d
  reg2 = _mm256_mul_pd(reg2, reg2);
  reg8 = _mm256_mul_pd(reg8, reg8);
  // e * e
  reg3 = _mm256_mul_pd(reg3, reg3);
  reg9 = _mm256_mul_pd(reg9, reg9);
  // g * g
  reg4 = _mm256_mul_pd(reg4, reg4);
  reg10 = _mm256_mul_pd(reg10, reg10);
  // h * h
  reg5 = _mm256_mul_pd(reg5, reg5);
  reg11 = _mm256_mul_pd(reg11, reg11);

  // c = a^2 + b^2
  reg0 = _mm256_add_pd(reg0, reg1);
  reg6 = _mm256_add_pd(reg6, reg7);
  // f = d^2 + e^2
  reg2 = _mm256_add_pd(reg2, reg3);
  reg8 = _mm256_add_pd(reg8, reg9);
  // i = g^2 + h^2
  reg4 = _mm256_add_pd(reg4, reg5);
  reg10 = _mm256_add_pd(reg10, reg11);

  // // Store c
  // _mm256_store_pd(&c[0], reg0);
  // _mm256_store_pd(&c[4], reg6);
  c[0] = reg0;
  c[1] = reg6;
  // // Store f
  // _mm256_store_pd(&f[0], reg2);
  // _mm256_store_pd(&f[4], reg8);
  f[0] = reg2;
  f[1] = reg8;
  // // Store i
  // _mm256_store_pd(&i[0], reg4);
  // _mm256_store_pd(&i[4], reg10);
  i[0] = reg4;
  i[1] = reg10;
}

static inline void kernel_det2
(
  int               m,
  int               n,
  int               k,
  __m256d*      a,
  __m256d*      b,
  __m256d*      c,
  __m256d*      d,
  __m256d*      e,
  __m256d*      f,
  __m256d*      g,
  __m256d*      h,
  __m256d*      i,
  __m256d*      det2_out1,
  __m256d*      det2_out2,
  __m256d*      det2_out3
){
  __m256d reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  __m256d reg8, reg9, reg10, reg11, reg12, reg13, reg14, reg15;

  // reg0 = _mm256_load_pd(&d[0]);
  // reg1 = _mm256_load_pd(&e[0]);
  // reg2 = _mm256_load_pd(&f[0]);
  // reg3 = _mm256_load_pd(&g[0]);
  // reg4 = _mm256_load_pd(&h[0]);
  // reg5 = _mm256_load_pd(&i[0]);

  // reg6 = _mm256_load_pd(&d[4]);
  // reg7 = _mm256_load_pd(&e[4]);
  // reg8 = _mm256_load_pd(&f[4]);
  // reg9 = _mm256_load_pd(&g[4]);
  // reg10 = _mm256_load_pd(&h[4]);
  // reg11 = _mm256_load_pd(&i[4]);
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
  reg12 = _mm256_mul_pd(reg1, reg5);
  reg13 = _mm256_mul_pd(reg7, reg11);

  // eg = e * g
  reg14 = _mm256_mul_pd(reg1, reg3);
  reg15 = _mm256_mul_pd(reg7, reg9);
  // reg1 and reg7 now free

  // di = d * i
  reg1 = _mm256_mul_pd(reg0, reg5);
  reg7 = _mm256_mul_pd(reg6, reg11);
  // reg5, reg11 now free

  // fg = f * g
  reg5 = _mm256_mul_pd(reg2, reg3);
  reg11 = _mm256_mul_pd(reg8, reg9);
  // reg3, reg9 now free

  // dh = d*h
  reg3 = _mm256_mul_pd(reg0, reg4);
  reg9 = _mm256_mul_pd(reg6, reg10);
  // reg0, reg6 now free

  // fh = f*h
  reg0 = _mm256_mul_pd(reg2, reg4);
  reg6 = _mm256_mul_pd(reg8, reg10);
  // reg2, reg4, reg8, reg10 now free

  /*** Load a,b,c when I can ***/
  // reg2 = _mm256_load_pd(&a[4]);
  // reg4 = _mm256_load_pd(&b[0]);
  // reg8 = _mm256_load_pd(&c[0]);
  // reg10 = _mm256_load_pd(&c[4]);
  reg2 = a[1];
  reg4 = b[0];
  reg8 = c[0];
  reg10 = c[1];

  // ei - fh
  reg12 = _mm256_sub_pd(reg12, reg0);
  reg13 = _mm256_sub_pd(reg13, reg6);
  // reg0, reg6 now free

  /*** Load a,b,c when I can ***/
  // reg0 = _mm256_load_pd(&a[0]);
  // reg6 = _mm256_load_pd(&b[4]);
  reg0 = a[0];
  reg6 = b[1];

  // fg - di
  reg5 = _mm256_sub_pd(reg5, reg1);
  reg11 = _mm256_sub_pd(reg11, reg7);
  // reg1, reg7 now free

  // dh - eg
  reg3 = _mm256_sub_pd(reg3, reg14);
  reg9 = _mm256_sub_pd(reg9, reg15);
  // reg14, reg15 now free

  // a(ei-fh)
  reg0 = _mm256_mul_pd(reg0, reg12);
  reg2 = _mm256_mul_pd(reg2, reg13);

  // b(fg-di)
  reg4 = _mm256_mul_pd(reg4, reg5);
  reg6 = _mm256_mul_pd(reg6, reg11);

  // c(dh-eg)
  reg8 = _mm256_mul_pd(reg8, reg3);
  reg10 = _mm256_mul_pd(reg10, reg9);
  
  // // Store det2_out1
  // _mm256_store_pd(&det2_out1[0], reg0);
  // _mm256_store_pd(&det2_out1[4], reg2);
  det2_out1[0] = reg0;
  det2_out1[1] = reg2;
  // // Store det2_out2
  // _mm256_store_pd(&det2_out2[0], reg4);
  // _mm256_store_pd(&det2_out2[4], reg6);
  det2_out2[0] = reg4;
  det2_out2[1] = reg6;
  // // Store det2_out3
  // _mm256_store_pd(&det2_out3[0], reg8);
  // _mm256_store_pd(&det2_out3[4], reg10);
  det2_out3[0] = reg8;
  det2_out3[1] = reg10;
}

void kernel
(
  int               m,
  int               n,
  int               k,
  double*     restrict Ax,
  double*     restrict Ay,
  double*     restrict Bx,
  double*     restrict By,
  double*     restrict Cx,
  double*     restrict Cy,
  double*     restrict Dx,
  double*     restrict Dy,
  double*     restrict det3_out
){
  
  __m256d a[2], b[2], c[2], d[2], e[2], f[2], g[2], h[2], i[2], det2_out1[2], det2_out2[2], det2_out3[2];
  __m256d reg0, reg1, reg2, reg3, reg4, reg5;
  int idx = 0;

  for (int p = 0; p < k; p++){
    idx = m*n*p;
    // part 1
    kernel_sub(m, n, k, &Ax[idx], &Ay[idx], &Bx[idx], &By[idx], &Cx[idx], &Cy[idx], &Dx[idx], &Dy[idx], \
              a, b, d, e, g, h);
    kernel_square_add(m, n, k, a, b, c, d, e, f, g, h, i);
    kernel_det2(m, n, k, a, b, c, d, e, f, g, h, i, \
               det2_out1, det2_out2, det2_out3);

    // part 2
    // reg0 = _mm256_load_pd(&det2_out1[idx]);
    // reg1 = _mm256_load_pd(&det2_out1[idx+4]);
    // reg2 = _mm256_load_pd(&det2_out2[idx]);
    // reg3 = _mm256_load_pd(&det2_out2[idx+4]);
    // reg4 = _mm256_load_pd(&det2_out3[idx]);
    // reg5 = _mm256_load_pd(&det2_out3[idx+4]);
    reg0 = det2_out1[0];
    reg1 = det2_out1[1];
    reg2 = det2_out2[0];
    reg3 = det2_out2[1];
    reg4 = det2_out3[0];
    reg5 = det2_out3[1];

    reg0 = _mm256_add_pd(reg0, reg2);
    reg1 = _mm256_add_pd(reg1, reg3);
    reg0 = _mm256_add_pd(reg0, reg4);
    reg1 = _mm256_add_pd(reg1, reg5);
    // CMP

    // Store det3_out
    _mm256_store_pd(&det3_out[idx], reg0);
    _mm256_store_pd(&det3_out[idx+4], reg1);
  }
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

void kernel_sub
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
  double*     restrict a,
  double*     restrict b,
  double*     restrict d,
  double*     restrict e,
  double*     restrict g,
  double*     restrict h
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

  // Store a
  _mm256_store_pd(&a[0], reg0);
  _mm256_store_pd(&a[4], reg8);
  // Store b
  _mm256_store_pd(&b[0], reg1);
  _mm256_store_pd(&b[4], reg9);
  // Store d
  _mm256_store_pd(&d[0], reg2);
  _mm256_store_pd(&d[4], reg10);
  // Store e
  _mm256_store_pd(&e[0], reg3);
  _mm256_store_pd(&e[4], reg11);
  // Store g
  _mm256_store_pd(&g[0], reg4);
  _mm256_store_pd(&g[4], reg12);
  // Store h
  _mm256_store_pd(&h[0], reg5);
  _mm256_store_pd(&h[4], reg13);
}

void kernel_square_add
(
  int               m,
  int               n,
  int               k,
  double*     restrict a,
  double*     restrict b,
  double*     restrict c,
  double*     restrict d,
  double*     restrict e,
  double*     restrict f,
  double*     restrict g,
  double*     restrict h,
  double*     restrict i
){
  __m256d reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  __m256d reg8, reg9, reg10, reg11, reg12, reg13, reg14, reg15;

  reg0 = _mm256_load_pd(&a[0]);
  reg1 = _mm256_load_pd(&b[0]);
  reg2 = _mm256_load_pd(&d[0]);
  reg3 = _mm256_load_pd(&e[0]);
  reg4 = _mm256_load_pd(&g[0]);
  reg5 = _mm256_load_pd(&h[0]);

  reg6 = _mm256_load_pd(&a[4]);
  reg7 = _mm256_load_pd(&b[4]);
  reg8 = _mm256_load_pd(&d[4]);
  reg9 = _mm256_load_pd(&e[4]);
  reg10 = _mm256_load_pd(&g[4]);
  reg11 = _mm256_load_pd(&h[4]);

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

  // Store c
  _mm256_store_pd(&c[0], reg0);
  _mm256_store_pd(&c[4], reg6);
  // Store f
  _mm256_store_pd(&f[0], reg2);
  _mm256_store_pd(&f[4], reg8);
  // Store i
  _mm256_store_pd(&i[0], reg4);
  _mm256_store_pd(&i[4], reg10);
}

void kernel_det2
(
  int               m,
  int               n,
  int               k,
  double*     restrict a,
  double*     restrict b,
  double*     restrict c,
  double*     restrict d,
  double*     restrict e,
  double*     restrict f,
  double*     restrict g,
  double*     restrict h,
  double*     restrict i,
  double*     restrict det2_out1,
  double*     restrict det2_out2,
  double*     restrict det2_out3
){
  __m256d reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  __m256d reg8, reg9, reg10, reg11, reg12, reg13, reg14, reg15;

  reg0 = _mm256_load_pd(&d[0]);
  reg1 = _mm256_load_pd(&e[0]);
  reg2 = _mm256_load_pd(&f[0]);
  reg3 = _mm256_load_pd(&g[0]);
  reg4 = _mm256_load_pd(&h[0]);
  reg5 = _mm256_load_pd(&i[0]);

  reg6 = _mm256_load_pd(&d[4]);
  reg7 = _mm256_load_pd(&e[4]);
  reg8 = _mm256_load_pd(&f[4]);
  reg9 = _mm256_load_pd(&g[4]);
  reg10 = _mm256_load_pd(&h[4]);
  reg11 = _mm256_load_pd(&i[4]);

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
  reg2 = _mm256_load_pd(&a[4]);
  reg4 = _mm256_load_pd(&b[0]);
  reg8 = _mm256_load_pd(&c[0]);
  reg10 = _mm256_load_pd(&c[4]);

  // ei - fh
  reg12 = _mm256_sub_pd(reg12, reg0);
  reg13 = _mm256_sub_pd(reg13, reg6);
  // reg0, reg6 now free

  /*** Load a,b,c when I can ***/
  reg0 = _mm256_load_pd(&a[0]);
  reg6 = _mm256_load_pd(&b[4]);

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
  
  // Store det2_out1
  _mm256_store_pd(&det2_out1[0], reg0);
  _mm256_store_pd(&det2_out1[4], reg2);
  // Store det2_out2
  _mm256_store_pd(&det2_out2[0], reg4);
  _mm256_store_pd(&det2_out2[4], reg6);
  // Store det2_out3
  _mm256_store_pd(&det2_out3[0], reg8);
  _mm256_store_pd(&det2_out3[4], reg10);
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
  double *a, *b, *c, *d, *e, *f, *g, *h, *i, *det2_out1, *det2_out2, *det2_out3;

  //create memory aligned buffers
  posix_memalign((void**) &a, 64, m * n * k * sizeof(double));
  posix_memalign((void**) &b, 64, m * n * k * sizeof(double));
  posix_memalign((void**) &c, 64, m * n * k * sizeof(double));
  posix_memalign((void**) &d, 64, m * n * k * sizeof(double));
  posix_memalign((void**) &e, 64, m * n * k * sizeof(double));
  posix_memalign((void**) &f, 64, m * n * k * sizeof(double));
  posix_memalign((void**) &g, 64, m * n * k * sizeof(double));
  posix_memalign((void**) &h, 64, m * n * k * sizeof(double));
  posix_memalign((void**) &i, 64, m * n * k * sizeof(double));
  posix_memalign((void**) &det2_out1, 64, m * n * k * sizeof(double));
  posix_memalign((void**) &det2_out2, 64, m * n * k * sizeof(double));
  posix_memalign((void**) &det2_out3, 64, m * n * k * sizeof(double));

  // Initialize buffers
  for (int j = 0; j < k * m * n; j++){
    a[j] = 0.0;
    b[j] = 0.0;
    c[j] = 0.0;
    d[j] = 0.0;
    e[j] = 0.0;
    f[j] = 0.0;
    g[j] = 0.0;
    h[j] = 0.0;
    i[j] = 0.0;
    det2_out1[j] = 0.0;
    det2_out2[j] = 0.0;
    det2_out3[j] = 0.0;
  }

  __m256d reg0, reg1, reg2, reg3, reg4, reg5;
  int idx = 0;

  for (int p = 0; p < k; p++){
    idx = m*n*p;
    kernel_sub(m, n, k, &Ax[idx], &Ay[idx], &Bx[idx], &By[idx], &Cx[idx], &Cy[idx], &Dx[idx], &Dy[idx], \
              &a[idx], &b[idx], &d[idx], &e[idx], &g[idx], &h[idx]);
    kernel_square_add(m, n, k, &a[idx], &b[idx], &c[idx], &d[idx], &e[idx], &f[idx], &g[idx], &h[idx], &i[idx]);
    kernel_det2(m, n, k, &a[idx], &b[idx], &c[idx], &d[idx], &e[idx], &f[idx], &g[idx], &h[idx], &i[idx], \
               &det2_out1[idx], &det2_out2[idx], &det2_out3[idx]);

    reg0 = _mm256_load_pd(&det2_out1[idx]);
    reg1 = _mm256_load_pd(&det2_out1[idx+4]);
    reg2 = _mm256_load_pd(&det2_out2[idx]);
    reg3 = _mm256_load_pd(&det2_out2[idx+4]);
    reg4 = _mm256_load_pd(&det2_out3[idx]);
    reg5 = _mm256_load_pd(&det2_out3[idx+4]);

    reg0 = _mm256_add_pd(reg0, reg2);
    reg1 = _mm256_add_pd(reg1, reg3);
    reg0 = _mm256_add_pd(reg0, reg4);
    reg1 = _mm256_add_pd(reg1, reg5);

    // Store det3_out
    _mm256_store_pd(&det3_out[idx], reg0);
    _mm256_store_pd(&det3_out[idx+4], reg1);
  }

  free(a);
  free(b);
  free(c);
  free(d);
  free(e);
  free(f);
  free(g);
  free(h);
  free(i);
  free(det2_out1);
  free(det2_out2);
  free(det2_out3);
}

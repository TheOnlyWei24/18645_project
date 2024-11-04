#ifndef __CIRCUMCENTER_KERNEL_H_
#define __CIRCUMCENTER_KERNEL_H_

#include <immintrin.h>

void kernel0(double *Ax, double *Ay, double *Bx, double *By, double *Cx,
             double *Cy, double *partUx, double *partUy, double *partD) {
  __m256d reg0 = _mm256_load_pd(Ax);
  __m256d reg1 = _mm256_load_pd(Ay);
  __m256d reg2 = _mm256_load_pd(Bx);
  __m256d reg3 = _mm256_load_pd(By);
  __m256d reg4 = _mm256_load_pd(Cx);
  __m256d reg5 = _mm256_load_pd(Cy);

  __m256d reg6 = _mm256_mul_pd(reg0, reg0);  // Ax^2
  __m256d reg7 = _mm256_mul_pd(reg1, reg1);  // Ay^2
  __m256d reg8 = _mm256_mul_pd(reg2, reg2);  // Bx^2
  __m256d reg9 = _mm256_mul_pd(reg3, reg3);  // By^2
  __m256d reg10 = _mm256_mul_pd(reg4, reg4); // Cx^2
  __m256d reg11 = _mm256_mul_pd(reg5, reg5); // Cy^2

  reg6 = _mm256_add_pd(reg6, reg7);   // Ax^2 + Ay^2
  reg7 = _mm256_add_pd(reg8, reg9);   // Bx^2 + By^2
  reg8 = _mm256_add_pd(reg10, reg11); // Cx^2 + Cy^2

  reg9 = _mm256_sub_pd(reg3, reg5);  // By - Cy
  reg10 = _mm256_sub_pd(reg5, reg1); // Cy - Ay
  reg11 = _mm256_sub_pd(reg1, reg3); // Ay - By

  reg1 = _mm256_sub_pd(reg4, reg2); // Cx - Bx
  reg3 = _mm256_sub_pd(reg0, reg4); // Ax - Cx
  reg5 = _mm256_sub_pd(reg2, reg0); // Bx - Ax

  reg0 = _mm256_mul_pd(reg9, reg0);  // (By - Cy) * Ax
  reg2 = _mm256_mul_pd(reg10, reg2); // (Cy - Ay) * Bx
  reg4 = _mm256_mul_pd(reg11, reg4); // (Ay - By) * Cx

  reg9 = _mm256_mul_pd(reg6, reg9);   // (Ax^2 + Ay^2) * (By - Cy)
  reg10 = _mm256_mul_pd(reg7, reg10); // (Bx^2 + By^2) * (Cy - Ay)
  reg11 = _mm256_mul_pd(reg8, reg11); // (Cx^2 + Cy^2) * (Ay - By)

  reg1 = _mm256_mul_pd(reg6, reg1); // (Ax^2 + Ay^2) * (Cx - Bx)
  reg3 = _mm256_mul_pd(reg7, reg3); // (Bx^2 + By^2) * (Ax - Cx)
  reg5 = _mm256_mul_pd(reg8, reg5); // (Cx^2 + Cy^2) * (Bx - Ax)

  // reg6, reg7, reg8 are free
  // reg12, reg13, reg14, reg 15 are free
  // TODO : Could begin second round of kernel here, ammortize cost of loads and
  // stores

  reg6 = _mm256_add_pd(reg0, reg2);  // D0 + D1
  reg7 = _mm256_add_pd(reg9, reg10); // Ux0 + Ux1
  reg8 = _mm256_add_pd(reg1, reg3);  // Uy0 + Uy1

  reg6 = _mm256_add_pd(reg6, reg4);  // D0 + D1 + D2
  reg7 = _mm256_add_pd(reg7, reg11); // Ux0 + Ux1 + Ux2
  reg8 = _mm256_add_pd(reg8, reg5);  // Uy0 + Uy1 + Uy2

  _mm256_store_pd(partD, reg6);
  _mm256_store_pd(partUx, reg7);
  _mm256_store_pd(partUy, reg8);
}

void kernel1(double *partD, double *partUx, double *partUy, double *Ux,
             double *Uy) {
  // TODO: Make this work for multiple values to ammosrtize the cost of division
  __m256d reg0 = _mm256_load_pd(partD);
  __m256d reg1 = _mm256_load_pd(partUx);
  __m256d reg2 = _mm256_load_pd(partUy);

  double two[] = {2.0, 2.0, 2.0, 2.0};
  __m256d reg3 = _mm256_load_pd(&two[0]);
  reg0 = _mm256_mul_pd(reg0, reg3); // D

  reg1 = _mm256_div_pd(reg1, reg0); // Ux
  reg2 = _mm256_div_pd(reg2, reg0); // Uy

  _mm256_store_pd(Ux, reg1);
  _mm256_store_pd(Uy, reg2);
}

#endif

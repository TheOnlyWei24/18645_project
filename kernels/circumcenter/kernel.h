#ifndef __CIRCUMCENTER_KERNEL_H_
#define __CIRCUMCENTER_KERNEL_H_

#include <immintrin.h>

#define SIMD_SIZE 8

void kernel0(float *Ax, float *Ay, float *Bx, float *By, float *Cx, float *Cy,
             float *partUx, float *partUy, float *partD) {
  // First half of first kernel
  __m256 reg0 = _mm256_load_ps(Ax);
  __m256 reg1 = _mm256_load_ps(Ay);
  __m256 reg2 = _mm256_load_ps(Bx);
  __m256 reg3 = _mm256_load_ps(By);
  __m256 reg4 = _mm256_load_ps(Cx);
  __m256 reg5 = _mm256_load_ps(Cy);

  __m256 reg6 = _mm256_mul_ps(reg0, reg0);  // Ax^2
  __m256 reg7 = _mm256_mul_ps(reg1, reg1);  // Ay^2
  __m256 reg8 = _mm256_mul_ps(reg2, reg2);  // Bx^2
  __m256 reg9 = _mm256_mul_ps(reg3, reg3);  // By^2
  __m256 reg10 = _mm256_mul_ps(reg4, reg4); // Cx^2
  __m256 reg11 = _mm256_mul_ps(reg5, reg5); // Cy^2

  reg6 = _mm256_add_ps(reg6, reg7);   // Ax^2 + Ay^2
  reg7 = _mm256_add_ps(reg8, reg9);   // Bx^2 + By^2
  reg8 = _mm256_add_ps(reg10, reg11); // Cx^2 + Cy^2

  reg9 = _mm256_sub_ps(reg3, reg5);  // By - Cy
  reg10 = _mm256_sub_ps(reg5, reg1); // Cy - Ay
  reg11 = _mm256_sub_ps(reg1, reg3); // Ay - By

  reg1 = _mm256_sub_ps(reg4, reg2); // Cx - Bx
  reg3 = _mm256_sub_ps(reg0, reg4); // Ax - Cx
  reg5 = _mm256_sub_ps(reg2, reg0); // Bx - Ax

  reg0 = _mm256_mul_ps(reg9, reg0);  // (By - Cy) * Ax
  reg2 = _mm256_mul_ps(reg10, reg2); // (Cy - Ay) * Bx
  reg4 = _mm256_mul_ps(reg11, reg4); // (Ay - By) * Cx

  reg9 = _mm256_mul_ps(reg6, reg9);   // (Ax^2 + Ay^2) * (By - Cy)
  reg10 = _mm256_mul_ps(reg7, reg10); // (Bx^2 + By^2) * (Cy - Ay)
  reg11 = _mm256_mul_ps(reg8, reg11); // (Cx^2 + Cy^2) * (Ay - By)

  reg1 = _mm256_mul_ps(reg6, reg1); // (Ax^2 + Ay^2) * (Cx - Bx)
  reg3 = _mm256_mul_ps(reg7, reg3); // (Bx^2 + By^2) * (Ax - Cx)
  reg5 = _mm256_mul_ps(reg8, reg5); // (Cx^2 + Cy^2) * (Bx - Ax)

  // reg6, reg7, reg8 are free
  // reg12, reg13, reg14, reg 15 are free
  // Execute loads for second kernel in order to hide the latency
  reg6 = _mm256_load_ps(Ax + SIMD_SIZE);
  reg7 = _mm256_load_ps(Ay + SIMD_SIZE);
  reg8 = _mm256_load_ps(Bx + SIMD_SIZE);
  __m256 reg12 = _mm256_load_ps(By + SIMD_SIZE);
  __m256 reg13 = _mm256_load_ps(Cx + SIMD_SIZE);
  __m256 reg14 = _mm256_load_ps(Cy + SIMD_SIZE);

  // Finish second half of first kernel
  reg0 = _mm256_add_ps(reg0, reg2);  // D0 + D1
  reg2 = _mm256_add_ps(reg9, reg10); // Ux0 + Ux1
  reg1 = _mm256_add_ps(reg1, reg3);  // Uy0 + Uy1

  reg0 = _mm256_add_ps(reg0, reg4);  // D0 + D1 + D2
  reg2 = _mm256_add_ps(reg2, reg11); // Ux0 + Ux1 + Ux2
  reg1 = _mm256_add_ps(reg1, reg5);  // Uy0 + Uy1 + Uy2

  _mm256_store_ps(partD, reg0);
  _mm256_store_ps(partUx, reg2);
  _mm256_store_ps(partUy, reg1);

  // All reg except 6, 7, 8, 12, 13, 14 are free
  // Begin execution of second kernel
  reg0 = _mm256_mul_ps(reg6, reg6);   // (Ax + SIMD_SIZE)^2
  reg1 = _mm256_mul_ps(reg7, reg7);   // (Ay + SIMD_SIZE)^2
  reg2 = _mm256_mul_ps(reg8, reg8);   // (Bx + SIMD_SIZE)^2
  reg3 = _mm256_mul_ps(reg12, reg12); // (By + SIMD_SIZE)^2
  reg4 = _mm256_mul_ps(reg13, reg13); // (Cx + SIMD_SIZE)^2
  reg5 = _mm256_mul_ps(reg14, reg14); // (Cy + SIMD_SIZE)^2

  reg9 = _mm256_add_ps(reg0, reg1);  // (Ax + SIMD_SIZE)^2 + (Ay + SIMD_SIZE)^2
  reg10 = _mm256_add_ps(reg2, reg3); // (Bx + SIMD_SIZE)^2  + (By + SIMD_SIZE)^2
  reg11 = _mm256_add_ps(reg4, reg5); // (Cx + SIMD_SIZE)^2  + (Cy + SIMD_SIZE)^2

  reg0 = _mm256_sub_ps(reg12, reg14); // (By + SIMD_SIZE) - (Cy + SIMD_SIZE)
  reg1 = _mm256_sub_ps(reg14, reg7);  // (Cy + SIMD_SIZE) - (Ay + SIMD_SIZE)
  reg2 = _mm256_sub_ps(reg7, reg12);  // (Ay + SIMD_SIZE) - (By + SIMD_SIZE)

  reg3 = _mm256_sub_ps(reg13, reg8); // (Cx + SIMD_SIZE) - (Bx + SIMD_SIZE)
  reg4 = _mm256_sub_ps(reg6, reg13); // (Ax + SIMD_SIZE) - (Cx + SIMD_SIZE)
  reg5 = _mm256_sub_ps(reg8, reg6);  // (Bx + SIMD_SIZE) - (Ax + SIMD_SIZE)

  // ((By + SIMD_SIZE) - (Cy + SIMD_SIZE)) * (Ax + SIMD_SIZE)
  reg7 = _mm256_mul_ps(reg0, reg6);
  // ((Cy + SIMD_SIZE) - (Ay + SIMD_SIZE)) * (Bx + SIMD_SIZE)
  reg12 = _mm256_mul_ps(reg1, reg8);
  // ((Ay + SIMD_SIZE) - (By + SIMD_SIZE)) * (Cx + SIMD_SIZE)
  reg14 = _mm256_mul_ps(reg2, reg13);

  // ((Ax + SIMD_SIZE)^2 + (Ay + SIMD_SIZE)^2) * ((By + SIMD_SIZE) - (Cy +
  // SIMD_SIZE))
  reg6 = _mm256_mul_ps(reg9, reg0);
  // ((Bx + SIMD_SIZE)^2 + (By + SIMD_SIZE)^2) * ((Cy + SIMD_SIZE) - (Ay +
  // SIMD_SIZE))
  reg8 = _mm256_mul_ps(reg10, reg1);
  // (Cx + SIMD_SIZE)^2  + (Cy + SIMD_SIZE)^2) * ((Ay + SIMD_SIZE) - (By +
  // SIMD_SIZE))
  reg13 = _mm256_mul_ps(reg11, reg2);

  // ((Ax + SIMD_SIZE)^2 + (Ay + SIMD_SIZE)^2) * ((Cx + SIMD_SIZE) - (Bx +
  // SIMD_SIZE))
  reg9 = _mm256_mul_ps(reg3, reg9);
  // ((Bx + SIMD_SIZE)^2  + (By + SIMD_SIZE)^2) * ((Ax + SIMD_SIZE) - (Cx +
  // SIMD_SIZE))
  reg10 = _mm256_mul_ps(reg4, reg10);
  // ((Cx + SIMD_SIZE)^2  + (Cy + SIMD_SIZE)^2) * ((Bx + SIMD_SIZE) - (Ax +
  // SIMD_SIZE))
  reg11 = _mm256_mul_ps(reg5, reg11);

  // reg0, reg1, reg2, reg3, reg4, reg5 are free
  // Execute loads for 3rd kernel
  // TODO

  reg7 = _mm256_add_ps(reg7, reg12);
  reg6 = _mm256_add_ps(reg6, reg8);
  reg9 = _mm256_add_ps(reg9, reg10);

  reg7 = _mm256_add_ps(reg7, reg14);
  reg6 = _mm256_add_ps(reg6, reg13);
  reg9 = _mm256_add_ps(reg9, reg11);

  _mm256_store_ps((partD + SIMD_SIZE), reg7);
  _mm256_store_ps((partUx + SIMD_SIZE), reg6);
  _mm256_store_ps((partUy + SIMD_SIZE), reg9);
}

void kernel1(float *partD, float *partUx, float *partUy, float *Ux, float *Uy) {
  // TODO: Make this work for multiple values to ammosrtize the cost of division
  __m256 reg0 = _mm256_load_ps(partD);
  __m256 reg1 = _mm256_load_ps(partUx);
  __m256 reg2 = _mm256_load_ps(partUy);
  __m256 reg3 = _mm256_load_ps(partD + SIMD_SIZE);
  __m256 reg4 = _mm256_load_ps(partUx + SIMD_SIZE);
  __m256 reg5 = _mm256_load_ps(partUy + SIMD_SIZE);

  float two[] = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
  __m256 reg15 = _mm256_load_ps(&two[0]);
  reg0 = _mm256_mul_ps(reg0, reg15); // D
  reg3 = _mm256_mul_ps(reg3, reg15);

  reg1 = _mm256_div_ps(reg1, reg0); // Ux
  reg2 = _mm256_div_ps(reg2, reg0); // Uy
  reg4 = _mm256_div_ps(reg4, reg3);
  reg5 = _mm256_div_ps(reg5, reg3);

  _mm256_store_ps(Ux, reg1);
  _mm256_store_ps(Uy, reg2);
  _mm256_store_ps((Ux + SIMD_SIZE), reg4);
  _mm256_store_ps((Uy + SIMD_SIZE), reg5);
}

#endif

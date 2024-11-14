#ifndef _DET3_KERNEL_H_
#define __DET3_KERNEL_H_

#include <immintrin.h>

#define SIMD_SIZE 8

void kernel(float *Ax, float *Ay, float *Bx, float *By, float *Cx, float *Cy,
            float *Dx, float *Dy, float *out) {

  /*---------------------------------------------------------------------------
                          Starting 1st kernel
  ---------------------------------------------------------------------------*/
  __m256 reg0 = _mm256_load_ps(Ax);
  __m256 reg1 = _mm256_load_ps(Ay);
  __m256 reg2 = _mm256_load_ps(Bx);
  __m256 reg3 = _mm256_load_ps(By);
  __m256 reg4 = _mm256_load_ps(Cx);
  __m256 reg5 = _mm256_load_ps(Cy);
  __m256 reg6 = _mm256_load_ps(Dx);
  __m256 reg7 = _mm256_load_ps(Dy);

  __m256 reg8 = _mm256_sub_ps(reg0, reg6);  // a = Ax - Dx
  __m256 reg9 = _mm256_sub_ps(reg1, reg7);  // b = Ay - Dy
  __m256 reg10 = _mm256_sub_ps(reg2, reg6); // d = Bx - Dx
  __m256 reg11 = _mm256_sub_ps(reg3, reg7); // e = By - Dy
  __m256 reg12 = _mm256_sub_ps(reg4, reg6); // g = Cx - Dx
  __m256 reg13 = _mm256_sub_ps(reg5, reg7); // h = Cy - Dy

  __m256 reg14 = _mm256_mul_ps(reg8, reg8); // a * a
  __m256 reg15 = _mm256_mul_ps(reg9, reg9); // b * b
  reg0 = _mm256_mul_ps(reg10, reg10);       // d * d
  reg1 = _mm256_mul_ps(reg11, reg11);       // e * e
  reg2 = _mm256_mul_ps(reg12, reg12);       // g * g
  reg3 = _mm256_mul_ps(reg13, reg13);       // h * h

  reg4 = _mm256_add_ps(reg14, reg15); // c = (a * a) + (b * b)
  reg5 = _mm256_add_ps(reg0, reg1);   // f = (d * d) + (e * e)
  reg6 = _mm256_add_ps(reg2, reg3);   // i = (g * g) + (h * h)

  reg7 = _mm256_mul_ps(reg11, reg6);  // e * i
  reg14 = _mm256_mul_ps(reg5, reg13); // f * h
  reg15 = _mm256_mul_ps(reg5, reg12); // f * g
  reg0 = _mm256_mul_ps(reg10, reg6);  // d * i
  reg1 = _mm256_mul_ps(reg10, reg12); // d * g
  reg2 = _mm256_mul_ps(reg11, reg12); // e * g

  reg3 = _mm256_sub_ps(reg7, reg14); // (e * i) - (f * h)
  reg5 = _mm256_sub_ps(reg15, reg0); // (f * g) - (d * i)
  reg6 = _mm256_sub_ps(reg1, reg2);  // (d * g) - (e * g)

  reg13 = _mm256_mul_ps(reg3, reg8); // out0 = a * ((e * i) - (f * h))
  reg14 = _mm256_mul_ps(reg5, reg9); // out1 = b * ((f * g) - (d * i))
  reg15 = _mm256_mul_ps(reg6, reg4); // out2 = c * ((d * g) - (e * g))

  // TODO: figure out what registers are free
  // reg8, reg9, reg10 are required -- all others are free
  /*---------------------------------------------------------------------------
                          Starting 2nd kernel
  ---------------------------------------------------------------------------*/
  reg0 = _mm256_load_ps(Ax + SIMD_SIZE);
  reg1 = _mm256_load_ps(Ay + SIMD_SIZE);
  reg2 = _mm256_load_ps(Bx + SIMD_SIZE);
  reg13 = _mm256_add_ps(reg13, reg14); // out0 + out1         --> KERNEL 1
  reg3 = _mm256_load_ps(By + SIMD_SIZE);
  reg4 = _mm256_load_ps(Cx + SIMD_SIZE);
  reg5 = _mm256_load_ps(Cy + SIMD_SIZE);
  reg15 = _mm256_add_ps(reg13, reg15); // out0 + out1 + out2  --> KERNEL 1
  reg6 = _mm256_load_ps(Dx + SIMD_SIZE);
  reg7 = _mm256_load_ps(Dy + SIMD_SIZE);

  /*---------------------------------------------------------------------------
              Completing 1st kernel + 2nd kernel subtracts
  ---------------------------------------------------------------------------*/

  
  reg8 = _mm256_sub_ps(reg0, reg6);  // a = Ax - Dx    --> KERNEL 2
  reg9 = _mm256_sub_ps(reg1, reg7);  // b = Ay - Dy    --> KERNEL 2
  reg10 = _mm256_sub_ps(reg2, reg6); // d = Bx - Dx    --> KERNEL 2
  
  reg11 = _mm256_sub_ps(reg3, reg7); // e = By - Dy    --> KERNEL 2
  reg12 = _mm256_sub_ps(reg4, reg6); // g = Cx - Dx    --> KERNEL 2
  reg13 = _mm256_sub_ps(reg5, reg7); // h = Cy - Dy    --> KERNEL 2
  _mm256_store_ps(out, reg15);         // store out           --> KERNEL 1

  // TODO: figure out what registers are free

  /*---------------------------------------------------------------------------
                      Continuing 2nd Kernel
  ---------------------------------------------------------------------------*/

  reg14 = _mm256_mul_ps(reg8, reg8);        // a * a
  reg15 = _mm256_mul_ps(reg9, reg9);        // b * b
  reg0 = _mm256_mul_ps(reg10, reg10);       // d * d
  reg1 = _mm256_mul_ps(reg11, reg11);       // e * e
  reg2 = _mm256_mul_ps(reg12, reg12);       // g * g
  reg3 = _mm256_mul_ps(reg13, reg13);       // h * h

  reg4 = _mm256_add_ps(reg14, reg15); // c = (a * a) + (b * b)
  reg5 = _mm256_add_ps(reg0, reg1);   // f = (d * d) + (e * e)
  reg6 = _mm256_add_ps(reg2, reg3);   // i = (g * g) + (h * h)

  reg7 = _mm256_mul_ps(reg11, reg6);  // e * i
  reg14 = _mm256_mul_ps(reg5, reg13); // f * h
  reg15 = _mm256_mul_ps(reg5, reg12); // f * g
  reg0 = _mm256_mul_ps(reg10, reg6);  // d * i
  reg1 = _mm256_mul_ps(reg10, reg12); // d * g
  reg2 = _mm256_mul_ps(reg11, reg12); // e * g

  reg3 = _mm256_sub_ps(reg7, reg14); // (e * i) - (f * h)
  reg5 = _mm256_sub_ps(reg15, reg0); // (f * g) - (d * i)
  reg6 = _mm256_sub_ps(reg1, reg2);  // (d * g) - (e * g)

  reg13 = _mm256_mul_ps(reg3, reg8); // out0 = a * ((e * i) - (f * h))
  reg14 = _mm256_mul_ps(reg5, reg9); // out1 = b * ((f * g) - (d * i))
  reg15 = _mm256_mul_ps(reg6, reg4); // out2 = c * ((d * g) - (e * g))

  /*---------------------------------------------------------------------------
                          Starting 3rd kernel
  ---------------------------------------------------------------------------*/

  reg0 = _mm256_load_ps(Ax + (2 * SIMD_SIZE));
  reg1 = _mm256_load_ps(Ay + (2 * SIMD_SIZE));
  reg2 = _mm256_load_ps(Bx + (2 * SIMD_SIZE));
  reg3 = _mm256_load_ps(By + (2 * SIMD_SIZE));
  reg4 = _mm256_load_ps(Cx + (2 * SIMD_SIZE));
  reg5 = _mm256_load_ps(Cy + (2 * SIMD_SIZE));
  reg6 = _mm256_load_ps(Dx + (2 * SIMD_SIZE));
  reg7 = _mm256_load_ps(Dy + (2 * SIMD_SIZE));

  /*---------------------------------------------------------------------------
              Completing 2nd kernel + 3rd kernel subtracts
  ---------------------------------------------------------------------------*/

  reg13 = _mm256_add_ps(reg13, reg14); // out0 + out1         --> KERNEL 2
  reg8 = _mm256_sub_ps(reg0, reg6);  // a = Ax - Dx    --> KERNEL 3
  reg9 = _mm256_sub_ps(reg1, reg7);  // b = Ay - Dy    --> KERNEL 3
  reg10 = _mm256_sub_ps(reg2, reg6); // d = Bx - Dx    --> KERNEL 3
  reg15 = _mm256_add_ps(reg13, reg15); // out0 + out1 + out2  --> KERNEL 2
  reg11 = _mm256_sub_ps(reg3, reg7); // e = By - Dy    --> KERNEL 3
  reg12 = _mm256_sub_ps(reg4, reg6); // g = Cx - Dx    --> KERNEL 3
  reg13 = _mm256_sub_ps(reg5, reg7); // h = Cy - Dy    --> KERNEL 3
  _mm256_store_ps(out + SIMD_SIZE, reg15);  // store out      --> KERNEL 2

  // TODO: figure out what registers are free

  /*---------------------------------------------------------------------------
                      Continuing 3rd Kernel
  ---------------------------------------------------------------------------*/

  reg14 = _mm256_mul_ps(reg8, reg8);        // a * a
  reg15 = _mm256_mul_ps(reg9, reg9);        // b * b
  reg0 = _mm256_mul_ps(reg10, reg10);       // d * d
  reg1 = _mm256_mul_ps(reg11, reg11);       // e * e
  reg2 = _mm256_mul_ps(reg12, reg12);       // g * g
  reg3 = _mm256_mul_ps(reg13, reg13);       // h * h

  reg4 = _mm256_add_ps(reg14, reg15); // c = (a * a) + (b * b)
  reg5 = _mm256_add_ps(reg0, reg1);   // f = (d * d) + (e * e)
  reg6 = _mm256_add_ps(reg2, reg3);   // i = (g * g) + (h * h)

  reg7 = _mm256_mul_ps(reg11, reg6);  // e * i
  reg14 = _mm256_mul_ps(reg5, reg13); // f * h
  reg15 = _mm256_mul_ps(reg5, reg12); // f * g
  reg0 = _mm256_mul_ps(reg10, reg6);  // d * i
  reg1 = _mm256_mul_ps(reg10, reg12); // d * g
  reg2 = _mm256_mul_ps(reg11, reg12); // e * g

  reg3 = _mm256_sub_ps(reg7, reg14); // (e * i) - (f * h)
  reg5 = _mm256_sub_ps(reg15, reg0); // (f * g) - (d * i)
  reg6 = _mm256_sub_ps(reg1, reg2);  // (d * g) - (e * g)

  reg13 = _mm256_mul_ps(reg3, reg8); // out0 = a * ((e * i) - (f * h))
  reg14 = _mm256_mul_ps(reg5, reg9); // out1 = b * ((f * g) - (d * i))
  reg15 = _mm256_mul_ps(reg6, reg4); // out2 = c * ((d * g) - (e * g))

  /*---------------------------------------------------------------------------
                          Starting 4th kernel
  ---------------------------------------------------------------------------*/

  reg0 = _mm256_load_ps(Ax + (3 * SIMD_SIZE));
  reg1 = _mm256_load_ps(Ay + (3 * SIMD_SIZE));
  reg2 = _mm256_load_ps(Bx + (3 * SIMD_SIZE));
  reg3 = _mm256_load_ps(By + (3 * SIMD_SIZE));
  reg4 = _mm256_load_ps(Cx + (3 * SIMD_SIZE));
  reg5 = _mm256_load_ps(Cy + (3 * SIMD_SIZE));
  reg6 = _mm256_load_ps(Dx + (3 * SIMD_SIZE));
  reg7 = _mm256_load_ps(Dy + (3 * SIMD_SIZE));

  /*---------------------------------------------------------------------------
              Completing 3rd kernel + 4th kernel subtracts
  ---------------------------------------------------------------------------*/

  reg13 = _mm256_add_ps(reg13, reg14); // out0 + out1         --> KERNEL 3
  reg8 = _mm256_sub_ps(reg0, reg6);  // a = Ax - Dx    --> KERNEL 4
  reg9 = _mm256_sub_ps(reg1, reg7);  // b = Ay - Dy    --> KERNEL 4
  reg10 = _mm256_sub_ps(reg2, reg6); // d = Bx - Dx    --> KERNEL 4
  reg15 = _mm256_add_ps(reg13, reg15); // out0 + out1 + out2  --> KERNEL 3
  reg11 = _mm256_sub_ps(reg3, reg7); // e = By - Dy    --> KERNEL 4
  reg12 = _mm256_sub_ps(reg4, reg6); // g = Cx - Dx    --> KERNEL 4
  reg13 = _mm256_sub_ps(reg5, reg7); // h = Cy - Dy    --> KERNEL 4
  _mm256_store_ps(out + (2 * SIMD_SIZE), reg15); // store out --> KERNEL 3

  // TODO: figure out what registers are free

  /*---------------------------------------------------------------------------
                      Continuing 4th Kernel
  ---------------------------------------------------------------------------*/

  reg14 = _mm256_mul_ps(reg8, reg8);        // a * a
  reg15 = _mm256_mul_ps(reg9, reg9);        // b * b
  reg0 = _mm256_mul_ps(reg10, reg10);       // d * d
  reg1 = _mm256_mul_ps(reg11, reg11);       // e * e
  reg2 = _mm256_mul_ps(reg12, reg12);       // g * g
  reg3 = _mm256_mul_ps(reg13, reg13);       // h * h

  reg4 = _mm256_add_ps(reg14, reg15); // c = (a * a) + (b * b)
  reg5 = _mm256_add_ps(reg0, reg1);   // f = (d * d) + (e * e)
  reg6 = _mm256_add_ps(reg2, reg3);   // i = (g * g) + (h * h)

  reg7 = _mm256_mul_ps(reg11, reg6);  // e * i
  reg14 = _mm256_mul_ps(reg5, reg13); // f * h
  reg15 = _mm256_mul_ps(reg5, reg12); // f * g
  reg0 = _mm256_mul_ps(reg10, reg6);  // d * i
  reg1 = _mm256_mul_ps(reg10, reg12); // d * g
  reg2 = _mm256_mul_ps(reg11, reg12); // e * g

  reg3 = _mm256_sub_ps(reg7, reg14); // (e * i) - (f * h)
  reg5 = _mm256_sub_ps(reg15, reg0); // (f * g) - (d * i)
  reg6 = _mm256_sub_ps(reg1, reg2);  // (d * g) - (e * g)

  reg13 = _mm256_mul_ps(reg3, reg8); // out0 = a * ((e * i) - (f * h))
  reg14 = _mm256_mul_ps(reg5, reg9); // out1 = b * ((f * g) - (d * i))
  reg15 = _mm256_mul_ps(reg6, reg4); // out2 = c * ((d * g) - (e * g))
  // ______________________________________________________________________

  // Not enough instructions to hide the latency now
  // Run the top half a couple times until we have enough operands to do so

  reg13 = _mm256_add_ps(reg13, reg14); // out0 + out1
  reg15 = _mm256_add_ps(reg13, reg15); // out0 + out1 + out2
  _mm256_store_ps(out + (3 * SIMD_SIZE), reg15);       // store out
}

#endif
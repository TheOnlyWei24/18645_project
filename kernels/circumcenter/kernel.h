#ifndef __CIRCUMCENTER_KERNEL_H_
#define __CIRCUMCENTER_KERNEL_H_

#include <immintrin.h>

#define SIMD_SIZE 8

#define NUM_SIMD_IN_KERNEL 6

struct in_data {
  float Ax[SIMD_SIZE];
  float Ay[SIMD_SIZE];
  float Bx[SIMD_SIZE];
  float By[SIMD_SIZE];
  float Cx[SIMD_SIZE];
  float Cy[SIMD_SIZE];
};

typedef struct in_data in_data_t;

struct kernel_in_data {
  in_data_t data[NUM_SIMD_IN_KERNEL];
};

typedef struct kernel_in_data kernel_in_data_t;

struct out_data {
  float Ux[SIMD_SIZE];
  float Uy[SIMD_SIZE];
};

typedef struct out_data out_data_t;

struct kernel_out_data {
  out_data_t data[NUM_SIMD_IN_KERNEL];
};

typedef struct kernel_out_data kernel_out_data_t;

struct buffer {
  float partUx[SIMD_SIZE];
  float partUy[SIMD_SIZE];
  float partD[SIMD_SIZE];
};

typedef struct buffer buffer_t;

struct kernel_buffer {
  buffer_t buffer[NUM_SIMD_IN_KERNEL];
};

typedef struct kernel_buffer kernel_buffer_t;

void baseline(kernel_in_data_t *restrict in_data, kernel_out_data_t *restrict out_data) {
  float Ax, Ay, Bx, By, Cx, Cy;
  float Ax2_Ay2, Bx2_Bx2, Cx2_Cx2, D;

  for (int i = 0; i < NUM_SIMD_IN_KERNEL; i++) {
    for (int j = 0; j < SIMD_SIZE; j++) {
      Ax = in_data->data[i].Ax[j];
      Ay = in_data->data[i].Ay[j];
      Bx = in_data->data[i].Bx[j];
      By = in_data->data[i].By[j];
      Cx = in_data->data[i].Cx[j];
      Cy = in_data->data[i].Cy[j];
      
      Ax2_Ay2 = (Ax * Ax) + (Ay * Ay);
      Bx2_Bx2 = (Bx * Bx) + (By * By);
      Cx2_Cx2 = (Cx * Cx) + (Cy * Cy);

      D = 2 * (((By - Cy) * Ax) + ((Cy - Ay) * Bx) +
               ((Ay - By) * Cx));

      out_data->data[i].Ux[j] = ((Ax2_Ay2 * (By - Cy)) + (Bx2_Bx2 * (Cy - Ay)) +
                             (Cx2_Cx2 * (Ay - By))) / D;

      out_data->data[i].Uy[j] = ((Ax2_Ay2 * (Cx - Bx)) + (Bx2_Bx2 * (Ax - Cx)) +
                             (Cx2_Cx2 * (Bx - Ax))) / D;
    }
  }
}


static inline void kernel0(kernel_in_data_t *restrict in_data, kernel_buffer_t *restrict buffer) {
  // First half of first kernel
  __m256 reg0 = _mm256_load_ps(in_data->data[0].Ax);
  __m256 reg1 = _mm256_load_ps(in_data->data[0].Ay);
  __m256 reg2 = _mm256_load_ps(in_data->data[0].Bx);
  __m256 reg3 = _mm256_load_ps(in_data->data[0].By);
  __m256 reg4 = _mm256_load_ps(in_data->data[0].Cx);
  __m256 reg5 = _mm256_load_ps(in_data->data[0].Cy);

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
  reg6 = _mm256_load_ps(in_data->data[1].Ax);
  reg7 = _mm256_load_ps(in_data->data[1].Ay);
  reg8 = _mm256_load_ps(in_data->data[1].Bx);
  __m256 reg12 = _mm256_load_ps(in_data->data[1].By);
  __m256 reg13 = _mm256_load_ps(in_data->data[1].Cx);
  __m256 reg14 = _mm256_load_ps(in_data->data[1].Cy);

  // Finish second half of first kernel
  reg0 = _mm256_add_ps(reg0, reg2);  // D0 + D1
  reg2 = _mm256_add_ps(reg9, reg10); // Ux0 + Ux1
  reg1 = _mm256_add_ps(reg1, reg3);  // Uy0 + Uy1

  reg0 = _mm256_add_ps(reg0, reg4);  // D0 + D1 + D2
  reg2 = _mm256_add_ps(reg2, reg11); // Ux0 + Ux1 + Ux2
  reg1 = _mm256_add_ps(reg1, reg5);  // Uy0 + Uy1 + Uy2

  _mm256_store_ps(buffer->buffer[0].partD, reg0);
  _mm256_store_ps(buffer->buffer[0].partUx, reg2);
  _mm256_store_ps(buffer->buffer[0].partUy, reg1);

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
  reg0 = _mm256_load_ps(in_data->data[2].Ax);
  reg1 = _mm256_load_ps(in_data->data[2].Ay);
  reg2 = _mm256_load_ps(in_data->data[2].Bx);
  reg3 = _mm256_load_ps(in_data->data[2].By);
  reg4 = _mm256_load_ps(in_data->data[2].Cx);
  reg5 = _mm256_load_ps(in_data->data[2].Cy);

  reg7 = _mm256_add_ps(reg7, reg12);
  reg6 = _mm256_add_ps(reg6, reg8);
  reg9 = _mm256_add_ps(reg9, reg10);

  reg7 = _mm256_add_ps(reg7, reg14);
  reg6 = _mm256_add_ps(reg6, reg13);
  reg9 = _mm256_add_ps(reg9, reg11);

  _mm256_store_ps(buffer->buffer[1].partD, reg7);
  _mm256_store_ps(buffer->buffer[1].partUx, reg6);
  _mm256_store_ps(buffer->buffer[1].partUy, reg9);

  /*** *** *** *** *** *** *** *** ***/

  reg6 = _mm256_mul_ps(reg0, reg0);
  reg7 = _mm256_mul_ps(reg1, reg1);
  reg8 = _mm256_mul_ps(reg2, reg2);
  reg9 = _mm256_mul_ps(reg3, reg3);
  reg10 = _mm256_mul_ps(reg4, reg4);
  reg11 = _mm256_mul_ps(reg5, reg5);

  reg6 = _mm256_add_ps(reg6, reg7);
  reg7 = _mm256_add_ps(reg8, reg9);
  reg8 = _mm256_add_ps(reg10, reg11);

  reg9 = _mm256_sub_ps(reg3, reg5);
  reg10 = _mm256_sub_ps(reg5, reg1);
  reg11 = _mm256_sub_ps(reg1, reg3);

  reg1 = _mm256_sub_ps(reg4, reg2);
  reg3 = _mm256_sub_ps(reg0, reg4);
  reg5 = _mm256_sub_ps(reg2, reg0);

  reg0 = _mm256_mul_ps(reg9, reg0);
  reg2 = _mm256_mul_ps(reg10, reg2);
  reg4 = _mm256_mul_ps(reg11, reg4);

  reg9 = _mm256_mul_ps(reg6, reg9);
  reg10 = _mm256_mul_ps(reg7, reg10);
  reg11 = _mm256_mul_ps(reg8, reg11);

  reg1 = _mm256_mul_ps(reg6, reg1);
  reg3 = _mm256_mul_ps(reg7, reg3);
  reg5 = _mm256_mul_ps(reg8, reg5);

  reg6 = _mm256_load_ps(in_data->data[3].Ax);
  reg7 = _mm256_load_ps(in_data->data[3].Ay);
  reg8 = _mm256_load_ps(in_data->data[3].Bx);
  reg12 = _mm256_load_ps(in_data->data[3].By);
  reg13 = _mm256_load_ps(in_data->data[3].Cx);
  reg14 = _mm256_load_ps(in_data->data[3].Cy);

  reg0 = _mm256_add_ps(reg0, reg2);
  reg2 = _mm256_add_ps(reg9, reg10);
  reg1 = _mm256_add_ps(reg1, reg3);

  reg0 = _mm256_add_ps(reg0, reg4);
  reg2 = _mm256_add_ps(reg2, reg11);
  reg1 = _mm256_add_ps(reg1, reg5);

  _mm256_store_ps(buffer->buffer[2].partD, reg0);
  _mm256_store_ps(buffer->buffer[2].partUx, reg2);
  _mm256_store_ps(buffer->buffer[2].partUy, reg1);

  reg0 = _mm256_mul_ps(reg6, reg6);
  reg1 = _mm256_mul_ps(reg7, reg7);
  reg2 = _mm256_mul_ps(reg8, reg8);
  reg3 = _mm256_mul_ps(reg12, reg12);
  reg4 = _mm256_mul_ps(reg13, reg13);
  reg5 = _mm256_mul_ps(reg14, reg14);

  reg9 = _mm256_add_ps(reg0, reg1);
  reg10 = _mm256_add_ps(reg2, reg3);
  reg11 = _mm256_add_ps(reg4, reg5);

  reg0 = _mm256_sub_ps(reg12, reg14);
  reg1 = _mm256_sub_ps(reg14, reg7);
  reg2 = _mm256_sub_ps(reg7, reg12);

  reg3 = _mm256_sub_ps(reg13, reg8);
  reg4 = _mm256_sub_ps(reg6, reg13);
  reg5 = _mm256_sub_ps(reg8, reg6);

  reg7 = _mm256_mul_ps(reg0, reg6);
  reg12 = _mm256_mul_ps(reg1, reg8);
  reg14 = _mm256_mul_ps(reg2, reg13);

  reg6 = _mm256_mul_ps(reg9, reg0);
  reg8 = _mm256_mul_ps(reg10, reg1);
  reg13 = _mm256_mul_ps(reg11, reg2);

  reg9 = _mm256_mul_ps(reg3, reg9);
  reg10 = _mm256_mul_ps(reg4, reg10);
  reg11 = _mm256_mul_ps(reg5, reg11);

  reg0 = _mm256_load_ps(in_data->data[4].Ax);
  reg1 = _mm256_load_ps(in_data->data[4].Ay);
  reg2 = _mm256_load_ps(in_data->data[4].Bx);
  reg3 = _mm256_load_ps(in_data->data[4].By);
  reg4 = _mm256_load_ps(in_data->data[4].Cx);
  reg5 = _mm256_load_ps(in_data->data[4].Cy);

  reg7 = _mm256_add_ps(reg7, reg12);
  reg6 = _mm256_add_ps(reg6, reg8);
  reg9 = _mm256_add_ps(reg9, reg10);

  reg7 = _mm256_add_ps(reg7, reg14);
  reg6 = _mm256_add_ps(reg6, reg13);
  reg9 = _mm256_add_ps(reg9, reg11);

  _mm256_store_ps(buffer->buffer[3].partD, reg7);
  _mm256_store_ps(buffer->buffer[3].partUx, reg6);
  _mm256_store_ps(buffer->buffer[3].partUy, reg9);

  /*** *** *** *** *** *** *** *** ***/

  reg6 = _mm256_mul_ps(reg0, reg0);
  reg7 = _mm256_mul_ps(reg1, reg1);
  reg8 = _mm256_mul_ps(reg2, reg2);
  reg9 = _mm256_mul_ps(reg3, reg3);
  reg10 = _mm256_mul_ps(reg4, reg4);
  reg11 = _mm256_mul_ps(reg5, reg5);

  reg6 = _mm256_add_ps(reg6, reg7);
  reg7 = _mm256_add_ps(reg8, reg9);
  reg8 = _mm256_add_ps(reg10, reg11);

  reg9 = _mm256_sub_ps(reg3, reg5);
  reg10 = _mm256_sub_ps(reg5, reg1);
  reg11 = _mm256_sub_ps(reg1, reg3);

  reg1 = _mm256_sub_ps(reg4, reg2);
  reg3 = _mm256_sub_ps(reg0, reg4);
  reg5 = _mm256_sub_ps(reg2, reg0);

  reg0 = _mm256_mul_ps(reg9, reg0);
  reg2 = _mm256_mul_ps(reg10, reg2);
  reg4 = _mm256_mul_ps(reg11, reg4);

  reg9 = _mm256_mul_ps(reg6, reg9);
  reg10 = _mm256_mul_ps(reg7, reg10);
  reg11 = _mm256_mul_ps(reg8, reg11);

  reg1 = _mm256_mul_ps(reg6, reg1);
  reg3 = _mm256_mul_ps(reg7, reg3);
  reg5 = _mm256_mul_ps(reg8, reg5);

  reg6 = _mm256_load_ps(in_data->data[5].Ax);
  reg7 = _mm256_load_ps(in_data->data[5].Ay);
  reg8 = _mm256_load_ps(in_data->data[5].Bx);
  reg12 = _mm256_load_ps(in_data->data[5].By);
  reg13 = _mm256_load_ps(in_data->data[5].Cx);
  reg14 = _mm256_load_ps(in_data->data[5].Cy);

  reg0 = _mm256_add_ps(reg0, reg2);
  reg2 = _mm256_add_ps(reg9, reg10);
  reg1 = _mm256_add_ps(reg1, reg3);

  reg0 = _mm256_add_ps(reg0, reg4);
  reg2 = _mm256_add_ps(reg2, reg11);
  reg1 = _mm256_add_ps(reg1, reg5);

  _mm256_store_ps(buffer->buffer[4].partD, reg0);
  _mm256_store_ps(buffer->buffer[4].partUx, reg2);
  _mm256_store_ps(buffer->buffer[4].partUy, reg1);

  reg0 = _mm256_mul_ps(reg6, reg6);
  reg1 = _mm256_mul_ps(reg7, reg7);
  reg2 = _mm256_mul_ps(reg8, reg8);
  reg3 = _mm256_mul_ps(reg12, reg12);
  reg4 = _mm256_mul_ps(reg13, reg13);
  reg5 = _mm256_mul_ps(reg14, reg14);

  reg9 = _mm256_add_ps(reg0, reg1);
  reg10 = _mm256_add_ps(reg2, reg3);
  reg11 = _mm256_add_ps(reg4, reg5);

  reg0 = _mm256_sub_ps(reg12, reg14);
  reg1 = _mm256_sub_ps(reg14, reg7);
  reg2 = _mm256_sub_ps(reg7, reg12);

  reg3 = _mm256_sub_ps(reg13, reg8);
  reg4 = _mm256_sub_ps(reg6, reg13);
  reg5 = _mm256_sub_ps(reg8, reg6);

  reg7 = _mm256_mul_ps(reg0, reg6);
  reg12 = _mm256_mul_ps(reg1, reg8);
  reg14 = _mm256_mul_ps(reg2, reg13);

  reg6 = _mm256_mul_ps(reg9, reg0);
  reg8 = _mm256_mul_ps(reg10, reg1);
  reg13 = _mm256_mul_ps(reg11, reg2);

  reg9 = _mm256_mul_ps(reg3, reg9);
  reg10 = _mm256_mul_ps(reg4, reg10);
  reg11 = _mm256_mul_ps(reg5, reg11);

  // reg0 = _mm256_load_ps(Ax + (6 * SIMD_SIZE));
  // reg1 = _mm256_load_ps(Ay + (6 * SIMD_SIZE));
  // reg2 = _mm256_load_ps(Bx + (6 * SIMD_SIZE));
  // reg3 = _mm256_load_ps(By + (6 * SIMD_SIZE));
  // reg4 = _mm256_load_ps(Cx + (6 * SIMD_SIZE));
  // reg5 = _mm256_load_ps(Cy + (6 * SIMD_SIZE));

  reg7 = _mm256_add_ps(reg7, reg12);
  reg6 = _mm256_add_ps(reg6, reg8);
  reg9 = _mm256_add_ps(reg9, reg10);

  reg7 = _mm256_add_ps(reg7, reg14);
  reg6 = _mm256_add_ps(reg6, reg13);
  reg9 = _mm256_add_ps(reg9, reg11);

  _mm256_store_ps(buffer->buffer[5].partD, reg7);
  _mm256_store_ps(buffer->buffer[5].partUx, reg6);
  _mm256_store_ps(buffer->buffer[5].partUy, reg9);
}

static inline void kernel1(kernel_out_data_t *restrict out_data, kernel_buffer_t *restrict buffer) {
  float two[] = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
  __m256 reg0 = _mm256_load_ps(&two[0]);
  __m256 reg1 = _mm256_load_ps(buffer->buffer[0].partD); // D = partD / 2
  __m256 reg2 = _mm256_load_ps(buffer->buffer[1].partD);
  __m256 reg3 = _mm256_load_ps(buffer->buffer[2].partD);
  __m256 reg4 = _mm256_load_ps(buffer->buffer[3].partD);
  __m256 reg5 = _mm256_load_ps(buffer->buffer[4].partD);
  __m256 reg6 = _mm256_load_ps(buffer->buffer[5].partD);

  reg1 = _mm256_mul_ps(reg1, reg0);
  reg2 = _mm256_mul_ps(reg2, reg0);
  reg3 = _mm256_mul_ps(reg3, reg0);
  reg4 = _mm256_mul_ps(reg4, reg0);
  reg5 = _mm256_mul_ps(reg5, reg0);
  reg6 = _mm256_mul_ps(reg6, reg0);

  __m256 reg7 = _mm256_load_ps(buffer->buffer[0].partUx);
  __m256 reg8 = _mm256_load_ps(buffer->buffer[0].partUy);
  reg7 = _mm256_div_ps(reg7, reg1); // Ux = partUx / D
  reg8 = _mm256_div_ps(reg8, reg1); // Uy = partUy / D

  __m256 reg9 = _mm256_load_ps(buffer->buffer[1].partUx);
  __m256 reg10 = _mm256_load_ps(buffer->buffer[1].partUy);
  reg9 = _mm256_div_ps(reg9, reg2);
  reg10 = _mm256_div_ps(reg10, reg2);

  __m256 reg11 = _mm256_load_ps(buffer->buffer[2].partUx);
  __m256 reg12 = _mm256_load_ps(buffer->buffer[2].partUy);
  reg11 = _mm256_div_ps(reg11, reg3);
  reg12 = _mm256_div_ps(reg12, reg3);

  __m256 reg13 = _mm256_load_ps(buffer->buffer[3].partUx);
  __m256 reg14 = _mm256_load_ps(buffer->buffer[3].partUy);
  reg13 = _mm256_div_ps(reg13, reg4);
  reg14 = _mm256_div_ps(reg14, reg4);

  __m256 reg15 = _mm256_load_ps(buffer->buffer[4].partUx);
  reg0 = _mm256_load_ps(buffer->buffer[4].partUy);
  reg15 = _mm256_div_ps(reg15, reg5);
  reg0 = _mm256_div_ps(reg0, reg5);

  reg1 = _mm256_load_ps(buffer->buffer[5].partUx);
  reg2 = _mm256_load_ps(buffer->buffer[5].partUy);
  reg1 = _mm256_div_ps(reg1, reg6);
  reg2 = _mm256_div_ps(reg2, reg6);

  _mm256_store_ps(out_data->data[0].Ux, reg7);
  _mm256_store_ps(out_data->data[0].Uy, reg8);
  _mm256_store_ps(out_data->data[1].Ux, reg9);
  _mm256_store_ps(out_data->data[1].Uy, reg10);
  _mm256_store_ps(out_data->data[2].Ux, reg11);
  _mm256_store_ps(out_data->data[2].Uy, reg12);
  _mm256_store_ps(out_data->data[3].Ux, reg13);
  _mm256_store_ps(out_data->data[3].Uy, reg14);
  _mm256_store_ps(out_data->data[4].Ux, reg15);
  _mm256_store_ps(out_data->data[4].Uy, reg0);
  _mm256_store_ps(out_data->data[5].Ux, reg1);
  _mm256_store_ps(out_data->data[5].Uy, reg2);
}

#endif

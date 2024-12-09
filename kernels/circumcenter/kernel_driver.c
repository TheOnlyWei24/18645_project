#include "kernel.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BASE_FREQ 2.4

#define MAX_FREQ 3.4

#define ALIGNMENT 64

#define RUNS 100000

#define KERNEL_ITERS 4

// kernel0 + kernel1
// SIMD_SIZE * NUM_OPS * NUM ITER
#define OPS ((SIMD_SIZE * 30 * 6) + (SIMD_SIZE * 18 * 6))

static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

int main(void) {
  srand(time(NULL));

  // Set up data structures
  kernel_data_t *data;
  kernel_buffer_t *buffer;

  posix_memalign((void **)&data, ALIGNMENT,
                 KERNEL_ITERS * sizeof(kernel_data_t));
  posix_memalign((void **)&buffer, ALIGNMENT, sizeof(kernel_buffer_t));

  // Initialize data
  for (int i = 0; i < KERNEL_ITERS; i++) {
    for (int j = 0; j < NUM_SIMD_IN_KERNEL; j++) {
      for (int k = 0; k < SIMD_SIZE; k++) {
        // in_data[i].data[j].Ax[k] = 0.0;
        // in_data[i].data[j].Ay[k] = 0.0;
        // in_data[i].data[j].Bx[k] = 1.0;
        // in_data[i].data[j].By[k] = 0.0;
        // in_data[i].data[j].Cx[k] = 0.5;
        // in_data[i].data[j].Cy[k] = 1.0;
        data[i].data[j].Ax[k] = rand();
        data[i].data[j].Ay[k] = rand();
        data[i].data[j].Bx[k] = rand();
        data[i].data[j].By[k] = rand();
        data[i].data[j].Cx[k] = rand();
        data[i].data[j].Cy[k] = rand();
        data[i].data[j].Ux[k] = 0.0;
        data[i].data[j].Uy[k] = 0.0;
      }
    }
  }

  // Initialize buffer
  for (int i = 0; i < NUM_SIMD_IN_KERNEL; i++) {
    for (int j = 0; j < SIMD_SIZE; j++) {
      buffer->buffer[i].partUx[j] = 0.0;
      buffer->buffer[i].partUy[j] = 0.0;
      buffer->buffer[i].partD[j] = 0.0;
    }
  }

  unsigned long long sum, t0, t1;

  sum = 0;

  // Test kernels
  for (int i = 0; i < RUNS; i++) {
    for (int j = 0; j < KERNEL_ITERS; j++) {
      t0 = rdtsc();
      kernel0(&(data[j]), buffer);
      kernel1(&(data[j]), buffer);
      // baseline(&(data[j]));
      t1 = rdtsc();
    }
    sum += (t1 - t0);
  }

  printf("%d\n", OPS * KERNEL_ITERS);
  printf(" %lf\n", (OPS * KERNEL_ITERS) /
                       ((double)(sum / RUNS) * (MAX_FREQ / BASE_FREQ)));

  printf("First kernel: %f %f\n", data[0].data[0].Ux[0], data[0].data[0].Uy[0]);
  // printf("Second kernel: %f %f\n", data[1].data[0].Ux[0],
  //        data[1].data[0].Uy[0]);

  // Clean up
  free(data);
  free(buffer);

  return 0;
}
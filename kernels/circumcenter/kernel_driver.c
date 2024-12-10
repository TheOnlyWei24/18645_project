#include "kernel.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BASE_FREQ 2.4

#define MAX_FREQ 3.4

#define ALIGNMENT 64

#define RUNS 100000

#define KERNEL_ITERS_PER_THREAD 18

#define NUM_THREADS 2

#define KERNEL_ITERS (KERNEL_ITERS_PER_THREAD * NUM_THREADS)

// kernel0 + kernel1
// SIMD_SIZE * NUM_OPS * NUM_SIMD_IN_KERNEL
#define OPS ((SIMD_SIZE * 30 * 6) + (SIMD_SIZE * 4 * 6))

#define CACHELINE 64

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
  posix_memalign((void **)&buffer, ALIGNMENT,
                 NUM_THREADS * sizeof(kernel_buffer_t));

  // Initialize data
  for (int i = 0; i < KERNEL_ITERS; i++) {
    for (int j = 0; j < NUM_SIMD_IN_KERNEL; j++) {
      for (int k = 0; k < SIMD_SIZE; k++) {
        data[i].data[j].Ax[k] = 0.0;
        data[i].data[j].Ay[k] = 0.0;
        data[i].data[j].Bx[k] = 1.0;
        data[i].data[j].By[k] = 0.0;
        data[i].data[j].Cx[k] = 0.5;
        data[i].data[j].Cy[k] = 1.0;
        // data[i].data[j].Ax[k] = rand();
        // data[i].data[j].Ay[k] = rand();
        // data[i].data[j].Bx[k] = rand();
        // data[i].data[j].By[k] = rand();
        // data[i].data[j].Cx[k] = rand();
        // data[i].data[j].Cy[k] = rand();
        data[i].data[j].Ux[k] = 0.0;
        data[i].data[j].Uy[k] = 0.0;
      }
    }
  }

  // Initialize buffer
  for (int i = 0; i < NUM_THREADS; i++) {
    for (int j = 0; j < NUM_SIMD_IN_KERNEL; j++) {
      for (int k = 0; k < SIMD_SIZE; k++) {
        buffer[i].buffer[j].partUx[k] = 0.0;
        buffer[i].buffer[j].partUy[k] = 0.0;
        buffer[i].buffer[j].partD[k] = 0.0;
      }
    }
  }

  unsigned long long sum[NUM_THREADS * CACHELINE], t0[NUM_THREADS * CACHELINE],
      t1[NUM_THREADS * CACHELINE];
  for (int i = 0; i < NUM_THREADS; i++) {
    sum[i * CACHELINE] = 0;
    t0[i * CACHELINE] = 0;
    t1[i * CACHELINE] = 0;
  }

  // Test kernels
  for (int i = 0; i < RUNS; i++) {
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int j = 0; j < NUM_THREADS; j++) {
      for (int k = 0; k < KERNEL_ITERS_PER_THREAD; k++) {
        t0[j * CACHELINE] = rdtsc();
        kernel0(&(data[j * KERNEL_ITERS_PER_THREAD + k]), &buffer[j]);
        kernel1(&(data[j * KERNEL_ITERS_PER_THREAD + k]), &buffer[j]);
        // baseline(&(data[j * KERNEL_ITERS_PER_THREAD + k]));
        t1[j * CACHELINE] = rdtsc();
        sum[j * CACHELINE] += (t1[j * CACHELINE] - t0[j * CACHELINE]);
      }
    }
  }

  // printf("%d\n", OPS * KERNEL_ITERS);
  printf(" %lf\n", (OPS * KERNEL_ITERS_PER_THREAD) /
                       ((double)(sum[0] / RUNS) * (MAX_FREQ / BASE_FREQ)));

  printf("First kernel: %f %f\n", data[0].data[0].Ux[0], data[0].data[0].Uy[0]);
  // printf("Second kernel: %f %f\n", data[1].data[0].Ux[0],
  //        data[1].data[0].Uy[0]);

  // Clean up
  free(data);
  free(buffer);

  return 0;
}
#include "kernel.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BASE_FREQ 2.4

#define MAX_FREQ 3.4

#define ALIGNMENT 64

#define RUNS 10000

#define KERNEL_ITERS 8192

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
  // Set up data structures
  kernel_data_t *data;
  kernel_buffer_t *buffer;
  int thread_counts[6] = {1, 2, 4, 8, 16, 32};

  for (int t = 0; t < 6; t++) {
    int num_threads = thread_counts[t];
    posix_memalign((void **)&data, ALIGNMENT,
                   KERNEL_ITERS * sizeof(kernel_data_t));
    posix_memalign((void **)&buffer, ALIGNMENT,
                   num_threads * sizeof(kernel_buffer_t));

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
          data[i].data[j].Ux[k] = 0.0;
          data[i].data[j].Uy[k] = 0.0;
        }
      }
    }

    // Initialize buffer
    for (int i = 0; i < num_threads; i++) {
      for (int j = 0; j < NUM_SIMD_IN_KERNEL; j++) {
        for (int k = 0; k < SIMD_SIZE; k++) {
          buffer[i].buffer[j].partUx[k] = 0.0;
          buffer[i].buffer[j].partUy[k] = 0.0;
          buffer[i].buffer[j].partD[k] = 0.0;
        }
      }
    }
    unsigned long long sum = 0;
    unsigned long long t0, t1;

    // Test kernels
    for (int i = 0; i < RUNS; i++) {
      int buff_idx = 0;
      if (thread_counts[t] == 1) {

        for (int j = 0; j < KERNEL_ITERS; j++) {
          t0 = rdtsc();
          vornoi_kernel0(&(data[j]), &buffer[0]);
          vornoi_kernel1(&(data[j]), &buffer[0]);
          t1 = rdtsc();
          sum += (t1 - t0);
        }
      } else {
        t0 = rdtsc();
#pragma omp parallel for private(buff_idx) num_threads(num_threads)            \
    reduction(+ : sum)
        for (int j = 0; j < KERNEL_ITERS; j++) {
          buff_idx = j / (KERNEL_ITERS / num_threads);
          vornoi_kernel0(&(data[j]), &buffer[buff_idx]);
          vornoi_kernel1(&(data[j]), &buffer[buff_idx]);
        }
        t1 = rdtsc();
        sum += (t1 - t0);
      }
    }

    printf("Thread_count: %d, Wall time: %lld, Throughput: %lf, "
           "Iter_per_thread: %lf\n",
           thread_counts[t], sum,
           (OPS * KERNEL_ITERS) /
               ((double)(sum / RUNS) * (MAX_FREQ / BASE_FREQ)),
           (double)KERNEL_ITERS / (double)num_threads);
  }

  // Clean up
  free(data);
  free(buffer);

  return 0;
}
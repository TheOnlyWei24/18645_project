#include "baseline.h"
#include "kernel.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BASE_FREQ 2.4

#define MAX_FREQ 3.4

#define ALIGNMENT 64

#define NUM_ELEMS 1

#define RUNS 1000000

// kernel0 + kernel1
// SIMD_SIZE * NUM_OPS * NUM ITER
// #define OPS ((SIMD_SIZE * 30 * 6) + (SIMD_SIZE * 18 * 6))

static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

int main(void) {
  // Set up data structures
  float *Ax;
  float *Ay;
  float *Bx;
  float *By;
  float *Cx;
  float *Cy;
  float *Dx;
  float *Dy;
  float *out;

  posix_memalign((void **)&Ax, ALIGNMENT,
                 NUM_ELEMS * SIMD_SIZE * sizeof(float));
  posix_memalign((void **)&Ay, ALIGNMENT,
                 NUM_ELEMS * SIMD_SIZE * sizeof(float));
  posix_memalign((void **)&Bx, ALIGNMENT,
                 NUM_ELEMS * SIMD_SIZE * sizeof(float));
  posix_memalign((void **)&By, ALIGNMENT,
                 NUM_ELEMS * SIMD_SIZE * sizeof(float));
  posix_memalign((void **)&Cx, ALIGNMENT,
                 NUM_ELEMS * SIMD_SIZE * sizeof(float));
  posix_memalign((void **)&Cy, ALIGNMENT,
                 NUM_ELEMS * SIMD_SIZE * sizeof(float));
  posix_memalign((void **)&Dx, ALIGNMENT,
                 NUM_ELEMS * SIMD_SIZE * sizeof(float));
  posix_memalign((void **)&Dy, ALIGNMENT,
                 NUM_ELEMS * SIMD_SIZE * sizeof(float));
  posix_memalign((void **)&out, ALIGNMENT,
                 NUM_ELEMS * SIMD_SIZE * sizeof(float));

  // Initialize data
  for (int i = 0; i < NUM_ELEMS * SIMD_SIZE; i++) {
    Ax[i] = 0.0;
    Ay[i] = 0.0;
    Bx[i] = 1.0;
    By[i] = 0.0;
    Cx[i] = 0.5;
    Cy[i] = 1.0;
    Dx[i] = 0.5;
    Dy[i] = 0.5;
    out[i] = 0.0;
  }

  unsigned long long sum, t0, t1;

  sum = 0;

  // Test kernels
  for (int i = 0; i < RUNS; i++) {
    t0 = rdtsc();
    baseline(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, out);
    // kernel(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, out);
    t1 = rdtsc();
    sum += (t1 - t0);
  }

  printf("cycles: %llu\n", sum);

  // printf(" %lf\n",
  //        (OPS) / ((double)(sum / (1.0 * RUNS)) * (MAX_FREQ /
  //        BASE_FREQ)));

  printf("out: %f\n", out[0]);

  // Clean up
  free(Ax);
  free(Ay);
  free(Bx);
  free(By);
  free(Cx);
  free(Cy);
  free(Dx);
  free(Dy);
  free(out);

  return 0;
}
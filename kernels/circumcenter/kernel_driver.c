#include "kernel.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BASE_FREQ 2.4

#define MAX_FREQ 3.4

#define ALIGNMENT 64

// #define SIMD_SIZE 8

#define NUM_ELEMS 6

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
  float *partUx;
  float *partUy;
  float *partD;
  float *Ux;
  float *Uy;

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
  posix_memalign((void **)&partUx, ALIGNMENT,
                 NUM_ELEMS * SIMD_SIZE * sizeof(float));
  posix_memalign((void **)&partUy, ALIGNMENT,
                 NUM_ELEMS * SIMD_SIZE * sizeof(float));
  posix_memalign((void **)&partD, ALIGNMENT,
                 NUM_ELEMS * SIMD_SIZE * sizeof(float));
  posix_memalign((void **)&Ux, ALIGNMENT,
                 NUM_ELEMS * SIMD_SIZE * sizeof(float));
  posix_memalign((void **)&Uy, ALIGNMENT,
                 NUM_ELEMS * SIMD_SIZE * sizeof(float));

  // Initialize data
  for (int i = 0; i < NUM_ELEMS * SIMD_SIZE; i++) {
    Ax[i] = 0.0;
    Ay[i] = 0.0;
    Bx[i] = 1.0;
    By[i] = 0.0;
    Cx[i] = 0.5;
    Cy[i] = 1.0;
    partUx[i] = 0.0;
    partUy[i] = 0.0;
    partD[i] = 0.0;
    Ux[i] = 0.0;
    Uy[i] = 0.0;
  }

  // Test kernels
  kernel0(Ax, Ay, Bx, By, Cx, Cy, partUx, partUy, partD);
  kernel1(partD, partUx, partUy, Ux, Uy);

  printf("First kernel: %f %f\n", Ux[0], Uy[0]);
  printf("Second kernel: %f %f\n", Ux[SIMD_SIZE], Uy[SIMD_SIZE]);
  printf("Third kernel: %f %f\n", Ux[2 * SIMD_SIZE], Uy[2 * SIMD_SIZE]);
  printf("Fourth kernel: %f %f\n", Ux[3 * SIMD_SIZE], Uy[3 * SIMD_SIZE]);
  printf("Fifth kernel: %f %f\n", Ux[4 * SIMD_SIZE], Uy[4 * SIMD_SIZE]);
  printf("Sixth kernel: %f %f\n", Ux[5 * SIMD_SIZE], Uy[5 * SIMD_SIZE]);

  // Clean up
  free(Ax);
  free(Ay);
  free(Bx);
  free(By);
  free(Cx);
  free(Cy);
  free(partUx);
  free(partUy);
  free(partD);
  free(Ux);
  free(Uy);

  return 0;
}
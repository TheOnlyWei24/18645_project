#include "kernel.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BASE_FREQ 2.4

#define MAX_FREQ 3.4

#define ALIGNMENT 64

#define SIMD_SIZE 4

static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

int main(void) {
  // Set up data structures
  double *Ax;
  double *Ay;
  double *Bx;
  double *By;
  double *Cx;
  double *Cy;
  double *partUx;
  double *partUy;
  double *partD;
  double *Ux;
  double *Uy;

  posix_memalign((void **)&Ax, ALIGNMENT, SIMD_SIZE * sizeof(double));
  posix_memalign((void **)&Ay, ALIGNMENT, SIMD_SIZE * sizeof(double));
  posix_memalign((void **)&Bx, ALIGNMENT, SIMD_SIZE * sizeof(double));
  posix_memalign((void **)&By, ALIGNMENT, SIMD_SIZE * sizeof(double));
  posix_memalign((void **)&Cx, ALIGNMENT, SIMD_SIZE * sizeof(double));
  posix_memalign((void **)&Cy, ALIGNMENT, SIMD_SIZE * sizeof(double));
  posix_memalign((void **)&partUx, ALIGNMENT, SIMD_SIZE * sizeof(double));
  posix_memalign((void **)&partUy, ALIGNMENT, SIMD_SIZE * sizeof(double));
  posix_memalign((void **)&partD, ALIGNMENT, SIMD_SIZE * sizeof(double));
  posix_memalign((void **)&Ux, ALIGNMENT, SIMD_SIZE * sizeof(double));
  posix_memalign((void **)&Uy, ALIGNMENT, SIMD_SIZE * sizeof(double));

  // Initialize data
  for (int i = 0; i < SIMD_SIZE; i++) {
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

  printf("%f %f\n", Ux[0], Uy[0]);

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
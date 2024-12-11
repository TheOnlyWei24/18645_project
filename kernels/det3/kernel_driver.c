#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "globals.h"
#include "kernel1.h"
#include "kernel2.h"


static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

void check
(
 float*     restrict Ax,
 float*     restrict Ay,
 float*     restrict Bx,
 float*     restrict By,
 float*     restrict Cx,
 float*     restrict Cy,
 float*     restrict Dx,
 float*     restrict Dy,
 float*     restrict res,
 int        NUM_ELEMS
 ){
  float a, b, c, d, e, f, g, h, i;

  for (int p = 0; p < NUM_ELEMS * SIMD_SIZE; p++){
    a = Ax[p] - Dx[p];
    b = Ay[p] - Dy[p];
    d = Bx[p] - Dx[p];
    e = By[p] - Dy[p];
    g = Cx[p] - Dx[p];
    h = Cy[p] - Dy[p];
    c = a*a + b*b;
    f = d*d + e*e;
    i = g*g + h*h;
    res[p] = a*(e*i - f*h) + b*(f*g - d*i) + c*(d*h - e*g);
  }
 }

int main(){

  float *Ax, *Ay;
  float *Bx, *By;
  float *Cx, *Cy;
  float *Dx, *Dy;
  float *kernel1_out, *kernel2_out;
  float *res;

  unsigned long long t0, t1, t2, t3, t4, t5, tp0, tp1;
  int NUM_THREADS = 1;
  int threads[10] = {1, 2, 4, 8, 12, 16, 20, 24, 32, 40};
  int elems[10] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
  unsigned long long sum_kernel2 = 0;
  //int data_dim[10] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};





        int idx = 0;
        for (int p = 0; p < (int)((NUM_ELEMS*SIMD_SIZE)/KERNEl1_SIZE); p++){
          idx = KERNEl1_SIZE*p;
        }
      }

    }
  }
  // Clean up
  free(Ax);
  free(Ay);
  free(Bx);
  free(By);
  free(Cx);
  free(Cy);
  free(Dx);
  free(Dy);
  free(kernel1_out);
  free(kernel2_out);
  free(res);

  return 0;
}
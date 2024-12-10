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
  unsigned long long sum_kernel2 = 0;
  //int data_dim[10] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

  for (int t=0; t<10; t++){
    //int NUM_ELEMS = data_dim[t];
    int NUM_ELEMS = 32768;
    NUM_THREADS = threads[t];
    //create memory aligned buffers
    posix_memalign((void**) &Ax, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));
    posix_memalign((void**) &Ay, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));
    posix_memalign((void**) &Bx, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));
    posix_memalign((void**) &By, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));
    posix_memalign((void**) &Cx, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));
    posix_memalign((void**) &Cy, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));
    posix_memalign((void**) &Dx, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));
    posix_memalign((void**) &Dy, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));
    posix_memalign((void**) &kernel1_out, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));
    posix_memalign((void**) &res, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));
    posix_memalign((void**) &kernel2_out, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));

    srand((unsigned int)time(NULL));
    float scale = 64;
    float shift = 32;
    //initialize A
    for (int i = 0; i < NUM_ELEMS * SIMD_SIZE; i++){
      Ax[i] = (((float) rand())/ ((float) RAND_MAX))*scale - shift;
      Ay[i] = (((float) rand())/ ((float) RAND_MAX))*scale - shift;
    }
    //initialize B
    for (int i = 0; i < NUM_ELEMS * SIMD_SIZE; i++){
      Bx[i] = (((float) rand())/ ((float) RAND_MAX))*scale - shift;
      By[i] = (((float) rand())/ ((float) RAND_MAX))*scale - shift;
    }
    //initialize C
    for (int i = 0; i < NUM_ELEMS * SIMD_SIZE; i++){
      Cx[i] = (((float) rand())/ ((float) RAND_MAX))*scale - shift;
      Cy[i] = (((float) rand())/ ((float) RAND_MAX))*scale - shift;
    }

    //initialize D
    for (int i = 0; i < NUM_ELEMS * SIMD_SIZE; i++){
      Dx[i] = (((float) rand())/ ((float) RAND_MAX))*scale - shift;
      Dy[i] = (((float) rand())/ ((float) RAND_MAX))*scale - shift;
    }
    //initialize output
    for (int i = 0; i < NUM_ELEMS * SIMD_SIZE; i++){
      kernel1_out[i] = 0.0;
      kernel2_out[i] = 0.0;
      res[i] = 0.0;
    }

    unsigned long long sum_kernel1 = 0;
    //unsigned long long sum_kernel2 = 0;
    unsigned long long sum_check = 0;

    for (int r = 0; r<RUNS; r++){

      const int KERNEl1_SIZE = 4*SIMD_SIZE;

      // Run kernel sequential
      if (t==0){
        int idx = 0;
        t2 = rdtsc();
        for (int p = 0; p < (int)((NUM_ELEMS*SIMD_SIZE)/KERNEl1_SIZE); p++){
          idx = KERNEl1_SIZE*p;
          kernel2(&Ax[idx], &Ay[idx], &Bx[idx], &By[idx], &Cx[idx], &Cy[idx], &Dx[idx], &Dy[idx], &kernel2_out[idx]);
        }
        t3 = rdtsc();
        sum_kernel2 += (t3 - t2);
      }
    }
    

    for (int r = 0; r<RUNS; r++){

      const int KERNEl1_SIZE = 4*SIMD_SIZE;

      // Run kernel parallel
      int idx = 0;
      t0 = rdtsc();
      #pragma omp parallel for private(idx) num_threads(NUM_THREADS) //reduction(+:sum_kernel1)
      for (int p = 0; p < (int)((NUM_ELEMS*SIMD_SIZE)/KERNEl1_SIZE); p++){
        idx = KERNEl1_SIZE*p;
        kernel2(&Ax[idx], &Ay[idx], &Bx[idx], &By[idx], &Cx[idx], &Cy[idx], &Dx[idx], &Dy[idx], &kernel1_out[idx]);
      }
      t1 = rdtsc();
      sum_kernel1 += (t1 - t0); 
    }

    // Run correctness check
    // for (int r = 0; r<RUNS; r++){
    //   check(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, res, NUM_ELEMS);
    // }

    // int correct = 1;
    // for (int i = 0; i != NUM_ELEMS * SIMD_SIZE; ++i) {
    //   correct &= (fabs(kernel1_out[i] - res[i]) < 1e-13);
    // }
    
    printf("\n");
    printf("Thread Count: %d\n", NUM_THREADS);
    // printf("Correctness: %d \n", correct);
    printf("Parallel cycles/RUNS: %llu, throughput: %lf\n", sum_kernel1 / RUNS, (30*NUM_ELEMS*SIMD_SIZE) / ( (double)(sum_kernel1 / RUNS) * (MAX_FREQ/BASE_FREQ) ));
    printf("Sequential cycles/RUNS: %llu, throughput: %lf\n", sum_kernel2 / RUNS, (30*NUM_ELEMS*SIMD_SIZE) / ( (double)(sum_kernel2 / RUNS) * (MAX_FREQ/BASE_FREQ) ));
    printf("Speed-up: %f\n\n", (float)((30*NUM_ELEMS*SIMD_SIZE) / ( (double)(sum_kernel1 / RUNS) * (MAX_FREQ/BASE_FREQ) )) / (float)((30*NUM_ELEMS*SIMD_SIZE) / ( (double)(sum_kernel2 / RUNS) * (MAX_FREQ/BASE_FREQ) )));
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
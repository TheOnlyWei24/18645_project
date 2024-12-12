#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "globals.h"
#include "kernel.h"
unsigned long long tg0, tg1, sum_check, tg0s, tg1s, sum_check_seq;

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
 float      Dx,
 float      Dy,
 float*     restrict res,
 int        NUM_ELEMS
 ){
  float a, b, c, d, e, f, g, h, i;
  tg0 = rdtsc();
  for (int p = 0; p < NUM_ELEMS * SIMD_SIZE; p++){
    
    a = Ax[p] - Dx;
    b = Ay[p] - Dy;
    d = Bx[p] - Dx;
    e = By[p] - Dy;
    g = Cx[p] - Dx;
    h = Cy[p] - Dy;
    c = a*a + b*b;
    f = d*d + e*e;
    i = g*g + h*h;
    res[p] = a*(e*i - f*h) + b*(f*g - d*i) + c*(d*h - e*g);
    
  }
  tg1 = rdtsc();
  sum_check += (tg1-tg0);
 }

void check_seq
(
 float*     restrict Ax,
 float*     restrict Ay,
 float*     restrict Bx,
 float*     restrict By,
 float*     restrict Cx,
 float*     restrict Cy,
 float      Dx,
 float      Dy,
 float*     restrict res,
 int        NUM_ELEMS
 ){
  float a, b, c, d, e, f, g, h, i;
  
  for (int p = 0; p < NUM_ELEMS * SIMD_SIZE; p++){
    tg0s = rdtsc();
    a = Ax[p] - Dx;
    b = Ay[p] - Dy;
    d = Bx[p] - Dx;
    e = By[p] - Dy;
    g = Cx[p] - Dx;
    h = Cy[p] - Dy;
    c = a*a + b*b;
    f = d*d + e*e;
    i = g*g + h*h;
    res[p] = a*(e*i - f*h) + b*(f*g - d*i) + c*(d*h - e*g);
    tg1s = rdtsc();
    sum_check_seq += (tg1s-tg0s);
  }
 }

int main(){

  float *Ax, *Ay;
  float *Bx, *By;
  float *Cx, *Cy;
  float *Dx, *Dy;
  float *kernel_out;
  float *res, *res_seq;

  unsigned long long t0, t1, t2, t3, t4, t5, tp0, tp1;
  int NUM_THREADS = 1;
  int threads[10] = {1, 2, 4, 8, 12, 16, 20, 24, 32, 40};
  int elems[10] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};

  for (int s=0; s<10; s++){
    for (int t=0; t<1; t++){
      int NUM_ELEMS = elems[s];
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
      posix_memalign((void**) &kernel_out, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));
      posix_memalign((void**) &res, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));
      posix_memalign((void**) &res_seq, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));

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
        kernel_out[i] = 0.0;
        res[i] = 0.0;
        res_seq[i] = 0.0;
      }

      unsigned long long sum_kernel = 0;
      sum_check = 0;
      sum_check_seq = 0;

    for (int r = 0; r<RUNS; r++){
      int idx = 0;
      check(Ax, Ay, Bx, By, Cx, Cy, Dx[0], Dy[0], res, NUM_ELEMS);
    }

    for (int r = 0; r<RUNS; r++){
      int idx = 0;
      check_seq(Ax, Ay, Bx, By, Cx, Cy, Dx[0], Dy[0], res_seq, NUM_ELEMS);
    }

    for (int r = 0; r<RUNS; r++){
      const int KERNEl_SIZE = 2*SIMD_SIZE;

      // Run kernel single-threaded
      t2 = rdtsc();
      int idx = 0;
      for (int p = 0; p < (int)((NUM_ELEMS*SIMD_SIZE)/KERNEl_SIZE); p++){
        idx = KERNEl_SIZE*p;
        kernel(&Ax[idx], &Ay[idx], &Bx[idx], &By[idx], &Cx[idx], &Cy[idx], Dx[0], Dy[0], &kernel_out[idx]);
      }
      t3 = rdtsc();
      sum_kernel += (t3 - t2);
    }

      // for (int r = 0; r<RUNS; r++){

      //   const int KERNEl_SIZE = 2*SIMD_SIZE;

      //   // Run kernel parallel
      //   int idx = 0;
      //   t0 = rdtsc();
      //   #pragma omp parallel for private(idx) num_threads(NUM_THREADS)
      //   for (int p = 0; p < (int)((NUM_ELEMS*SIMD_SIZE)/KERNEl_SIZE); p++){
      //     idx = KERNEl1_SIZE*p;
      //     kernel(&Ax[idx], &Ay[idx], &Bx[idx], &By[idx], &Cx[idx], &Cy[idx], Dx[0], Dy[0], &kernel_out[idx]);
      //   }
      //   t1 = rdtsc();
      //   sum_kernel += (t1 - t0); 
      // }

      // Run correctness check
      int correct = 1;
      for (int i = 0; i != NUM_ELEMS * SIMD_SIZE; ++i) {
        correct &= (fabs(kernel_out[i] - res[i]) < 1e-13);
      }
      
      printf("\n");
      printf("Num elems: %d\n", NUM_ELEMS);
      printf("Thread Count: %d\n", NUM_THREADS);
      printf("Correctness: %d \n", correct);
      printf("Check cycles/RUNS: %llu, throughput: %lf\n", sum_check / RUNS, (30*NUM_ELEMS*SIMD_SIZE) / ( (double)(sum_check / RUNS) * (MAX_FREQ/BASE_FREQ) ));
      printf("Check_seq cycles/RUNS: %llu, throughput: %lf\n", sum_check_seq / RUNS, (30*NUM_ELEMS*SIMD_SIZE) / ( (double)(sum_check_seq / RUNS) * (MAX_FREQ/BASE_FREQ) ));
      printf("Kernel cycles/RUNS: %llu, throughput: %lf\n", sum_kernel / RUNS, (30*NUM_ELEMS*SIMD_SIZE) / ( (double)(sum_kernel / RUNS) * (MAX_FREQ/BASE_FREQ) ));
      printf("Speed-up: %f\n", (float)((30*NUM_ELEMS*SIMD_SIZE) / ( (double)(sum_kernel / RUNS) * (MAX_FREQ/BASE_FREQ) )) / (float)((30*NUM_ELEMS*SIMD_SIZE) / ( (double)(sum_check / RUNS) * (MAX_FREQ/BASE_FREQ) )));
      printf("Speed-up_seq: %f\n\n", (float)((30*NUM_ELEMS*SIMD_SIZE) / ( (double)(sum_kernel / RUNS) * (MAX_FREQ/BASE_FREQ) )) / (float)((30*NUM_ELEMS*SIMD_SIZE) / ( (double)(sum_check_seq / RUNS) * (MAX_FREQ/BASE_FREQ) )));
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
  free(kernel_out);
  free(res_seq);
  free(res);

  return 0;
}
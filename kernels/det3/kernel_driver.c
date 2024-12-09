#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "globals.h"
#include "kernel1.h"
#include "kernel2.h"

//timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}
//void kernel2(float * restrict Ax, float * restrict Ay, float * restrict Bx, float * restrict By, float * restrict Cx, float * restrict Cy, float * restrict Dx, float * restrict Dy, float * restrict out);

// void kernel
// (
//  float*     restrict Ax,
//  float*     restrict Ay,
//  float*     restrict Bx,
//  float*     restrict By,
//  float*     restrict Cx,
//  float*     restrict Cy,
//  float*     restrict Dx,
//  float*     restrict Dy,
//  float*     restrict det3_out
//  );

// __attribute__((optimize("no-tree-vectorize")))
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

/*
 void baseline(float * restrict Ax, float * restrict Ay, float * restrict Bx, float * restrict By, float * restrict Cx, float * restrict Cy,
              float * restrict Dx, float * restrict Dy, float * restrict out) {
  for (int z = 0; z < (8 * NUM_ELEMS); z++) {
    float a = Ax[z] - Dx[z];
    float b = Ay[z] - Dy[z];
    float d = Bx[z] - Dx[z];
    float e = By[z] - Dy[z];
    float g = Cx[z] - Dx[z];
    float h = Cy[z] - Dy[z];

    float c = (a * a) + (b * b);
    float f = (d * d) + (e * e);
    float i = (g * g) + (h * h);

    float out0 = a * ((e * i) - (f * h));
    float out1 = b * ((f * g) - (d * i));
    float out2 = c * ((d * h) - (e * g));

    out[z] = out0 + out1 + out2;
  }
}
*/

int main(){

  float *Ax, *Ay;
  float *Bx, *By;
  float *Cx, *Cy;
  float *Dx, *Dy;
  float *kernel1_out, *kernel2_out;
  float *res;

  unsigned long long t0, t1, t2, t3, t4, t5;
  int data_dim[11] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};

  for (int t=0; t<11; t++){
    int NUM_ELEMS = data_dim[t];
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
    unsigned long long sum_kernel2 = 0;
    unsigned long long sum_check = 0;
    
    for (int r = 0; r<RUNS; r++){
      // Run kernel 1
      int idx = 0;
      const int KERNEl1_SIZE = 2*SIMD_SIZE;
      for (int p = 0; p < (int)((NUM_ELEMS*SIMD_SIZE)/KERNEl1_SIZE); p++){
        idx = KERNEl1_SIZE*p;
        t0 = rdtsc();
        kernel(&Ax[idx], &Ay[idx], &Bx[idx], &By[idx], &Cx[idx], &Cy[idx], &Dx[idx], &Dy[idx], &kernel1_out[idx]);
        t1 = rdtsc();
        sum_kernel1 += (t1 - t0);  
      }
      
      // Run kernel 2
      idx = 0;
      const int KERNEl2_SIZE = 4*SIMD_SIZE;
      for (int p = 0; p < (int)((NUM_ELEMS*SIMD_SIZE)/KERNEl2_SIZE); p++){
        idx = KERNEl2_SIZE*p;
        t2 = rdtsc();
        kernel2(&Ax[idx], &Ay[idx], &Bx[idx], &By[idx], &Cx[idx], &Cy[idx], &Dx[idx], &Dy[idx], &kernel2_out[idx]);
        t3 = rdtsc();
        sum_kernel2 += (t3 - t2); 
      }

      // Run correctness check
      t4 = rdtsc();
      check(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, res, NUM_ELEMS);
      t5 = rdtsc();
      sum_check += (t5 - t4);
    }

    int correct = 1;
    for (int i = 0; i != NUM_ELEMS * SIMD_SIZE; ++i) {
      correct &= (fabs(kernel1_out[i] - res[i]) < 1e-13);
    }

    printf("check cycles/RUNS: %llu, throughput: %lf\n", sum_check / RUNS, (30*NUM_ELEMS*SIMD_SIZE) / ( (double)(sum_check / RUNS) * (MAX_FREQ/BASE_FREQ) ));
    printf("kernel1 cycles/RUNS: %llu, throughput: %lf\n", sum_kernel1 / RUNS, (30*NUM_ELEMS*SIMD_SIZE) / ( (double)(sum_kernel1 / RUNS) * (MAX_FREQ/BASE_FREQ) ));
    printf("kernel2 cycles/RUNS: %llu, throughput: %lf\n", sum_kernel2 / RUNS, (30*NUM_ELEMS*SIMD_SIZE) / ( (double)(sum_kernel2 / RUNS) * (MAX_FREQ/BASE_FREQ) ));
    printf("%d \n", correct);
  }

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

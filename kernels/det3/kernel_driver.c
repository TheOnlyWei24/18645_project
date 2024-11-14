#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "immintrin.h"
#include <time.h>

#define MAX_FREQ 3.4
#define BASE_FREQ 2.4
#define RUNS 10000
static const int SIMD_SIZE = 8;
static const int NUM_ELEMS = 4;
static const int ALIGNMENT = 32;

//timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

void kernel
(
 float*     restrict Ax,
 float*     restrict Ay,
 float*     restrict Bx,
 float*     restrict By,
 float*     restrict Cx,
 float*     restrict Cy,
 float*     restrict Dx,
 float*     restrict Dy,
 float*     restrict det3_out
 );

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
 float*     restrict res
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

 void baseline(float *Ax, float *Ay, float *Bx, float *By, float *Cx, float *Cy,
              float *Dx, float *Dy, float *out) {
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

int main(){

  float *Ax, *Ay;
  float *Bx, *By;
  float *Cx, *Cy;
  float *Dx, *Dy;
  float *det3_out;
  float *res;

  unsigned long long t0, t1, t2, t3;
  
  //create memory aligned buffers
  posix_memalign((void**) &Ax, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));
  posix_memalign((void**) &Ay, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));
  posix_memalign((void**) &Bx, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));
  posix_memalign((void**) &By, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));
  posix_memalign((void**) &Cx, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));
  posix_memalign((void**) &Cy, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));
  posix_memalign((void**) &Dx, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));
  posix_memalign((void**) &Dy, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));
  posix_memalign((void**) &det3_out, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));
  posix_memalign((void**) &res, ALIGNMENT, NUM_ELEMS * SIMD_SIZE * sizeof(float));

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
    det3_out[i] = 0.0;
    res[i] = 0.0;
  }

  unsigned long long sum = 0;
  unsigned long long sum_check = 0;
  for (int r = 0; r<RUNS; r++){

    t0 = rdtsc();
    kernel(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, det3_out);
    t1 = rdtsc();
    sum += (t1 - t0);  

    t2 = rdtsc();
    check(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, res);
    t3 = rdtsc();
    sum_check += (t3 - t2); 
  }

  baseline(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, res);
  int correct = 1;
  for (int i = 0; i != NUM_ELEMS * SIMD_SIZE; ++i) {
    correct &= (fabs(det3_out[i] - res[i]) < 1e-13);
  }

  printf("baseline cycles/RUNS: %llu\n", sum_check / RUNS);
  printf("kernel cycles/RUNS: %llu\n", sum / RUNS);
  // printf("%llu, %llu\t, %d\n", (sum / RUNS), (sum_check / RUNS), correct);

  free(Ax);
  free(Ay);
  free(Bx);
  free(By);
  free(Cx);
  free(Cy);
  free(Dx);
  free(Dy);
  free(det3_out);

  return 0;
}

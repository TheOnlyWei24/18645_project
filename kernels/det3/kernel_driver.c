#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "immintrin.h"
#include <time.h>

#define MAX_FREQ 3.4
#define BASE_FREQ 2.4
#define RUNS 1

//timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

void kernel
(
 int               m,
 int               n,
 int               k,
 double*     restrict Ax,
 double*     restrict Ay,
 double*     restrict Bx,
 double*     restrict By,
 double*     restrict Cx,
 double*     restrict Cy,
 double*     restrict Dx,
 double*     restrict Dy,
 double*     restrict det3_out
 );

void check
(
 int               m,
 int               n,
 int               k,
 double*     restrict Ax,
 double*     restrict Ay,
 double*     restrict Bx,
 double*     restrict By,
 double*     restrict Cx,
 double*     restrict Cy,
 double*     restrict Dx,
 double*     restrict Dy,
 double*     restrict res
 ){
  double a, b, c, d, e, f, g, h, i;

  for (int p = 0; p < m*n*k; p++){
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

  double *Ax, *Ay;
  double *Bx, *By;
  double *Cx, *Cy;
  double *Dx, *Dy;
  double *det3_out;
  double *res;

  unsigned long long t0, t1, t2, t3;

  // Kernel dim
  int m = 1;
  int n = 8;
  int k = 1024;
  
  //create memory aligned buffers
  posix_memalign((void**) &Ax, 64, m * n * k * sizeof(double));
  posix_memalign((void**) &Ay, 64, m * n * k * sizeof(double));
  posix_memalign((void**) &Bx, 64, m * n * k * sizeof(double));
  posix_memalign((void**) &By, 64, m * n * k * sizeof(double));
  posix_memalign((void**) &Cx, 64, m * n * k * sizeof(double));
  posix_memalign((void**) &Cy, 64, m * n * k * sizeof(double));
  posix_memalign((void**) &Dx, 64, m * n * k * sizeof(double));
  posix_memalign((void**) &Dy, 64, m * n * k * sizeof(double));
  posix_memalign((void**) &det3_out, 64, m * n * k * sizeof(double));
  posix_memalign((void**) &res, 64, m * n * k * sizeof(double));

  srand((unsigned int)time(NULL));
  double scale = 64;
  double shift = 32;
  //initialize A
  for (int i = 0; i < k * m * n; i++){
    Ax[i] = (((double) rand())/ ((double) RAND_MAX))*scale - shift;
    Ay[i] = (((double) rand())/ ((double) RAND_MAX))*scale - shift;
  }
  //initialize B
  for (int i = 0; i < k * m * n; i++){
    Bx[i] = (((double) rand())/ ((double) RAND_MAX))*scale - shift;
    By[i] = (((double) rand())/ ((double) RAND_MAX))*scale - shift;
  }
  //initialize C
  for (int i = 0; i < k * m * n; i++){
    Cx[i] = (((double) rand())/ ((double) RAND_MAX))*scale - shift;
    Cy[i] = (((double) rand())/ ((double) RAND_MAX))*scale - shift;
  }

  //initialize D
  for (int i = 0; i < k * m * n; i++){
    Dx[i] = (((double) rand())/ ((double) RAND_MAX))*scale - shift;
    Dy[i] = (((double) rand())/ ((double) RAND_MAX))*scale - shift;
  }
  //initialize output
  for (int i = 0; i < k * m * n; i++){
    det3_out[i] = 0.0;
    res[i] = 0.0;
  }

  unsigned long long sum = 0;
  unsigned long long sum_check = 0;
  for (int r = 0; r<RUNS; r++){

    t0 = rdtsc();
    kernel(m, n, k, Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, det3_out);
    t1 = rdtsc();
    sum += (t1 - t0);  

    t2 = rdtsc();
    check(m, n, k, Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, res);
    t3 = rdtsc();
    sum_check += (t3 - t2); 
  }

  check(m, n, k, Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, res);
  int correct = 1;
  for (int i = 0; i != m * n * k; ++i) {
    correct &= (fabs(det3_out[i] - res[i]) < 1e-13);
  }

  // printf("%d\t %d\t %d\t %lf\t %lf\t %d\n", m, n, k, (29.0*m*n*k)/((double)(sum/(1.0*RUNS))*(MAX_FREQ/BASE_FREQ)), (29.0*m*n*k)/((double)(sum_check/(1.0*RUNS))*(MAX_FREQ/BASE_FREQ)), correct);
  printf("%ld\t %ld\t\n", sum, sum_check);

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

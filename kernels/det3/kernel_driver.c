#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "immintrin.h"

#define MAX_FREQ 3.4
#define BASE_FREQ 2.4

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

int main(){

  double *Ax, *Ay;
  double *Bx, *By;
  double *Cx, *Cy;
  double *Dx, *Dy;
  double *det3_out;

  unsigned long long t0, t1;

  // Kernel dim
  int m = 1;
  int n = 8;
  int k = 1;
  
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

  //initialize A
  for (int i = 0; i < k * m * n; i++){
    Ax[i] = 2;
    Ay[i] = 3;
  }
  //initialize B
  for (int i = 0; i < k * m * n; i++){
    Bx[i] = 4;
    By[i] = 5;
  }
  //initialize C
  for (int i = 0; i < k * m * n; i++){
    Cx[i] = 6;
    Cy[i] = 7;
  }
  //initialize D
  for (int i = 0; i < k * m * n; i++){
    Dx[i] = 1;
    Dy[i] = 1;
  }
  //initialize output
  for (int i = 0; i < k * m * n; i++){
    det3_out[i] = 0.0;
  }

  t0 = rdtsc();

  kernel(m, n, k, Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, det3_out);

  t1 = rdtsc();


  // int correct = 1;
  // for (int i = 0; i != m * n; ++i) {
  //   correct &= (fabs(c[i] - c_check[i]) < 1e-13);
  // }

  //printf("%d\t %d\t %d\t %lf %d\n", m, n, k, (2.0*m*n*k)/((double)(t1-t0)*MAX_FREQ/BASE_FREQ), correct);
  printf("det3_out = %f, %f\n", det3_out[0], det3_out[4]);

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

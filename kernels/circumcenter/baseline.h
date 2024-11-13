#ifndef _BASELINE_KERNEL_H_
#define _BASELINE_KERNEL_H_

void baseline(float *Ax, float *Ay, float *Bx, float *By, float *Cx, float *Cy,
              float *Ux, float *Uy) {
  for (int i = 0; i < (8 * 6); i++) {
    float Ax2_Ay2 = (Ax[i] * Ax[i]) + (Ay[i] * Ay[i]);
    float Bx2_Bx2 = (Bx[i] * Bx[i]) + (By[i] * By[i]);
    float Cx2_Cx2 = (Cx[i] * Cx[i]) + (Cy[i] * Cy[i]);

    float D = 2 * (((By[i] - Cy[i]) * Ax[i]) + ((Cy[i] - Ay[i]) * Bx[i]) +
                   ((Ay[i] - By[i]) * Cx[i]));

    Ux[i] = ((Ax2_Ay2 * (By[i] - Cy[i])) + (Bx2_Bx2 * (Cy[i] - Ay[i])) +
             (Cx2_Cx2 * (Ay[i] - By[i]))) /
            D;

    Uy[i] = ((Ax2_Ay2 * (Cx[i] - Bx[i])) + (Bx2_Bx2 * (Ax[i] - Cx[i])) +
             (Cx2_Cx2 * (Bx[i] - Ax[i]))) /
            D;
  }
}

#endif
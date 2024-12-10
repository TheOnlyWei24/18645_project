#ifndef _BASELINE_KERNEL_H_
#define _BASELINE_KERNEL_H_

#define NUM_ELEMS 4

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

#endif
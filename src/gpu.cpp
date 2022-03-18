#include "stock_market_monte_carlo/gpu.h"

#include <iostream>

using namespace std;

void vector_add(float *out, float *a, float *b, int n) {
  clock_t start = clock();
  for (int i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
  clock_t end = clock();
  double time = (double)(end - start) / CLOCKS_PER_SEC;
  printf("CPU time: %f\n", time);
}
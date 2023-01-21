#include <cstdlib>

#include "stock_market_monte_carlo/gpu.h"
#include "stock_market_monte_carlo/helpers.h"
#include "stock_market_monte_carlo/simulations.h"

int main(int argc, char **argv) {
  int N;
  char *tmp;
  if (argc == 2)
    N = std::strtol(argv[1], &tmp, 10);
  else
    N = 1000000;

  float *a, *b, *out;

  // Allocate memory
  a = (float *)malloc(sizeof(float) * N);
  b = (float *)malloc(sizeof(float) * N);
  out = (float *)malloc(sizeof(float) * N);

  // Initialize array
  for (int i = 0; i < N; i++) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }

  //    // CPU
  vector_add(out, a, b, N);

  //     GPU
  vector_add_gpu(out, a, b, N);

  std::vector<float> out_v(out, out + N);
  print_vector(out_v);
}

#include <cstdlib>

#include "stock_market_monte_carlo/gpu.h"
#include "stock_market_monte_carlo/helpers.h"
#include "stock_market_monte_carlo/simulations.h"

//unsigned TausStep(unsigned &z, int S1, int S2, int S3, unsigned M) {
//  unsigned b = (((z << S1) ^ z) >> S2);
//  return z = (((z & M) << S3) ^ b);
//}
//unsigned LCGStep(unsigned &z, unsigned A, unsigned C) {
//  return z = (A * z + C);
//}
//float HybridTaus(unsigned int& z1, unsigned int &z2, unsigned int &z3, unsigned int &z4) {
//  // Combined period is lcm(p1,p2,p3,p4)~ 2^121
//  return 2.3283064365387e-10 * (TausStep(z1, 13, 19, 12, 4294967294UL) ^
//                                TausStep(z2, 2, 25, 4, 4294967288UL) ^
//                                TausStep(z3, 3, 11, 17, 4294967280UL) ^
//                                LCGStep(z4, 1664525, 1013904223UL));
//}
//
//void testRNG(int n){
//  unsigned int rstate[4];
//  for (int i=0; i<4;i++)
//    rstate[i] = (i+1) * 12371;
//
//  for (int i=0; i<n; i++) {
//    float next_val = HybridTaus(rstate[0], rstate[1], rstate[2], rstate[3]);
//    printf("%f ", next_val);
//    printf("%d \n", rstate[0]);
//
//  }
//}

int main(int argc, char **argv) {
//  int N;
//  char *tmp;
//  if (argc == 2)
//    N = std::strtol(argv[1], &tmp, 10);
//  else
//    N = 1000000;
//
//  float *a, *b, *out;
//
//  // Allocate memory
//  a = (float *)malloc(sizeof(float) * N);
//  b = (float *)malloc(sizeof(float) * N);
//  out = (float *)malloc(sizeof(float) * N);
//
//  // Initialize array
//  for (int i = 0; i < N; i++) {
//    a[i] = 1.0f;
//    b[i] = 2.0f;
//  }
//
//  //    // CPU
//  vector_add(out, a, b, N);
//
//  //     GPU
//  vector_add_gpu(out, a, b, N);

  //    std::vector<float> out_v(out, out + N);
  //    print_vector(out_v);
  fmt::print("starting RNG test...");
  testRNG(1000); // TODO for some reason doesn't give any output...
}
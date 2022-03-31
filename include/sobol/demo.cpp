#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>

#include "sobol.hpp"

int main(int argc, char *argv[]) {
  char *end;
  int m = std::strtol(argv[1], &end, 10);
  int n = std::strtol(argv[2], &end, 10);
  int skip = std::strtol(argv[3], &end, 10);
  printf("m: %d | n: %d | skip: %d", m,n,skip);

  double *values = i8_sobol_generate(m, n, skip);
  for (int i=0; i<m*n; i++){
    if (i%m == 0)
      printf("\n");
    printf("%f ", values[i]);
  }
  // todo how to generate 1 by 1?

}
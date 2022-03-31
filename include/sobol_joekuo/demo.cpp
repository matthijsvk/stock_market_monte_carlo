#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>

#include "sobol.cc"

int main(int argc, char *argv[]) {
  char *end;
  int n = atoi(argv[1]);
  int d = atoi(argv[2]);
  int skip = atoi(argv[3]);
  printf("n: %d | d: %d | skip: %d", n,d,skip);

  double **P = sobol_points(n,d,argv[3]);

  for (int i=0; i<m*n; i++){
    printf("%f ", values[i]);
  }
  // todo how to generate 1 by 1?

}

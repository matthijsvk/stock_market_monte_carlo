unsigned TausStep(unsigned int &z, int S1, int S2, int S3, unsigned int M) {
  unsigned b = (((z << S1) ^ z) >> S2);
  return z = (((z & M) << S3) ^ b);
}
unsigned LCGStep(unsigned int &z, unsigned int A, unsigned int C) {
  return z = (A * z + C);
}
float HybridTaus(unsigned int &z1,
                 unsigned int &z2,
                 unsigned int &z3,
                 unsigned int &z4) {
  // Combined period is lcm(p1,p2,p3,p4)~ 2^121
  return float(2.3283064365387e-10) * (TausStep(z1, 13, 19, 12, 4294967294UL) ^
      TausStep(z2, 2, 25, 4, 4294967288UL) ^
      TausStep(z3, 3, 11, 17, 4294967280UL) ^
      LCGStep(z4, 1664525, 1013904223UL));
}
//HybridTaus(rstate[0], rstate[1], rstate[2], rstate[3]);

unsigned int rstate[] = {N, 21701, 1297, 65537};

void kernel(float *returns_arr, unsigned int n_returns, unsigned int N) {
  // N steps, every step sample random idx and take corresponding return from returns_arr to update total
  __shared__ float returns_arr_shmem[n_returns];
  // copy returns_arr to shared memory buffer...

  // main compute loop
  float total = 100.0;
  for (unsigned int i = 0; i < N; i++) {
    unsigned int return_idx = n_returns * get_random_number();
    float percent_increase = returns_arr_shmem[return_idx];
    total = total * (float(100.0) + percent_increase) / 100;
  }
}

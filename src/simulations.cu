#include <cuda.h>

//__global__ void update_fund(float fund_value, float period_return, float &new_fund_value) {
//    new_fund_value = fund_value * (float(100.0) + period_return) / 100;
//}

__global__ void __many_updates_gpu(float* returns, float* totals, int n_periods){
    for (int i=0; i<n_periods; i++) {
        totals[i+1] = totals[i] * (float(100.0) + returns[i]) / 100;
//        update_fund(totals[i], returns[i], totals[i+1]);
    }
}
/*
  Run with:
    nvcc -Xptxas=-dlcm=ca l2_peak_bandwidth.cu -o peak
    nvprof --print-gpu-summary ./peak
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

#define SIZE (1024*1024)

__global__ void withl2(int *a, int *b) {
  int tid = threadIdx.x;
  int tid2 = blockIdx.x*blockDim.x + threadIdx.x;
  a[tid2] = b[tid];
}

__global__ void withoutl2(int *a, int *b) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  a[tid] = b[tid];
}

int main() {
  int *a, *b;
  checkCudaErrors(cudaMalloc(&a, SIZE*sizeof(int)));
  checkCudaErrors(cudaMalloc(&b, SIZE*sizeof(int)));
  int threads = 512;
  int blocks = SIZE/512;
    
  withl2<<<blocks,threads>>>(a, b);
  checkCudaErrors(cudaDeviceSynchronize());
  
  withoutl2<<<blocks,threads>>>(a, b);
  checkCudaErrors(cudaDeviceSynchronize());
  
  return 0;
}
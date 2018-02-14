/* To run: 

nvcc -Xptxas=dlcm=ca l2_peak_bandwidth.cu -o peak
nvprof --print-gpu-summary ./peak
nvprof -m l2_read_throughput ./peak
nvprof -m l2_write_throughput ./peak 

*/ 

# include <stdio.h>
# include <stdint.h>
# include "cuda_runtime.h"

__global__ void global_latency (unsigned int* a, unsigned int* b, unsigned int* c, int N, int iterations) {
  for (int k = 0; k < iterations; k++) {
    for (int j = 0; j < N; j++) {
       c[j] = a[j] + b[j];
    }
  }
}

void parametric_measure_global(int N, int iterations) {
  cudaDeviceReset();
  
  // host (CPU) arrays
  unsigned int *ha0;
  unsigned int *ha1;
  unsigned int *ha2;
  ha0 = (unsigned int*) malloc(sizeof(unsigned int) * N);
  ha1 = (unsigned int*) malloc(sizeof(unsigned int) * N);
  ha2 = (unsigned int*) malloc(sizeof(unsigned int) * N);
  
  // device (GPU) arrays
  unsigned int *da0;
  unsigned int *da1;
  unsigned int *da2;
  cudaMalloc((void**) &da0, sizeof(unsigned int) * N);
  cudaMalloc((void**) &da1, sizeof(unsigned int) * N);
  cudaMalloc((void**) &da2, sizeof(unsigned int) * N);

  // initialize host (CPU) array 
  for (int i = 0; i < N; i++) {   
    ha0[i] = 1; 
    ha1[i] = 1;
    ha2[i] = 0;
  }

  // copy array from host to device
  cudaMemcpy(da0, ha0, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(da1, ha1, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(da2, ha2, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);

  cudaThreadSynchronize();
 
  // timing code
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  // launch kernel with a large number of threads to saturate bandwidth 
  dim3 Db = dim3(32,32,1);
  dim3 Dg = dim3(512,1,1);
  global_latency <<<Dg, Db>>>(da0, da1, da2, N, iterations);
  cudaEventRecord(stop);
  cudaThreadSynchronize();

  cudaError_t error_id = cudaGetLastError();
    if (error_id != cudaSuccess) {
    printf("Error kernel is %s\n", cudaGetErrorString(error_id));
  }

  cudaThreadSynchronize();

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("time in ms: %f\n", milliseconds);

  // free memory on GPu
  cudaFree(da0);
  cudaFree(da1);
  cudaFree(da2);
    
  // free memory on CPU 
  free(ha0);
  free(ha1);
  free(ha2);
  
  cudaDeviceReset();  
}

void measure_global() {
  // access 3 1 MB arrays 1000s of times
  int iterations = 1;
  int N = 1024* 1024/sizeof(unsigned int); // 1 MB
  
  printf("\n=====3 %ld MB arrays * %d times====\n", sizeof(unsigned int)*N/1024/1024, iterations);
  parametric_measure_global(N, iterations);
}

int main(){
  cudaSetDevice(0);
  
  measure_global();
  
  cudaDeviceReset();
  
  return 0;
}

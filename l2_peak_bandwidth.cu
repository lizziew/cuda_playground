/* To run: 

nvcc -Xptxas=dlcm=ca l2_peak_bandwidth.cu -o peak
nvprof --print-gpu-summary ./peak
nvprof -m l2_read_throughput ./peak
nvprof -m l2_write_throughput ./peak 

*/ 

# include <stdio.h>
# include <stdint.h>
# include "cuda_runtime.h"

__global__ void global_latency (unsigned int* my_array, int N, int iterations) {
  for (int k = 0; k < iterations; k++) {
    for (int j = 0; j < N-2; j++) {
       my_array[j+2] = my_array[j+1] + my_array[j];
    }
  }
}

void parametric_measure_global(int N, int iterations) {
  cudaDeviceReset();
  
  // host (CPU) array
  unsigned int *h_a;
  h_a = (unsigned int *) malloc(sizeof(unsigned int) * N);
  
  // device (GPU) array
  unsigned int *d_a;
  cudaMalloc((void**) &d_a, sizeof(unsigned int) * N);

  // initialize host (CPU) array 
  for (int i = 0; i < N; i++) {   
    h_a[i] = 0; 
  }

  // copy array from host to device
  cudaMemcpy(d_a, h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);

  cudaThreadSynchronize();
 
  // timing code
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  // launch kernel with a large number of threads to saturate bandwidth 
  dim3 Db = dim3(32,32,1);
  dim3 Dg = dim3(512,1,1);
  global_latency <<<Dg, Db>>>(d_a, N, iterations);
  cudaEventRecord(stop);

  cudaThreadSynchronize();

  cudaEventSynchronize(stop);

  cudaError_t error_id = cudaGetLastError();
    if (error_id != cudaSuccess) {
    printf("Error kernel is %s\n", cudaGetErrorString(error_id));
  }

  cudaThreadSynchronize();

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("time in ms: %f\n", milliseconds);

  // free memory on GPu
  cudaFree(d_a);
    
  // free memory on CPU 
  free(h_a);
  
  cudaDeviceReset();  
}

void measure_global() {
  // access 3 MB array 1000s of times
  int iterations = 100;
  int N = 3 * 1024* 1024/sizeof(unsigned int); // 3MB
  
  printf("\n=====%ld MB array * %d times====\n", sizeof(unsigned int)*N/1024/1024, iterations);
  parametric_measure_global(N, iterations);
}

int main(){
  cudaSetDevice(0);
  
  measure_global();
  
  cudaDeviceReset();
  
  return 0;
}

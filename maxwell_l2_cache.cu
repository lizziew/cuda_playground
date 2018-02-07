# include <stdio.h>
# include <stdint.h>

# include "cuda_runtime.h"

#define LEN 256

__global__ void global_latency (unsigned int* my_array, int N, int iterations, unsigned int* duration, unsigned int* index) {
  unsigned int start_time, end_time;
  unsigned int j = 0; 

  __shared__ unsigned int s_tvalue[LEN];
  __shared__ unsigned int s_index[LEN];

  for(int k = 0; k < LEN; k++){
    s_index[k] = 0;
    s_tvalue[k] = 0;
  }

  for (int k = 0; k < iterations*LEN; k++) {
    start_time = clock();
    
    j = my_array[j];
    s_index[k]= j;

    end_time = clock();

    s_tvalue[k] = end_time - start_time;
  }

  for(int k = 0; k < LEN; k++){
    index[k] = s_index[k];
    duration[k] = s_tvalue[k];
  }
}

void parametric_measure_global(int N, int iterations, int stride) {
  cudaDeviceReset();
  
  unsigned int *h_a;
  // host (CPU) array
  h_a = (unsigned int *) malloc(sizeof(unsigned int) * N);
  unsigned int *d_a;
  // device (GPU) array
  cudaMalloc((void**) &d_a, sizeof(unsigned int) * N);

  // initialize host (CPU) array 
  for (int i = 0; i < N; i++) {   
    h_a[i] = (i+stride)%N;  
  }

  // copy array from host to device
  cudaMemcpy(d_a, h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);

  unsigned int *h_index = (unsigned int*) malloc(sizeof(unsigned int)*LEN);
  unsigned int *h_duration = (unsigned int*) malloc(sizeof(unsigned int)*LEN);

  unsigned int *d_duration;
  cudaMalloc((void **) &d_duration, sizeof(unsigned int)*LEN);
  unsigned int *d_index;
  cudaMalloc((void**) &d_index, sizeof(unsigned int)*LEN );

  cudaThreadSynchronize();
  
  // launch kernel 
  dim3 Db = dim3(1);
  dim3 Dg = dim3(1,1,1);
  global_latency <<<Dg, Db>>>(d_a, N, iterations, d_duration, d_index);

  cudaThreadSynchronize();

  cudaError_t error_id = cudaGetLastError();
    if (error_id != cudaSuccess) {
    printf("Error kernel is %s\n", cudaGetErrorString(error_id));
  }

  cudaThreadSynchronize();

  cudaMemcpy((void*) h_duration, (void*) d_duration, sizeof(unsigned int)*LEN, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*) h_index, (void*) d_index, sizeof(unsigned int)*LEN, cudaMemcpyDeviceToHost);

  cudaThreadSynchronize();

  for(int i = 0; i < LEN; i++)
    printf("%d\t %d\n", h_index[i], h_duration[i]);

  // free memory on GPu
  cudaFree(d_a);
  cudaFree(d_index);
  cudaFree(d_duration);
    
  // free memory on CPU 
  free(h_a);
  free(h_index);
  free(h_duration);
  
  cudaDeviceReset();  
}

void measure_global() {
  int iterations = 1;
  int N = 1024 * 1024* 1024/sizeof(unsigned int); 
  
  for (int stride = 1; stride <= N/2; stride *= 2) {
    printf("\n=====%d GB array, cold cache miss, read 256 element====\n", sizeof(unsigned int)*N/1024/1024/1024);
    printf("Stride = %d element, %ld bytes\n", stride, stride * sizeof(unsigned int));
    parametric_measure_global(N, iterations, stride );
    printf("===============================================\n\n");
  }
}

int main(){
  cudaSetDevice(0);
  
  measure_global();
  
  cudaDeviceReset();
  
  return 0;
}

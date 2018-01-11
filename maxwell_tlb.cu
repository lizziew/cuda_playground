/*                                                                                                                
  Finds: Maxwell TLB
  Soure code based on paper https://arxiv.org/pdf/1509.02308.pdf 
*/

#include <stdio.h>
#include <stdint.h>

#include "cuda_runtime.h"

#define LEN 256

__global__ void global_latency(unsigned int* my_array, int N, int iterations, unsigned int* duration, unsigned int* index) {
  // data access latencies array
    __shared__ unsigned int s_tvalue[LEN];
    // accessed data indices array
    __shared__ unsigned int s_index[LEN];

    // initialize arrays
    for (int k = 0; k < LEN; k++) {
        s_index[k] = 0;
        s_tvalue[k] = 0;
    }

    // warm up the TLB
    unsigned int j = 0; 
    for (int k = 0; k < LEN*iterations; k++) 
        j = my_array[j];
    
    // ready to begin benchmarking
    unsigned int start_time, end_time;
    for (int k = 0; k < iterations*LEN; k++) {
        start_time = clock();
       
    // traverse array with elements initialized as indices of next memory access
        j = my_array[j];
        // handles ILP with this data dependency 
        s_index[k]= j;

        end_time = clock();
        s_tvalue[k] = end_time - start_time;
    }

    my_array[N] = j;
    my_array[N+1] = my_array[j];

    for(int k = 0; k < LEN; k++){
        index[k] = s_index[k];
        duration[k] = s_tvalue[k];
    }
}

void parametric_measure_global(int N, int iterations, int stride) {
    // destroy context
    cudaDeviceReset(); 

    cudaError_t error_id;

  // host (CPU) array   
    unsigned int * h_a;

    h_a = (unsigned int*) malloc((N+2) * sizeof(unsigned int));

    for (int i = 0; i < N; i++) {       
        h_a[i] = (i+stride) % N;    
    }
    h_a[N] = 0;
    h_a[N+1] = 0;

  // device (GPU) array
    unsigned int * d_a;

    error_id = cudaMalloc((void **) &d_a, (N+2) * sizeof(unsigned int));
    if (error_id != cudaSuccess) {
        printf("Error from allocating device array is %s\n", cudaGetErrorString(error_id));
    }

  error_id = cudaMemcpy(d_a, h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);
    if (error_id != cudaSuccess) {
        printf("Error from copying over host array is %s\n", cudaGetErrorString(error_id));
    }

  // accessed data indices array on host (CPU)
    unsigned int *h_index = (unsigned int*) malloc(LEN*sizeof(unsigned int));

  // accessed data indices array on device (GPU)
    unsigned int *d_index;
    error_id = cudaMalloc((void **) &d_index, sizeof(unsigned int)*LEN );
    if (error_id != cudaSuccess) {
        printf("Error from allocating indices array is %s\n", cudaGetErrorString(error_id));
    }

  // data access latencies array on host (CPU)
    unsigned int *h_duration = (unsigned int*) malloc(LEN*sizeof(unsigned int));

  // data access latencies array on device (GPU)
    unsigned int *d_duration;
    error_id = cudaMalloc ((void**) &d_duration, LEN*sizeof(unsigned int));
    if (error_id != cudaSuccess) {
        printf("Error from allocating latencies array is %s\n", cudaGetErrorString(error_id));
    }

  // blocks until the device has completed all preceding requested tasks
    cudaThreadSynchronize();

    // 1 x 1 block of threads
    dim3 Db = dim3(1);
  // 1 x 1 x 1 block of threads
    dim3 Dg = dim3(1,1,1);

    // launch kernel
    global_latency<<<Dg, Db>>>(d_a, N, iterations, d_duration, d_index);
    cudaThreadSynchronize();

    error_id = cudaGetLastError();
  if (error_id != cudaSuccess) {
        printf("Error from kernel is %s\n", cudaGetErrorString(error_id));
    }
    cudaThreadSynchronize();

  // copy results from GPU to CPU
  error_id = cudaMemcpy((void*) h_duration, (void*) d_duration, LEN*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (error_id != cudaSuccess) {
        printf("Error 2.0 is %s\n", cudaGetErrorString(error_id));
    }
  error_id = cudaMemcpy((void*) h_index, (void*) d_index, LEN*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (error_id != cudaSuccess) {
        printf("Error 2.1 is %s\n", cudaGetErrorString(error_id));
    }
    cudaThreadSynchronize();

    for(int i = 0; i < LEN; i++) {
        printf("%d\t %d\n", h_index[i], h_duration[i]);
  }

  // free memory on GPU
    cudaFree(d_a);
    cudaFree(d_index);
    cudaFree(d_duration);

  // free memory on CPU 
  free(h_a);
  free(h_index);
    free(h_duration);
    
  // destroy context
    cudaDeviceReset();  
}

void measure_global() {
    int iterations = 1;
    
  // 2 MB stride
    int stride = 2*1024*1024/sizeof(unsigned int); 

    //1. The L1 TLB has 16 entries. Test with N_min=28 *1024*256, N_max>32*1024*256
    //2. The L2 TLB has 65 entries. Test with N_min=128*1024*256, N_max=160*1024*256
    for (int N = 28*1024*256; N <= 46*1024*256; N+=stride) {
        printf("\n=====%3.1f MB array, warm TLB, read 256 element====\n", sizeof(unsigned int)*(float)N/1024/1024);
        printf("Stride = %d element, %d MB\n", stride, stride * sizeof(unsigned int)/1024/1024);
        parametric_measure_global(N, iterations, stride);
        printf("===============================================\n\n");
    }
}

int main() {
    // current device
    cudaSetDevice(0);

    measure_global();

  // destroy context
    cudaDeviceReset();
    return 0;
}

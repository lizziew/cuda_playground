/*
  Finds: size of the read only cache
  For Maxwell microarchitecture
  Source code based on paper https://arxiv.org/pdf/1509.02308.pdf
*/

#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"

#define SHARED_LEN 2048
#define THREAD_LEN 512

__global__ void global_latency(const unsigned int * __restrict__ my_array, int N, int iterations, unsigned int* duration, unsigned int* index) {
	unsigned int start_time, end_time;

  // data access latencies array
	__shared__ unsigned int s_tvalue[SHARED_LEN];
  // accessed data indices array
	__shared__ unsigned int s_index[SHARED_LEN];

  // initialize arrays
	for (int i = 0; i < SHARED_LEN; i++){
		s_index[i] = 0;
		s_tvalue[i] = 0;
	}

  // thread index (to execute in parallel)
  unsigned int j = threadIdx.x;
  // run thru without timing, for large arrays
	for (int i = 0; i < THREAD_LEN; i++) {
    // load read-only data cache
		j = __ldg(&my_array[j]);
  }
	
  int k = 0;
	for (int block_i = 0; block_i < iterations; block_i++) {
		k = block_i * blockDim.x + threadIdx.x;
		
		start_time = clock();

    // load read-only data cache
		j = __ldg(&my_array[j]);
    // handles ILP with this data dependency 
		s_index[k]= j;

		end_time = clock();
		s_tvalue[k] = end_time - start_time;
	}

	// copy the indices and memory latencies back to global memory
	for (int block_i = 0; block_i < iterations; block_i++) {
		k = block_i * blockDim.x  + threadIdx.x;
		index[k]= s_index[k];
		duration[k] = s_tvalue[k];
	}
}

void parametric_measure_global(int N, int iterations, int stride) {
	cudaDeviceReset(); // destroy context

	cudaError_t error_id;
	
  // host (CPU) array
	unsigned int* h_a;

	h_a = (unsigned int*) malloc(N*sizeof(unsigned int));

  for (i = 0; i < N; i++) {		
		h_a[i] = (i + stride) % N;	
	}

  // device (GPU) array
	unsigned int * d_a;

	error_id = cudaMalloc ((void **) &d_a, N*sizeof(unsigned int));
	if (error_id != cudaSuccess) {
		printf("Error from allocating device array is %s\n", cudaGetErrorString(error_id));
	}

  error_id = cudaMemcpy(d_a, h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);
	if (error_id != cudaSuccess) {
		printf("Error from copying over host array is %s\n", cudaGetErrorString(error_id));
	}

  // accessed data indices array on host (CPU)
	unsigned int *h_index = (unsigned int*) malloc(SHARED_LEN * sizeof(unsigned int));

  // accessed data indices array on device (GPU)
  unsigned int *d_index;
	error_id = cudaMalloc((void **) &d_index, sizeof(unsigned int));
	if (error_id != cudaSuccess) {
		printf("Error from allocating indices array is %s\n", cudaGetErrorString(error_id));
	}

  // data access latencies array on host (CPU)
	unsigned int *h_duration = (unsigned int*) malloc(SHARED_LEN * sizeof(unsigned int));

  // data access latencies array on device (GPU)
	unsigned int *d_duration;
	error_id = cudaMalloc ((void **) &d_duration, SHARED_LEN * sizeof(unsigned int));
	if (error_id != cudaSuccess) {
		printf("Error from allocating latencies array is %s\n", cudaGetErrorString(error_id));
	}

  // blocks until the device has completed all preceding requested tasks
	cudaThreadSynchronize ();

  // thread blocks 
	dim3 Db = dim3(32,1,1);
	dim3 Dg = dim3(1,1,1);

  // launch kernel
	global_latency<<<Dg, Db>>>(d_a, N, iterations, d_duration, d_index);
	cudaThreadSynchronize();

	error_id = cudaGetLastError();
  if (error_id != cudaSuccess) {
		printf("Error from kernel is %s\n", cudaGetErrorString(error_id));
	}
	cudaThreadSynchronize ();

  error_id = cudaMemcpy((void*) h_duration, (void*) d_duration, SHARED_LEN * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (error_id != cudaSuccess) {
    printf("Error 1 from copying from device is %s\n", cudaGetErrorString(error_id));
	}
  
  error_id = cudaMemcpy((void*) h_index, (void*) d_index, SHARED_LEN * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (error_id != cudaSuccess) {
		printf("Error 2 from copying from device to host is %s\n", cudaGetErrorString(error_id));
	}
	cudaThreadSynchronize();
	
	for(int i = 0; i < SHARED_LEN; i += stride) {		
			printf("%d\n", h_duration[i]);
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
	int iterations = 64; 
	int stride = 32;
  //N_min =24, N_max=60
	int N = 24 * 256;

	parametric_measure_global(N, iterations, stride);
}

int main(){
	cudaSetDevice(0); // current device 

	measure_global();

	cudaDeviceReset(); // destroy context
	return 0;
}

/*
   Finds: C, the cache size 
   For the Tesla V100-SXM2-16GB's texture L1 cache 
   Soure code based on paper https://arxiv.org/pdf/1509.02308.pdf 
*/ 

#include <stdio.h>
#include <stdlib.h>

// array size is 24576 * 4 / 1024 = 96 KB       
#define ARR_SIZE 8192

// number of iterations 
#define ITER 6144

//declare the texture
texture<int, 1, cudaReadModeElementType> tex_ref; 

__global__ void texture_latency (int* my_array, int size, unsigned int *duration, int *index, int iter) {
  // data access latencies array
  __shared__ unsigned int s_tvalue[ITER];
  // accessed data indices array
  __shared__ int s_value[ITER];

  // initialize arrays
  for (int i = 0; i < ITER; i++) {
    s_value[i] = -1; 
    s_tvalue[i] = 0; 
  }
  
  unsigned int start, end;
  int j = 0;
  for (int k = 0; k <= iter; k++) {
    for (int cnt = 0; cnt < ITER; cnt++) { 
      start = clock();

      // traverse an array whose elements are initialized as the indices for the next memory access
      j = tex1Dfetch(tex_ref, j);
      // handles ILP with this data dependency 
      s_value[cnt] = j;
      
      end = clock();			
      s_tvalue[cnt] = end - start;
    }
  }

  for (int i = 0; i < ITER; i++) {
	  duration[i] = s_tvalue[i];
	  index[i] = s_value[i];
	}

	my_array[size] = ITER;
	my_array[size+1] = s_tvalue[ITER-1];
}

void parametric_measure_texture(int N, int iterations, int stride) {
	cudaError_t error_id;
  
  // host (CPU) array
	int* h_a;

  int size =  N * sizeof(int);
  h_a = (int*) malloc(size); 
  
	for (int i = 0; i < (N-2); i++) {
		h_a[i] = (i + stride) % (N-2);
	}
	h_a[N-2] = 0;
	h_a[N-1] = 0;

  // device (GPU array)
  int* d_a;
 	cudaMalloc((void **) &d_a, size);
	cudaMemcpy((void*) d_a, (void*) h_a, size, cudaMemcpyHostToDevice);

  // accessed data indices array on host (CPU)
	int *h_index = (int*) malloc(ITER * sizeof(int));	
  // data access latencies array on host (CPU)
	unsigned int *h_duration = (unsigned int*) malloc(ITER * sizeof(unsigned int));

  // accessed data indices array on device (GPU)
	int* d_index;
	error_id = cudaMalloc(&d_index, ITER * sizeof(int));
	if (error_id != cudaSuccess) {
		printf("Error from allocating indices array is %s\n", cudaGetErrorString(error_id));
	}

  // data access latencies array on device (GPU)
	unsigned int *d_duration;
	error_id = cudaMalloc(&d_duration, ITER * sizeof(unsigned int));
	if (error_id != cudaSuccess) {
		printf("Error from allocating latencies array is %s\n", cudaGetErrorString(error_id));
	}

	// before a kernel can use a texture reference to read from texture memory, 
  // the texture reference must be bound to a texture 
	cudaBindTexture(0, tex_ref, d_a, size);
  
  // blocks until the device has completed all preceding requested tasks
	cudaThreadSynchronize();
  error_id = cudaGetLastError();
  if (error_id != cudaSuccess) {
		printf("Error from binding texture is %s\n", cudaGetErrorString(error_id));
	}

  // 1 x 1 block of threads
  dim3 Db = dim3(1);
  // 1 x 1 x 1 block of threads
  dim3 Dg = dim3(1,1,1);
  // launch kernel
  texture_latency<<<Dg, Db>>>(d_a, size, d_duration, d_index, iterations);

  cudaThreadSynchronize();
  error_id = cudaGetLastError();
  if (error_id != cudaSuccess) {
    printf("Error from launching kernel is %s\n", cudaGetErrorString(error_id));
  }

  cudaThreadSynchronize();

	// copy results from GPU to CPU 
	cudaMemcpy((void*) h_index, (void*) d_index, ITER * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*) h_duration, (void*) d_duration, ITER * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	// print the resulting benchmarks
	printf("\n=====Visting the %f KB array, loop %d*%d times======\n", (float)(N-2)*sizeof(int)/1024.0f, ITER, 1);
	for (int i = 0; i < ITER; i++){	
		printf("%10d\t %10f\n", h_index[i], (float) h_duration[i]);
	}

	// unbind texture
	cudaUnbindTexture(tex_ref);

	// free memory on GPU
	cudaFree(d_a);
	cudaFree(d_duration);
	cudaFree(d_index);
  cudaThreadSynchronize();
	
	// free memory on CPU
  free(h_a);
  free(h_duration);
	free(h_index);
}

int main() {
	cudaSetDevice(0); // current device 

    /*int count = 0;
    cudaError_t error_id = cudaGetDeviceCount(&count);
    if (error_id != cudaSuccess) {
        printf("%d %s\n", error_id, cudaGetErrorString(error_id));
    }
    printf("count %d\n", count);*/

  // repeatedly executed this amount of times
	int iterations = 10;
  // array sequentially traversed with this amount
  int stride = 1;
	
	for (int N = ARR_SIZE; N <= ARR_SIZE; N += stride) {
    // last 2 elements of array contain special values
		parametric_measure_texture(N+2, iterations, stride);
	}

	cudaDeviceReset(); // destroy context
	return 0;
}

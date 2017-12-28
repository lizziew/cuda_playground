/*
   Finds: C, the cache size 
   For the Tesla V100-SXM2-16GB's texture L1 cache 
   Soure code based on paper https://arxiv.org/pdf/1509.02308.pdf 
*/ 

#include <stdio.h>
#include <stdlib.h>

// number of elements of array (8192 * 4 / 1024 = 32 KB)     
#define ARR_LEN 7168
// number of elements of shared memory
#define SHARED_LEN 3584
// number of warmups 
#define WARMUP 10

//declare the texture
texture<int, 1, cudaReadModeElementType> tex_ref; 

__global__ void texture_latency (int* my_array, int size, unsigned int *duration, int *index) {
  // data access latencies array
  __shared__ unsigned int s_tvalue[SHARED_LEN];
  // accessed data indices array
  __shared__ int s_value[SHARED_LEN];

  // initialize arrays
  for (int i = 0; i < SHARED_LEN; i++) {
    s_value[i] = -1; 
    s_tvalue[i] = 0; 
  }
  
  int j = 0;
  unsigned int start, end;
  for (int k = 0; k <= WARMUP; k++) {
      for (int i = 0; i < ARR_LEN; i++) { 
        int shared_i = i % SHARED_LEN;

        start = clock();

        // traverse an array whose elements are initialized as the indices for the next memory access
        j = tex1Dfetch(tex_ref, j);
        // handles ILP with this data dependency 
        s_value[shared_i] = j;
        
        end = clock();			
        s_tvalue[shared_i] = end - start;
      }
  }

  for (int i = 0; i < SHARED_LEN; i++) {
	  duration[i] = s_tvalue[i];
	  index[i] = s_value[i];
	}
}

void parametric_measure_texture(int N, int stride) {
	cudaError_t error_id;
  
  // host (CPU) array
	int* h_a;

  int size =  N * sizeof(int);
  h_a = (int*) malloc(size); 
  
	for (int i = 0; i < N; i++) {
		h_a[i] = (i + stride) % N;
	}

  // device (GPU array)
  int* d_a;
 	cudaMalloc((void **) &d_a, size);
	cudaMemcpy((void*) d_a, (void*) h_a, size, cudaMemcpyHostToDevice);

  // accessed data indices array on host (CPU)
	int *h_index = (int*) malloc(SHARED_LEN * sizeof(int));	

  // data access latencies array on host (CPU)
	unsigned int *h_duration = (unsigned int*) malloc(SHARED_LEN * sizeof(unsigned int));

  // accessed data indices array on device (GPU)
	int* d_index;
	error_id = cudaMalloc(&d_index, SHARED_LEN * sizeof(int));
	if (error_id != cudaSuccess) {
		printf("Error from allocating indices array is %s\n", cudaGetErrorString(error_id));
	}

  // data access latencies array on device (GPU)
	unsigned int *d_duration;
	error_id = cudaMalloc(&d_duration, SHARED_LEN * sizeof(unsigned int));
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
  texture_latency<<<Dg, Db>>>(d_a, size, d_duration, d_index);

  cudaThreadSynchronize();
  error_id = cudaGetLastError();
  if (error_id != cudaSuccess) {
    printf("Error from launching kernel is %s\n", cudaGetErrorString(error_id));
  }

  cudaThreadSynchronize();

	// copy results from GPU to CPU 
	cudaMemcpy((void*) h_index, (void*) d_index, SHARED_LEN * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*) h_duration, (void*) d_duration, SHARED_LEN * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	// print the resulting benchmarks
	printf("\n=====Visting the %f KB array, loop %d*%d times======\n", (float)(N-2)*sizeof(int)/1024.0f, SHARED_LEN, 1);
	for (int i = 0; i < SHARED_LEN; i++){	
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

  /*
  // print number of devices   
  int count = 0;
  cudaError_t error_id = cudaGetDeviceCount(&count);
  if (error_id != cudaSuccess) {
      printf("%d %s\n", error_id, cudaGetErrorString(error_id));
  }
  printf("count %d\n", count);*/

  // array sequentially traversed with this amount
  int stride = 1;
	
	for (int N = ARR_LEN; N <= ARR_LEN; N += stride) {
		parametric_measure_texture(N, stride);
	}

	cudaDeviceReset(); // destroy context
	return 0;
}

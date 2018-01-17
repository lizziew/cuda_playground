/*
   Code based on paper "Demystifying GPU Microarchitecture through Microbenchmarking"
*/

#include <stdio.h> 
#include <stdint.h>

const int page_size = 4;  // Scale stride and arrays by page size.

__global__ void global_latency (unsigned int ** my_array, int array_length, int iterations, int ignore_iterations, unsigned long long * duration) {
  unsigned int start_time, end_time;
  unsigned int *j = (unsigned int*)my_array; 
  volatile unsigned long long sum_time;

  sum_time = 0;
  duration[0] = 0;

  for (int k = -ignore_iterations; k < iterations; k++) {
    if (k==0) {
      sum_time = 0; // ignore some iterations: cold icache misses
    }

    start_time = clock();
    repeat256(j=*(unsigned int **)j;)
    end_time = clock();

    sum_time += (end_time - start_time);
  }

  ((unsigned int*)my_array)[array_length] = (unsigned int)j;
  ((unsigned int*)my_array)[array_length+1] = (unsigned int) sum_time;
  duration[0] = sum_time;
}

/* Test page size. Construct an access pattern of N elements spaced stride apart,
   followed by a gap of stride+offset, followed by N more elements spaced stride
   apart. */
void measure_pagesize(int N, int stride, int offset) {
  // host (CPU) array
  unsigned int** h_a;
  // device (GPU) array
  unsigned int** d_a;

  unsigned long long* duration;
  unsigned long long* latency;

  unsigned long long latency_sum = 0;
  
  // total size of array
  const int size = N * stride * 2 + offset + stride*2;
  const int iterations = 20;

  // Don't die if too much memory was requested.
  if (size > 241600000) { 
  printf ("PREVENTION OF RUNNING OUT OF MEMORY.\n"); 
  return; 
  }

  // allocate array on CPU 
  h_a = (unsigned int**) malloc(4 * size);
  latency = (unsigned long long*) malloc(sizeof(unsigned long long));

  // allocate array on GPU
  cudaMalloc ((void**) &d_a, sizeof(unsigned int) * size);
  cudaMalloc ((void**) &duration, sizeof(unsigned long long));

  // initialize array elements on CPU 
  for (int i=0;i<N; i++) {
  ((unsigned int*) h_a)[i*stride] = ((i*stride + stride)*4) + (uintptr_t) d_a;
  }

  ((unsigned int *)h_a)[(N-1)*stride] = ((N*stride + offset)*4) + (uintptr_t) d_a;  //point last element to stride+offset

  for (int i=0;i<N; i++)
  ((unsigned int *)h_a)[(i+N)*stride+offset] = (((i+N)*stride + offset + stride)*4) + (uintptr_t) d_a;

  ((unsigned int *)h_a)[(2*N-1)*stride+offset] = (uintptr_t) d_a; //wrap around.

  cudaThreadSynchronize();;

  /* copy array elements from CPU to GPU */
  cudaMemcpy((void *)d_a, (void *)h_a, sizeof(unsigned int) * size, cudaMemcpyHostToDevice);
  cudaThreadSynchronize();

  for (int l = 0; l < 10 ; l++) {
  
  /* launch kernel*/
  dim3 Db = dim3(1);
  dim3 Dg = dim3(1,1,1);

  //printf("Launch kernel with parameters: %d, N: %d, stride: %d\n", iterations, N, stride); 
  global_latency<<<Dg, Db>>>(d_a, N, iterations, 1, duration);

  cudaThreadSynchronize();

  cudaError_t error_id = cudaGetLastError();
  if (error_id != cudaSuccess) {
    printf("Error is %s\n", cudaGetErrorString(error_id));
  }

  /* copy results from GPU to CPU */
  cudaThreadSynchronize();

  //cudaMemcpy((void *)h_a, (void *)d_a, sizeof(unsigned int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)latency, (void *)duration, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

  cudaThreadSynchronize();

  latency_sum += latency[0];
  }

  /* free memory on GPU */
  cudaFree(d_a);
  cudaFree(duration);
  cudaThreadSynchronize();

  /*free memory on CPU */
  free(h_a);
  free(latency);

  printf("%f\n", (double)(latency_sum/(10.0*256*iterations)));
}

void measure_global()
{
  printf ("\nGlobal4: Measuring L2 TLB page size using %d MB stride\n", 2 * page_size/4);
  printf ("  offset (bytes), latency (clocks)\n");
  
  // Small offsets (approx. page size) are interesting. Search much bigger offsets to
  // ensure nothing else interesting happens.
  for (int offset = -2048/4; offset <= (2097152+1536)/4; offset += (offset < 1536) ? 128/4 : 4096/4)
  {
  printf ("  %d, ", offset*4 *page_size/4);
  measure_pagesize(10, 2097152/4 *page_size/4, offset* page_size/4);
  }
}

int main() {
  printf("Assuming page size is %d KB\n", page_size);
  measure_global();
  return 0;
}

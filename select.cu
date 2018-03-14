// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <curand.h>

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_select.cuh>

#include "cub/test/test_util.h" // TODO look at this
#include "utils/gpu_utils.h"

using namespace std;
using namespace cub;

/**
 * Globals, constants and typedefs
 */
bool          g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

/// Selection functor type
struct LessThan
{
  float compare;

  __host__ __device__ __forceinline__
  LessThan(float compare) : compare(compare) {}

  __host__ __device__ __forceinline__
  bool operator()(const float &a) const {
    return (a < compare);
  }
};

__global__ // TODO device function methods 
void genFlagged(float* __restrict__ in, float sel, int* __restrict__ out, int numElements)
{
  int stride = blockDim.x * gridDim.x;
  int offset = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = offset; i < numElements; i += stride) {
  char flag = 0;
  if (in[i] < sel) flag = 1;

  out[i] = flag;
  }
}

float selectFlaggedGPU(float* d_in, float* d_val, float* d_out, int num_items, float cutpoint,
  int& num_selected, CachingDeviceAllocator&  g_allocator) {
  SETUP_TIMING();

  float time_gen_flagged, time_flagged;

  int* d_num_selected_out = NULL;
  int* d_flags = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_num_selected_out, sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_flags, sizeof(int) * num_items));  

  TIME_FUNC((genFlagged<<<8192, 256>>>(d_in, cutpoint, d_flags, num_items)), time_gen_flagged);

  // Allocate temporary storage
  void        *d_temp_storage = NULL;
  size_t      temp_storage_bytes = 0;

  CubDebugExit(DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Run
  TIME_FUNC(CubDebugExit(DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_val, d_flags, d_out, d_num_selected_out, num_items)), time_flagged);

  CubDebugExit(cudaMemcpy(&num_selected, d_num_selected_out, sizeof(int), cudaMemcpyDeviceToHost));

  CLEANUP(d_num_selected_out);
  CLEANUP(d_temp_storage);
  CLEANUP(d_flags);

  return time_gen_flagged + time_flagged;
}

float selectIfGPU(int d, float* d_in, float* d_out, int num_items, float cutpoint,
  int& num_selected, CachingDeviceAllocator&  g_allocator) {
  SETUP_TIMING();

  float time_if;

  int* d_num_selected_out = NULL;
  CubDebugExit(g_allocator.DeviceAllocate(d, (void**)&d_num_selected_out, sizeof(int)));

  // Allocate temporary storage
  void        *d_temp_storage = NULL;
  size_t      temp_storage_bytes = 0;
  LessThan select_op(cutpoint);

  // TODO DeviceSelect 
  CubDebugExit(DeviceSelect::If(d_temp_storage, temp_storage_bytes, 
		d_in, d_out, d_num_selected_out, num_items, select_op));
  CubDebugExit(g_allocator.DeviceAllocate(d, &d_temp_storage, temp_storage_bytes));

  // Run
  TIME_FUNC(CubDebugExit(DeviceSelect::If(d_temp_storage, temp_storage_bytes,
    d_in, d_out, d_num_selected_out, num_items, select_op)), time_if);

  CubDebugExit(cudaMemcpy(&num_selected, d_num_selected_out, sizeof(int), cudaMemcpyDeviceToHost));

	if (d_num_selected_out) CubDebugExit(g_allocator.DeviceFree(d, d_num_selected_out));
	if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d, d_temp_storage));

  return time_if;
}

/**
 * Main
 */
int main(int argc, char** argv)
{
  int half_num_items = 1<<27; // 1<<28 before
  int num_trials = 3;

  // Initialize command line
  CommandLineArgs args(argc, argv);
  args.GetCmdLineArgument("n", half_num_items);
  args.GetCmdLineArgument("t", num_trials);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    printf("%s "
      "[--n=<input items>] "
      "[--t=<num trials>] "
      "[--device=<device-id>] "
      "[--v] "
      "\n", argv[0]);
    exit(0);
  }

  // Initialize device
  CubDebugExit(args.DeviceInit(0));
	CubDebugExit(args.DeviceInit(1))

  // Allocate problem device arrays
  float **d_in = new float*[2];
	for (int d = 0; d < 2; d++) {
		CubDebugExit(g_allocator.DeviceAllocate(d, (void**)&d_in[d], sizeof(float) * half_num_items));
	}	

  float **d_val = new float*[2]; 
	for (int d = 0; d < 2; d++) {
  	CubDebugExit(g_allocator.DeviceAllocate(d, (void**)&d_val[d], sizeof(float) * half_num_items));
	}

  curandGenerator_t generator;
  int seed = 0;
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(generator,seed);
	for (int d = 0; d < 2; d++) {
		curandGenerateUniform(generator, d_in[d], half_num_items);
  	curandGenerateUniform(generator, d_val[d], half_num_items);
	}

	// Allocate device output array and num selected
  float **d_out = new float*[2];
  int   **d_num_selected_out = new int*[2];
  int   **d_flags = new int*[2];

	for (int d = 0; d < 2; d++) {
  	CubDebugExit(g_allocator.DeviceAllocate(d, (void**)&d_num_selected_out[d], sizeof(int)));
  	CubDebugExit(g_allocator.DeviceAllocate(d, (void**)&d_out[d], sizeof(float) * half_num_items));
  	CubDebugExit(g_allocator.DeviceAllocate(d, (void**)&d_flags[d], sizeof(int) * half_num_items));
	}
 
  for (int d = 0; d < 2; d++) { 
    for (int t = 0; t < num_trials; t++) {
      for (int i = 0; i <= 10; i++) {
        float selectivity = i/10.0;

        float time_flagged_gpu, time_if_gpu;
        int num_selected_flagged_gpu, num_selected_if_gpu;

				cout << "iteration " << d << " " << t << endl;

        time_if_gpu = selectIfGPU(d, d_in[d], d_out[d], half_num_items, selectivity, 
          num_selected_if_gpu, g_allocator);

        // time_flagged_gpu = selectFlaggedGPU(d_in[device], d_val[device], 
					// d_out[device], half_num_items, selectivity, num_selected_flagged_gpu, g_allocator);

        int s = num_selected_flagged_gpu;
        if (s != num_selected_if_gpu) {
          cout << "Answers don't match. " 
             << endl;
        }

        cout<< "{"
          << "\"selectivity\":" << selectivity
          << ",\"time_if_gpu\":" << time_if_gpu
          << ",\"time_flagged_gpu\":" << time_flagged_gpu
          << "}" << endl;
      }
    }
  }

  // Cleanup
	for (int d = 0; d < 2; d++) {
  	if (d_in) CubDebugExit(g_allocator.DeviceFree(d, d_in));
  	if (d_out) CubDebugExit(g_allocator.DeviceFree(d, d_out));
  	if (d_num_selected_out) CubDebugExit(g_allocator.DeviceFree(d, d_num_selected_out));
	}

  return 0;
}


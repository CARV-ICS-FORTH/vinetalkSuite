/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <assert.h>
#include <helper_cuda.h>
#include "../include/convolutionSeparable_common.h"
#include "VineLibUtilsGPU.h"
#include "../include/convolutionSeparableArgs.h"

float *d_Buffer;
vine_task_state_e hostCodeCPU(vine_task_msg_s *vine_task);
////////////////////////////////////////////////////////////////////////////////
// Convolution kernel storage
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_Kernel[KERNEL_LENGTH];

extern "C" void setConvolutionKernel(float *h_Kernel) {
  checkCudaErrors(
      cudaMemcpyToSymbol(c_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float)));
}

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
#define ROWS_BLOCKDIM_X 16
#define ROWS_BLOCKDIM_Y 4
#define ROWS_RESULT_STEPS 8
#define ROWS_HALO_STEPS 1

__global__ void convolutionRowsKernel(float *d_Dst, float *d_Src, int imageW,
                                      int imageH, int pitch) {
  __shared__ float
      s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) *
                              ROWS_BLOCKDIM_X];

  // Offset to the left halo edge
  const int baseX =
      (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X +
      threadIdx.x;
  const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

  d_Src += baseY * pitch + baseX;
  d_Dst += baseY * pitch + baseX;

// Load main data
#pragma unroll

  for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
    s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
        d_Src[i * ROWS_BLOCKDIM_X];
  }

// Load left halo
#pragma unroll

  for (int i = 0; i < ROWS_HALO_STEPS; i++) {
    s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
        (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
  }

// Load right halo
#pragma unroll

  for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS;
       i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++) {
    s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
        (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
  }

  // Compute and store results
  __syncthreads();
#pragma unroll

  for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
    float sum = 0;

#pragma unroll

    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
      sum += c_Kernel[KERNEL_RADIUS - j] *
             s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
    }

    d_Dst[i * ROWS_BLOCKDIM_X] = sum;
  }
}

extern "C" void convolutionRowsGPU(float *d_Dst, float *d_Src, int imageW,
                                   int imageH) {
  assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS);
  assert(imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
  assert(imageH % ROWS_BLOCKDIM_Y == 0);

  dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X),
              imageH / ROWS_BLOCKDIM_Y);
  dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

  convolutionRowsKernel << <blocks, threads>>>
      (d_Dst, d_Src, imageW, imageH, imageW);
  getLastCudaError("convolutionRowsKernel() execution failed\n\n");
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
#define COLUMNS_BLOCKDIM_X 16
#define COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 8
#define COLUMNS_HALO_STEPS 1

__global__ void convolutionColumnsKernel(float *d_Dst, float *d_Src, int imageW,
                                         int imageH, int pitch) {
  __shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS +
                                               2 * COLUMNS_HALO_STEPS) *
                                                  COLUMNS_BLOCKDIM_Y +
                                              1];

  // Offset to the upper halo edge
  const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
  const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) *
                        COLUMNS_BLOCKDIM_Y +
                    threadIdx.y;
  d_Src += baseY * pitch + baseX;
  d_Dst += baseY * pitch + baseX;

// Main data
#pragma unroll

  for (int i = COLUMNS_HALO_STEPS;
       i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
    s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
        d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
  }

// Upper halo
#pragma unroll

  for (int i = 0; i < COLUMNS_HALO_STEPS; i++) {
    s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
        (baseY >= -i * COLUMNS_BLOCKDIM_Y)
            ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch]
            : 0;
  }

// Lower halo
#pragma unroll

  for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS;
       i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS;
       i++) {
    s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
        (imageH - baseY > i * COLUMNS_BLOCKDIM_Y)
            ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch]
            : 0;
  }

  // Compute and store results
  __syncthreads();
#pragma unroll

  for (int i = COLUMNS_HALO_STEPS;
       i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
    float sum = 0;
#pragma unroll

    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
      sum += c_Kernel[KERNEL_RADIUS - j] *
             s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
    }

    d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
  }
}

extern "C" void convolutionColumnsGPU(float *d_Dst, float *d_Src, int imageW,
                                      int imageH) {
  assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS);
  assert(imageW % COLUMNS_BLOCKDIM_X == 0);
  assert(imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

  dim3 blocks(imageW / COLUMNS_BLOCKDIM_X,
              imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
  dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

  convolutionColumnsKernel << <blocks, threads>>>
      (d_Dst, d_Src, imageW, imageH, imageW);
  getLastCudaError("convolutionColumnsKernel() execution failed\n");
}

/* ****** Convolution function call ****** */
extern "C" void convolution(float *d_Output, float *d_Input, int imageW,
                            int imageH) {
  /* Call the first kernel */
  convolutionRowsGPU(d_Buffer, d_Input, imageW, imageH);
  /* Call the second kernel */
  convolutionColumnsGPU(d_Output, d_Buffer, imageW, imageH);
}

/* contains the code that is executed in Host*/
vine_task_state_e hostCode(vine_task_msg_s *vine_task) {
  std::chrono::time_point<std::chrono::system_clock> start, end;

  std::vector<void *> ioVector;

  cout << "Convolution execution in GPU." << endl;
  
  convolutionArgs *d_args_cuda;
  d_args_cuda = (convolutionArgs *) vine_data_deref(vine_task->args.vine_data);

#if (DEBUG_ENABLED)
  cout << "host2gpu" << endl;
#endif

  /* Allocate memory in the device and transfer data*/
  if (!Host2GPU(vine_task, ioVector)) {
    cerr << "Host2GPU" << endl;
    cerr << __FILE__ << " Failed at " << __LINE__ << endl;
    return (task_failed);
  }

#if (DEBUG_ENABLED)
  cout << "Set convolution kernel!!" << endl;
#endif

  float *h_Kernel;
  h_Kernel = (float *)ioVector[2];
  setConvolutionKernel(h_Kernel);

  d_Buffer = (float *)ioVector[1];

#if (DEBUG_ENABLED)
  cout << "Call the kernel" << endl;
#endif
#ifdef TIMERS_ENABLED
  start = std::chrono::system_clock::now();
#endif

  /*convolution()*/
  convolution((float *)ioVector[3], (float *)ioVector[0], d_args_cuda->imageW,
              d_args_cuda->imageH);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    cerr << __FILE__ << " Failed at " << __LINE__ << endl;
    printf("Error: %s\n", cudaGetErrorString(err));
    return (task_failed);
  }

#ifdef TIMERS_ENABLED

  end = std::chrono::system_clock::now();

  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::chrono::duration<double> elapsed_seconds = end - start;

  cout << "Convolution  GPU kernel execution time: " << elapsed_seconds.count()
       << " secs." << endl;
#endif

#if (DEBUG_ENABLED)
  cout << "GPU to Host" << endl;
#endif
  /* Get the result back from GPU*/
  if (!GPU2Host(vine_task, ioVector)) {
    cerr << "GPU2Host" << endl;
    cerr << __FILE__ << " Failed at " << __LINE__ << endl;
    return (task_failed);
  }
#if (DEBUG_ENABLED)
  cout << "Free" << endl;
#endif
  ioVector.erase(ioVector.begin() + 2);

  if (!GPUMemFree(ioVector)) {
    cerr << "GPUMemFree" << endl;
    cerr << __FILE__ << " Failed at " << __LINE__ << endl;
    return (task_failed);
  }
#if (DEBUG_ENABLED)
  cout << "Completed" << endl;
#endif
  return vine_task_stat(vine_task, 0);
}

VINE_PROC_LIST_START()
VINE_PROCEDURE("convolution", GPU, hostCode, sizeof(convolutionArgs))
VINE_PROCEDURE("convolution", CPU, hostCodeCPU, sizeof(convolutionArgs))
VINE_PROC_LIST_END()

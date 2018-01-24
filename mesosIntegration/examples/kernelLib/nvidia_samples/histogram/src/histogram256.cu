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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_cuda.h>
#include "VineLibUtilsGPU.h"
#include "../include/histogramArgs.h"
#include "../include/histogram_common.h"
uint partialHistogram256Countget = 240;

////////////////////////////////////////////////////////////////////////////////
// Shortcut shared memory atomic addition functions
////////////////////////////////////////////////////////////////////////////////
/* define hostCode for 64 bin version histogram in order to use it
 * in the array for this .so file (see at the end of this file)
*/
vine_task_state_e hostCode64(vine_task_msg_s *vine_task);
vine_task_state_e hostCode64CPU(vine_task_msg_s *vine_task);
vine_task_state_e hostCode256CPU(vine_task_msg_s *vine_task);
#define TAG_MASK 0xFFFFFFFFU
inline __device__ void addByte(uint *s_WarpHist, uint data, uint threadTag) {
  atomicAdd(s_WarpHist + data, 1);
}

inline __device__ void addWord(uint *s_WarpHist, uint data, uint tag) {
  addByte(s_WarpHist, (data >> 0) & 0xFFU, tag);
  addByte(s_WarpHist, (data >> 8) & 0xFFU, tag);
  addByte(s_WarpHist, (data >> 16) & 0xFFU, tag);
  addByte(s_WarpHist, (data >> 24) & 0xFFU, tag);
}

__global__ void histogram256Kernel(uint *d_PartialHistograms, uint *d_Data,
                                   uint dataCount) {
  // Per-warp subhistogram storage
  __shared__ uint s_Hist[HISTOGRAM256_THREADBLOCK_MEMORY];
  uint *s_WarpHist =
      s_Hist + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM256_BIN_COUNT;

// Clear shared memory storage for current threadblock before processing
#pragma unroll

  for (uint i = 0;
       i < (HISTOGRAM256_THREADBLOCK_MEMORY / HISTOGRAM256_THREADBLOCK_SIZE);
       i++) {
    s_Hist[threadIdx.x + i * HISTOGRAM256_THREADBLOCK_SIZE] = 0;
  }

  // Cycle through the entire data set, update subhistograms for each warp
  const uint tag = threadIdx.x << (UINT_BITS - LOG2_WARP_SIZE);

  __syncthreads();

  for (uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount;
       pos += UMUL(blockDim.x, gridDim.x)) {
    uint data = d_Data[pos];
    addWord(s_WarpHist, data, tag);
  }

  // Merge per-warp histograms into per-block and write to global memory
  __syncthreads();

  for (uint bin = threadIdx.x; bin < HISTOGRAM256_BIN_COUNT;
       bin += HISTOGRAM256_THREADBLOCK_SIZE) {
    uint sum = 0;

    for (uint i = 0; i < WARP_COUNT; i++) {
      sum += s_Hist[bin + i * HISTOGRAM256_BIN_COUNT] & TAG_MASK;
    }

    d_PartialHistograms[blockIdx.x * HISTOGRAM256_BIN_COUNT + bin] = sum;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Merge histogram256() output
// Run one threadblock per bin; each threadblock adds up the same bin counter
// from every partial histogram. Reads are uncoalesced, but mergeHistogram256
// takes only a fraction of total processing time
////////////////////////////////////////////////////////////////////////////////
#define MERGE_THREADBLOCK_SIZE 256

__global__ void mergeHistogram256Kernel(uint *d_Histogram,
                                        uint *d_PartialHistograms,
                                        uint histogramCount) {
  uint sum = 0;

  for (uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE) {
    sum += d_PartialHistograms[blockIdx.x + i * HISTOGRAM256_BIN_COUNT];
  }

  __shared__ uint data[MERGE_THREADBLOCK_SIZE];
  data[threadIdx.x] = sum;

  for (uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    __syncthreads();

    if (threadIdx.x < stride) {
      data[threadIdx.x] += data[threadIdx.x + stride];
    }
  }

  if (threadIdx.x == 0) {
    d_Histogram[blockIdx.x] = data[0];
  }
}

////////////////////////////////////////////////////////////////////////////////
// Host interface to GPU histogram
////////////////////////////////////////////////////////////////////////////////

extern "C" void histogram256(uint *d_Histogram,  // output
                             void *d_Data,       // input
                             uint byteCount) {
  assert(byteCount % sizeof(uint) == 0);
  histogram256Kernel
          << <partialHistogram256Countget, HISTOGRAM256_THREADBLOCK_SIZE>>>
      (d_PartialHistograms,  // output for kernel bellow
       (uint *)d_Data,       // input
       byteCount / sizeof(uint));
  getLastCudaError("histogram256Kernel() execution failed\n");

  mergeHistogram256Kernel << <HISTOGRAM256_BIN_COUNT, MERGE_THREADBLOCK_SIZE>>>
      (d_Histogram,          // output
       d_PartialHistograms,  // input from above kernel
       partialHistogram256Countget);
  getLastCudaError("mergeHistogram256Kernel() execution failed\n");
}

/* contains the code that is executed in Host*/
vine_task_state_e hostCode256(vine_task_msg_s *vine_task) {
  std::chrono::time_point<std::chrono::system_clock> start, end;

  std::vector<void *> ioVector;

  cout << "histogram 256 execution in GPU." << endl;

  histogramArgs *d_args_cuda;
  d_args_cuda = (histogramArgs *)vine_data_deref(vine_task->args.vine_data);

  /* Allocate memory in the device and transfer data*/
  if (!Host2GPU(vine_task, ioVector)) {
    cerr << "Host2GPU" << endl;
    cerr << __FILE__ << " Failed at " << __LINE__ << endl;
    return (task_failed);
  }
  d_PartialHistograms = (uint *)ioVector[1];

#ifdef TIMERS_ENABLED
  start = std::chrono::system_clock::now();
#endif

  histogram256((uint *)ioVector[2], (void *)ioVector[0],
               d_args_cuda->byteCount);

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

  cout << "Histogram 256 GPU kernel execution time: " << elapsed_seconds.count()
       << " secs." << endl;
#endif

  /* Get the result back from GPU*/
  if (!GPU2Host(vine_task, ioVector)) {
    cerr << "GPU2Host" << endl;
    cerr << __FILE__ << " Failed at " << __LINE__ << endl;
    return (task_failed);
  }

  if (!GPUMemFree(ioVector)) {
    cerr << "GPUMemFree" << endl;
    cerr << __FILE__ << " Failed at " << __LINE__ << endl;
    return (task_failed);
  }

  return vine_task_stat(vine_task, 0);
}

VINE_PROC_LIST_START()
VINE_PROCEDURE("histogram256", GPU, hostCode256, sizeof(histogramArgs))
VINE_PROCEDURE("histogram64", GPU, hostCode64, sizeof(histogramArgs))
VINE_PROCEDURE("histogram64", CPU, hostCode64CPU, sizeof(histogramArgs))
VINE_PROCEDURE("histogram256", CPU, hostCode256CPU, sizeof(histogramArgs))
VINE_PROC_LIST_END()

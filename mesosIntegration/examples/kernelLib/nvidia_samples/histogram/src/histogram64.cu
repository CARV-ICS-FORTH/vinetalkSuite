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
#include "histogram_common.h"
#include "VineLibUtilsGPU.h"
#include "histogramArgs.h"

////////////////////////////////////////////////////////////////////////////////
// GPU-specific common definitions
////////////////////////////////////////////////////////////////////////////////
// Data type used for input data fetches
typedef uint4 data_t;

// May change on future hardware, so better parametrize the code
#define SHARED_MEMORY_BANKS 16
uint maxPartialHist64Countget = 32768;
uint *d_PartialHistograms;
////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute gridDim.x partial histograms
////////////////////////////////////////////////////////////////////////////////
// Count a byte into shared-memory storage
inline __device__ void addByte(uchar *s_ThreadBase, uint data) {
  s_ThreadBase[UMUL(data, HISTOGRAM64_THREADBLOCK_SIZE)]++;
}

// Count four bytes of a word
inline __device__ void addWord(uchar *s_ThreadBase, uint data) {
  // Only higher 6 bits of each byte matter, as this is a 64-bin histogram
  addByte(s_ThreadBase, (data >> 2) & 0x3FU);
  addByte(s_ThreadBase, (data >> 10) & 0x3FU);
  addByte(s_ThreadBase, (data >> 18) & 0x3FU);
  addByte(s_ThreadBase, (data >> 26) & 0x3FU);
}

__global__ void histogram64Kernel(uint *d_PartialHistograms, data_t *d_Data,
                                  uint dataCount) {
  // Encode thread index in order to avoid bank conflicts in s_Hist[] access:
  // each group of SHARED_MEMORY_BANKS threads accesses consecutive shared
  // memory banks
  // and the same bytes [0..3] within the banks
  // Because of this permutation block size should be a multiple of 4 *
  // SHARED_MEMORY_BANKS
  const uint threadPos = ((threadIdx.x & ~(SHARED_MEMORY_BANKS * 4 - 1)) << 0) |
                         ((threadIdx.x & (SHARED_MEMORY_BANKS - 1)) << 2) |
                         ((threadIdx.x & (SHARED_MEMORY_BANKS * 3)) >> 4);

  // Per-thread histogram storage
  __shared__ uchar s_Hist[HISTOGRAM64_THREADBLOCK_SIZE * HISTOGRAM64_BIN_COUNT];
  uchar *s_ThreadBase = s_Hist + threadPos;

// Initialize shared memory (writing 32-bit words)
#pragma unroll

  for (uint i = 0; i < (HISTOGRAM64_BIN_COUNT / 4); i++) {
    ((uint *)s_Hist)[threadIdx.x + i * HISTOGRAM64_THREADBLOCK_SIZE] = 0;
  }

  // Read data from global memory and submit to the shared-memory histogram
  // Since histogram counters are byte-sized, every single thread can't do more
  // than 255 submission
  __syncthreads();

  for (uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount;
       pos += UMUL(blockDim.x, gridDim.x)) {
    data_t data = d_Data[pos];
    addWord(s_ThreadBase, data.x);
    addWord(s_ThreadBase, data.y);
    addWord(s_ThreadBase, data.z);
    addWord(s_ThreadBase, data.w);
  }

  // Accumulate per-thread histograms into per-block and write to global memory
  __syncthreads();

  if (threadIdx.x < HISTOGRAM64_BIN_COUNT) {
    uchar *s_HistBase =
        s_Hist + UMUL(threadIdx.x, HISTOGRAM64_THREADBLOCK_SIZE);

    uint sum = 0;
    uint pos = 4 * (threadIdx.x & (SHARED_MEMORY_BANKS - 1));

#pragma unroll

    for (uint i = 0; i < (HISTOGRAM64_THREADBLOCK_SIZE / 4); i++) {
      sum += s_HistBase[pos + 0] + s_HistBase[pos + 1] + s_HistBase[pos + 2] +
             s_HistBase[pos + 3];
      pos = (pos + 4) & (HISTOGRAM64_THREADBLOCK_SIZE - 1);
    }

    d_PartialHistograms[blockIdx.x * HISTOGRAM64_BIN_COUNT + threadIdx.x] = sum;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Merge histogram64() output
// Run one threadblock per bin; each threadbock adds up the same bin counter
// from every partial histogram. Reads are uncoalesced, but mergeHistogram64
// takes only a fraction of total processing time
////////////////////////////////////////////////////////////////////////////////
#define MERGE_THREADBLOCK_SIZE 256

__global__ void mergeHistogram64Kernel(uint *d_Histogram,
                                       uint *d_PartialHistograms,
                                       uint histogramCount) {
  __shared__ uint data[MERGE_THREADBLOCK_SIZE];

  uint sum = 0;

  for (uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE) {
    sum += d_PartialHistograms[blockIdx.x + i * HISTOGRAM64_BIN_COUNT];
  }

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
// CPU interface to GPU histogram calculator
////////////////////////////////////////////////////////////////////////////////
// histogram64kernel() intermediate results buffer
// MAX_PARTIAL_HISTOGRAM64_COUNT == 32768 and HISTOGRAM64_THREADBLOCK_SIZE == 64
// amounts to max. 480MB of input data
// static uint *d_PartialHistograms;
// static const uint MAX_PARTIAL_HISTOGRAM64_COUNT = 32768;
// static uint *d_PartialHistograms;

// Round a / b to nearest higher integer value
inline uint iDivUp(uint a, uint b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

// Snap a to nearest lower multiple of b
inline uint iSnapDown(uint a, uint b) { return a - a % b; }

/* ****** Histogram 64 function call ****** */
extern "C" void histogram64(uint *d_Histogram, void *d_Data, uint byteCount) {
  const uint histogramCount = iDivUp(
      byteCount, HISTOGRAM64_THREADBLOCK_SIZE * iSnapDown(255, sizeof(data_t)));

  assert(byteCount % sizeof(data_t) == 0);
  assert(histogramCount <= maxPartialHist64Countget);

  histogram64Kernel << <histogramCount, HISTOGRAM64_THREADBLOCK_SIZE>>>
      (d_PartialHistograms, (data_t *)d_Data, byteCount / sizeof(data_t));
  getLastCudaError("histogram64Kernel() execution failed\n");

  mergeHistogram64Kernel << <HISTOGRAM64_BIN_COUNT, MERGE_THREADBLOCK_SIZE>>>
      (d_Histogram, d_PartialHistograms, histogramCount);
  getLastCudaError("mergeHistogram64() execution failed\n");
}

/* contains the code that is executed in Host*/
vine_task_state_e hostCode64(vine_task_msg_s *vine_task) {
  std::chrono::time_point<std::chrono::system_clock> start, end;

  std::vector<void *> ioVector;

  printf("histogram 64 execution in GPU.\n");

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

  histogram64((uint *)ioVector[2], (void *)ioVector[0], d_args_cuda->byteCount);

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

  cout << "Histogram 64 GPU kernel execution time: " << elapsed_seconds.count()
       << " secs." << endl;
#endif

  /* Get the result back from GPU*/
  if (!GPU2Host(vine_task, ioVector)) {
    cerr << "GPU2Host" << ioVector[1] << endl;
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

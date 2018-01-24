/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

////////////////////////////////////////////////////////////////////////////////
// Global types
////////////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <stdio.h>
#include <helper_cuda.h>
#include <curand_kernel.h>

#include "MonteCarlo_common.h"
#include "vine_talk.h"
#include "monteCarloArgs.h"
#include "VineLibUtilsGPU.h"

////////////////////////////////////////////////////////////////////////////////
// Helper reduction template
// Please see the "reduction" CUDA Sample for more information
////////////////////////////////////////////////////////////////////////////////
#include "MonteCarlo_reduction.cuh"

////////////////////////////////////////////////////////////////////////////////
// Internal GPU-side data structures
////////////////////////////////////////////////////////////////////////////////
#define MAX_OPTIONS (1024 * 1024)

////////////////////////////////////////////////////////////////////////////////
// Overloaded shortcut payoff functions for different precision modes
////////////////////////////////////////////////////////////////////////////////
__device__ inline float endCallValue(float S, float X, float r, float MuByT,
                                     float VBySqrtT) {
  float callValue = S * __expf(MuByT + VBySqrtT * r) - X;
  return (callValue > 0.0F) ? callValue : 0.0F;
}

__device__ inline double endCallValue(double S, double X, double r,
                                      double MuByT, double VBySqrtT) {
  double callValue = S * exp(MuByT + VBySqrtT * r) - X;
  return (callValue > 0.0) ? callValue : 0.0;
}

////////////////////////////////////////////////////////////////////////////////
// This kernel computes the integral over all paths using a single thread block
// per option. It is fastest when the number of thread blocks times the work per
// block is high enough to keep the GPU busy.
////////////////////////////////////////////////////////////////////////////////
static __global__ void MonteCarloOneBlockPerOption(curandState *__restrict rngStates,
                                                   const __TOptionData *__restrict d_OptionData,
                                                   __TOptionValue *__restrict d_CallValue, 
                                                   int pathN, 
                                                   int optionN) {
  const int SUM_N = THREAD_N;
  __shared__ real s_SumCall[SUM_N];
  __shared__ real s_Sum2Call[SUM_N];

  // determine global thread id
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Copy random number state to local memory for efficiency
  curandState localState = rngStates[tid];
  for (int optionIndex = blockIdx.x; optionIndex < optionN; optionIndex += gridDim.x) {
    const real S = d_OptionData[optionIndex].S;
    const real X = d_OptionData[optionIndex].X;
    const real MuByT = d_OptionData[optionIndex].MuByT;
    const real VBySqrtT = d_OptionData[optionIndex].VBySqrtT;

    // Cycle through the entire samples array:
    // derive end stock price for each path
    // accumulate partial integrals into intermediate shared memory buffer
    for (int iSum = threadIdx.x; iSum < SUM_N; iSum += blockDim.x) {
      __TOptionValue sumCall = {0, 0};

#pragma unroll 8
      for (int i = iSum; i < pathN; i += SUM_N) {
        real r = curand_normal(&localState);
        real callValue = endCallValue(S, X, r, MuByT, VBySqrtT);
        sumCall.Expected += callValue;
        sumCall.Confidence += callValue * callValue;
      }

      s_SumCall[iSum] = sumCall.Expected;
      s_Sum2Call[iSum] = sumCall.Confidence;
    }

    // Reduce shared memory accumulators
    // and write final result to global memory
    sumReduce<real, SUM_N, THREAD_N>(s_SumCall, s_Sum2Call);

    if (threadIdx.x == 0) {
      __TOptionValue t = {s_SumCall[0], s_Sum2Call[0]};
      d_CallValue[optionIndex] = t;
    }
  }
}

static __global__ void rngSetupStates(curandState *rngState, int device_id) {
  // determine global thread id
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // Each threadblock gets different seed,
  // Threads within a threadblock get different sequence numbers
  curand_init(blockIdx.x + gridDim.x, threadIdx.x, 0, &rngState[tid]);
}

////////////////////////////////////////////////////////////////////////////////
// Host-side interface to GPU Monte Carlo
////////////////////////////////////////////////////////////////////////////////
extern "C" void initMonteCarloGPU(TOptionPlan *plan) {
  checkCudaErrors(cudaMalloc(&plan->d_OptionData,
                             sizeof(__TOptionData) * (plan->optionCount)));
  checkCudaErrors(cudaMalloc(&plan->d_CallValue,
                             sizeof(__TOptionValue) * (plan->optionCount)));
  checkCudaErrors(cudaMallocHost(&plan->h_OptionData,
                                 sizeof(__TOptionData) * (plan->optionCount)));
  // Allocate internal device memory
  checkCudaErrors(cudaMallocHost(&plan->h_CallValue,
                                 sizeof(__TOptionValue) * (plan->optionCount)));
  // Allocate states for pseudo random number generators
  checkCudaErrors(cudaMalloc((void **)&plan->rngStates,
                             plan->gridSize * THREAD_N * sizeof(curandState)));

  // place each device pathN random numbers apart on the random number sequence
  rngSetupStates << <plan->gridSize, THREAD_N>>>(plan->rngStates, plan->device);
  getLastCudaError("rngSetupStates kernel failed.\n");
}

// Compute statistics and deallocate internal device memory
extern "C" void closeMonteCarloGPU(TOptionPlan *plan) {
  for (int i = 0; i < plan->optionCount; i++) {
    const double RT = plan->optionData[i].R * plan->optionData[i].T;
    const double sum = plan->h_CallValue[i].Expected;
    const double sum2 = plan->h_CallValue[i].Confidence;
    const double pathN = plan->pathN;
    // Derive average from the total sum and discount by riskfree rate
    plan->callValue[i].Expected = (float)(exp(-RT) * sum / pathN);
    // Standart deviation
    double stdDev = sqrt((pathN * sum2 - sum * sum) / (pathN * (pathN - 1)));
    // Confidence width; in 95% of all cases theoretical value lies within these
    // borders
    plan->callValue[i].Confidence =
        (float)(exp(-RT) * 1.96 * stdDev / sqrt(pathN));
  }

  checkCudaErrors(cudaFree(plan->rngStates));
  checkCudaErrors(cudaFreeHost(plan->h_CallValue));
  checkCudaErrors(cudaFreeHost(plan->h_OptionData));
  checkCudaErrors(cudaFree(plan->d_CallValue));
  checkCudaErrors(cudaFree(plan->d_OptionData));
}

// Main computations
extern "C" void MonteCarloGPU(TOptionPlan *plan, cudaStream_t stream) {
  __TOptionValue *h_CallValue = plan->h_CallValue;

  if (plan->optionCount <= 0 || plan->optionCount > MAX_OPTIONS) {
    printf("MonteCarloGPU(): bad option count.\n");
    return;
  }

  __TOptionData *h_OptionData = (__TOptionData *)plan->h_OptionData;

  for (int i = 0; i < plan->optionCount; i++) {
    const double T = plan->optionData[i].T;
    const double R = plan->optionData[i].R;
    const double V = plan->optionData[i].V;
    const double MuByT = (R - 0.5 * V * V) * T;
    const double VBySqrtT = V * sqrt(T);
    h_OptionData[i].S = (real)plan->optionData[i].S;
    h_OptionData[i].X = (real)plan->optionData[i].X;
    h_OptionData[i].MuByT = (real)MuByT;
    h_OptionData[i].VBySqrtT = (real)VBySqrtT;
  }

  checkCudaErrors(cudaMemcpyAsync(plan->d_OptionData, h_OptionData,
                                  plan->optionCount * sizeof(__TOptionData),
                                  cudaMemcpyHostToDevice, stream));

  MonteCarloOneBlockPerOption <<<plan->gridSize, THREAD_N, 0, stream>>>
      (plan->rngStates, (__TOptionData *)(plan->d_OptionData),
       (__TOptionValue *)(plan->d_CallValue), plan->pathN, plan->optionCount);
  getLastCudaError("MonteCarloOneBlockPerOption() execution failed\n");

  checkCudaErrors(cudaMemcpyAsync(h_CallValue, plan->d_CallValue,
                                  plan->optionCount * sizeof(__TOptionValue),
                                  cudaMemcpyDeviceToHost, stream));

  // cudaDeviceSynchronize();
}

// Wrapper for native job generator. 

extern "C" void cu_monteCarlo(curandState *rngStates, 
                              __TOptionData * d_OptionData, 
                              __TOptionValue *d_CallValue, 
							  int optionCount, 
							  int pathN,
							  int gridSize,
							  int device) {

    if (optionCount <= 0 || optionCount > MAX_OPTIONS) {
        printf("MonteCarloGPU(): bad option count.\n");
        return;
    }

	// place each device pathN random numbers apart on the random number sequence
    rngSetupStates<<<gridSize, THREAD_N>>>(rngStates, device);
    getLastCudaError("rngSetupStates kernel failed.\n");


    MonteCarloOneBlockPerOption<<<gridSize, THREAD_N, 0, 0>>>(rngStates,
															d_OptionData,
															d_CallValue,
															pathN,
															optionCount);
    getLastCudaError("MonteCarloOneBlockPerOption() execution failed\n");

}

//Main computations

/**
 * Contains the code that is being executed in the GPU accelerator.
 * It takes as a parameter a vine_task descriptor and returns
 * the status of the task after its execution.
 */
vine_task_state_e hostCodeGPU(vine_task_msg_s *vine_task) {
  std::vector<void *> ioVector;
  monteCarloArgs *d_args_cuda;

  d_args_cuda = (monteCarloArgs *)vine_data_deref(vine_task->args.vine_data);

  // Allocate memory in the device and transfer data.
  if (!Host2GPU(vine_task, ioVector)) {
    cerr << "Host2GPU" << endl;
    cerr << __FILE__ << " Failed at " << __LINE__ << endl;
    return (task_failed);
  }

  cout << "MonteCarlo execution in GPU." << endl;

#ifdef TIMERS_ENABLED
  std::chrono::time_point<std::chrono::system_clock> start, end;

  start = std::chrono::system_clock::now();
#endif

  // place each device pathN random numbers apart on the random number sequence
  rngSetupStates <<<d_args_cuda->gridSize, THREAD_N>>>((curandState *)ioVector[0], 0);
  getLastCudaError("rngSetupStates kernel failed.\n");

  MonteCarloOneBlockPerOption <<<d_args_cuda->gridSize, THREAD_N, 0, 0>>>
      ((curandState *)ioVector[0],     // rngStates,
       (__TOptionData *)ioVector[1],   // (__TOptionData *)(d_OptionData),
       (__TOptionValue *)ioVector[2],  // (__TOptionValue *)(d_CallValue),
       d_args_cuda->pathN, 
       d_args_cuda->optionCount);
  getLastCudaError("MonteCarloOneBlockPerOption() execution failed\n");
#ifdef TIMERS_ENABLED

  end = std::chrono::system_clock::now();

  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::chrono::duration<double> elapsed_seconds = end - start;

  cout << "MonteCarlo GPU kernel execution time: " << elapsed_seconds.count()
       << " secs." << endl;
#endif

  // Cuda error check.
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    cerr << __FILE__ << " Failed at " << __LINE__ << endl;
    printf("Error: %s\n", cudaGetErrorString(err));
    return (task_failed);
  }
  unsigned long long duration;
  // Memory transfer from device memory to shared segment.
  if (! (duration=GPU2Host(vine_task, ioVector)) ){
    cout << "GPU2Host" << endl;
    cerr << __FILE__ << " Failed at " << __LINE__ << endl;
    return (task_failed);
  }
  // Free allocated memory.
  if (!GPUMemFree(ioVector)) {
    cout << "GPUMemFree" << endl;
    cerr << __FILE__ << " Failed at " << __LINE__ << endl;
    return (task_failed);
  }

  //return vine_task_stat(vine_task, 0);
  return  (vine_task_state_e)duration;
}

/*
 * Mapping of function "MonteCarloGPU" with "hostCodeGPU" that executes a
 * blackScholes task.
 */
VINE_PROC_LIST_START()
VINE_PROCEDURE("MonteCarlo", GPU, hostCodeGPU, sizeof(monteCarloArgs))
VINE_PROC_LIST_END()

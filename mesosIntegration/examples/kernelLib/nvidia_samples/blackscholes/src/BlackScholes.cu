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

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
#include <helper_functions.h>  // helper functions for string parsing
#include <helper_cuda.h>  // helper functions CUDA error checking and initialization

#include "blackScholesArgs.h"
#include "BlackScholes_kernel.cuh"
#include "VineLibUtilsGPU.h"

// const int OPT_N = 4000000;

///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
__device__ inline float cndGPU(float d) {
  const float A1 = 0.31938153f;
  const float A2 = -0.356563782f;
  const float A3 = 1.781477937f;
  const float A4 = -1.821255978f;
  const float A5 = 1.330274429f;
  const float RSQRT2PI = 0.39894228040143267793994605993438f;

  float K = __fdividef(1.0f, (1.0f + 0.2316419f * fabsf(d)));

  float cnd = RSQRT2PI * __expf(-0.5f * d * d) *
              (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

  if (d > 0) cnd = 1.0f - cnd;

  return cnd;
}

///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
__device__ inline void BlackScholesBodyGPU(float &CallResult, float &PutResult,
                                           float S,  // Stock price
                                           float X,  // Option strike
                                           float T,  // Option years
                                           float R,  // Riskless rate
                                           float V   // Volatility rate
                                           ) {
  float sqrtT, expRT;
  float d1, d2, CNDD1, CNDD2;

  sqrtT = __fdividef(1.0F, rsqrtf(T));
  d1 = __fdividef(__logf(S / X) + (R + 0.5f * V * V) * T, V * sqrtT);
  d2 = d1 - V * sqrtT;

  CNDD1 = cndGPU(d1);
  CNDD2 = cndGPU(d2);

  // Calculate Call and Put simultaneously
  expRT = __expf(-R * T);
  CallResult = S * CNDD1 - X * expRT * CNDD2;
  PutResult = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
__launch_bounds__(128) __global__
    void BlackScholesGPU(float2 *__restrict d_CallResult,
                         float2 *__restrict d_PutResult,
                         float2 *__restrict d_StockPrice,
                         float2 *__restrict d_OptionStrike,
                         float2 *__restrict d_OptionYears, float Riskfree,
                         float Volatility, int optN) {
  ////Thread index
  // const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
  ////Total number of threads in execution grid
  // const int THREAD_N = blockDim.x * gridDim.x;

  const int opt = blockDim.x * blockIdx.x + threadIdx.x;

  // Calculating 2 options per thread to increase ILP (instruction level
  // parallelism)
  if (opt < (optN / 2)) {
    float callResult1, callResult2;
    float putResult1, putResult2;
    BlackScholesBodyGPU(callResult1, putResult1, d_StockPrice[opt].x,
                        d_OptionStrike[opt].x, d_OptionYears[opt].x, Riskfree,
                        Volatility);
    BlackScholesBodyGPU(callResult2, putResult2, d_StockPrice[opt].y,
                        d_OptionStrike[opt].y, d_OptionYears[opt].y, Riskfree,
                        Volatility);
    d_CallResult[opt] = make_float2(callResult1, callResult2);
    d_PutResult[opt] = make_float2(putResult1, putResult2);
  }
}

/* Wrapper for job generator */
void cu_BlackScholesGPU(float *d_CallResult, float *d_PutResult,
                        float *d_StockPrice, float *d_OptionStrike,
                        float *d_OptionYears, float Riskfree, float Volatility,
                        int optN) {
  BlackScholesGPU << <DIV_UP((optN / 2), 128), 128 /*480, 128*/>>>
      ((float2 *)d_CallResult,    // OUT
       (float2 *)d_PutResult,     // OUT
       (float2 *)d_StockPrice,    // IN
       (float2 *)d_OptionStrike,  // IN
       (float2 *)d_OptionYears,   // IN
       Riskfree, Volatility, optN);
}

/**
 * Contains the code that is being executed in the GPU accelerator.
 * It takes as a parameter a vine_task descriptor and returns
 * the status of the task after its execution.
 */
vine_task_state_e hostCode(vine_task_msg_s *vine_task) {
  std::vector<void *> ioVector;
  blackScholesArgs *d_args_cuda;
  std::chrono::time_point<std::chrono::system_clock> start, end;

  d_args_cuda = (blackScholesArgs *)vine_data_deref(vine_task->args.vine_data);

  // Allocate memory in the device and transfer data.
  if (!Host2GPU(vine_task, ioVector)) {
    cerr << "Host2GPU" << endl;
    cerr << __FILE__ << " Failed at " << __LINE__ << endl;
    return (task_failed);
  }

  cout << "BlackScholes execution in GPU." << endl;

#ifdef TIMERS_ENABLED
  start = std::chrono::system_clock::now();
#endif

  // The actual BlackScholes kernel call.
  BlackScholesGPU << <DIV_UP((d_args_cuda->optN / 2), 128), 128 /*480, 128*/>>>
      ((float2 *)ioVector[3],  // OUT
       (float2 *)ioVector[4],  // OUT
       (float2 *)ioVector[0],  // IN
       (float2 *)ioVector[1],  // IN
       (float2 *)ioVector[2],  // IN
       d_args_cuda->Riskfree, d_args_cuda->Volatility, d_args_cuda->optN);
#ifdef TIMERS_ENABLED

  end = std::chrono::system_clock::now();

  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::chrono::duration<double> elapsed_seconds = end - start;

  cout << "BlackScholesGPU kernel execution time: " << elapsed_seconds.count()
       << " secs." << endl;
#endif

  // Cuda error check.
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    cerr << __FILE__ << " Failed at " << __LINE__ << endl;
    printf("Error: %s\n", cudaGetErrorString(err));
    return (task_failed);
  }

  // Memory transfer from device memory to shared segment.
  if (!GPU2Host(vine_task, ioVector)) {
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

  return vine_task_stat(vine_task, 0);
}

/*
 * Mapping of function "BlackScholesCPU" with "hostCodeCPU" that executes a
 * blackScholes task.
 */
VINE_PROC_LIST_START()
VINE_PROCEDURE("BlackScholes", GPU, hostCode, sizeof(blackScholesArgs))
VINE_PROC_LIST_END()

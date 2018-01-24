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

/*
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>  // helper functions for string parsing
#include <helper_cuda.h>  // helper functions CUDA error checking and initialization
#include <chrono>
/* Custom headers */
#include "statisticsDefineEnable.h"
#include "vine_talk.h"
#include "blackScholesArgs.h"
#include "vineFunCaller.hpp"
#define DEBUG_ENABLED (0)


using std::cout;
using std::endl;
using std::cerr;

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high) {
  float t = (float)rand() / (float)RAND_MAX;
  return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int OPT_N = 4000000;
const int NUM_ITERATIONS = 1000;

const int OPT_SZ = OPT_N * sizeof(float);
const float RISKFREE = 0.02f;
const float VOLATILITY = 0.30f;

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
//int main(int argc, char **argv) {
float* vineFunCallerInterface::vine_blackScholes(){
  printf("BlackScholes through Vinetalk - Starting... GELLLOOOOOO\n");
  vine_talk_init();

  // Start logs
#if (DEBUG_ENABLED)
  printf("BlackScholes through Vinetalk - Starting... GELLLOOOOOO\n");
  /*
  if (argc != 2) {
    printf("Usage: [%s] <type>\n\twhere type is 1 for GPU, 3 for CPU.\n",
           argv[0]);
    return 0;
  }*/
#endif
  StopWatchInterface *hTimer = NULL;
  //'h_' prefix - CPU (host) memory space
  float *h_CallResultCPU,
      *h_PutResultCPU;  // Results calculated by CPU for reference
  float *h_CallResultGPU, *h_PutResultGPU;  // CPU copy of GPU results
  float *h_StockPrice, *h_OptionStrike,
      *h_OptionYears;  // CPU instance of input data
  int i;

  /**************************************************************************/
  // vine_talk additions.
  vine_accel *accel;  // The accelerator to use

  // Get type of accelerator from command line argument
  vine_accel_type_e type = (vine_accel_type_e)1;//atoi(argv[1]);

  // Choose an accelerator randomly.
  accel = vine_accel_acquire_type(type);

  // Allocate space in shared mem segment (only in CPU) for arguments.

  // Create a pointer to the vine argument struct.
  blackScholesArgs *arguments =
      (blackScholesArgs *)malloc(sizeof(blackScholesArgs));

  // Pass the appropriate data to the struct.
  arguments->Riskfree = RISKFREE;
  arguments->Volatility = VOLATILITY;
  arguments->optN = OPT_N;
  vine_buffer_s vine_args[1] = {
      VINE_BUFFER(arguments, sizeof(blackScholesArgs))};

  // End of vine_talk additions.
  /**************************************************************************/

  sdkCreateTimer(&hTimer);
#if (DEBUG_ENABLED)
  printf("Initializing data...\n");
  printf("...allocating CPU memory for options.\n");
#endif
  h_CallResultCPU = (float *)malloc(OPT_SZ);
  h_PutResultCPU = (float *)malloc(OPT_SZ);
  h_CallResultGPU = (float *)malloc(OPT_SZ);
  h_PutResultGPU = (float *)malloc(OPT_SZ);
  h_StockPrice = (float *)malloc(OPT_SZ);
  h_OptionStrike = (float *)malloc(OPT_SZ);
  h_OptionYears = (float *)malloc(OPT_SZ);
#if (DEBUG_ENABLED)
  printf("...generating input data in CPU mem.\n");
#endif
#ifdef STATISTICS_TIMERS_ENABLED
  /*timer for input creation*/
  std::chrono::time_point<std::chrono::system_clock> startCreating, endCreating,
      startMemCpySHM, stopMemCpySHM;

  startCreating = std::chrono::system_clock::now();
#endif

  srand(5347);

  // Generate options set
  for (i = 0; i < OPT_N; i++) {
    h_CallResultCPU[i] = 0.0f;
    h_PutResultCPU[i] = -1.0f;
    h_StockPrice[i] = RandFloat(5.0f, 30.0f);
    h_OptionStrike[i] = RandFloat(1.0f, 100.0f);
    h_OptionYears[i] = RandFloat(0.25f, 10.0f);
  }
#ifdef STATISTICS_TIMERS_ENABLED
  endCreating = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = endCreating - startCreating;
  cout << endl;
  cout << "--------------------Breakdown Stats----------------------" << endl;
  cout << "Creating  Inputs with size: " << OPT_N
       << " took: " << elapsed.count() << " sec." << endl;
#endif
float totalGpuTime=0;
float* gpuTimesArray;
gpuTimesArray=(float*)malloc(NUM_ITERATIONS*sizeof(float));

  // If the task is going to be issued for GPU, execute the following code.
  if (type == GPU) {
    //'d_' prefix - GPU (device) memory space
    double delta, ref, sum_delta, sum_ref, max_delta, L1norm, gpuTime;
    // Create a task id.
    vine_proc *process_id = vine_proc_get(type, "BlackScholes");

    // vine_talk additions.

    // Pointers to the shared memory segment containing input data.
    vine_buffer_s inputs[3] = {VINE_BUFFER(h_StockPrice, OPT_SZ),
                               VINE_BUFFER(h_OptionStrike, OPT_SZ),
                               VINE_BUFFER(h_OptionYears, OPT_SZ)};

    vine_buffer_s outputs[2] = {VINE_BUFFER(h_CallResultGPU, OPT_SZ),
                                VINE_BUFFER(h_PutResultGPU, OPT_SZ)};

    /**********************************************************************/


    for(int index=0; index<NUM_ITERATIONS;index++)
    {
      sdkResetTimer(&hTimer);
      sdkStartTimer(&hTimer);
        // Issue task to accelerator.
        vine_task *task =
            vine_task_issue(accel, process_id, vine_args, 3, inputs, 2, outputs);

        // Wait for task or exit if it fails.11
        if (vine_task_wait(task) != task_completed) {
          cerr << "Task failed!" << endl;
          return NULL;
        }
        vine_task_free(task);

        sdkStopTimer(&hTimer);
        gpuTimesArray[index]=sdkGetTimerValue(&hTimer);
        totalGpuTime+=gpuTimesArray[index];
    }

    gpuTime = totalGpuTime / NUM_ITERATIONS;
#if (DEBUG_ENABLED)
    // Both call and put is calculated
    printf("Options count             : %i     \n", 2 * OPT_N);
    printf("BlackScholesGPU() time    : %f msec\n", gpuTime);
    printf("Effective memory bandwidth: %f GB/s\n",
           ((double)(5 * OPT_N * sizeof(float)) * 1E-9) / (gpuTime * 1E-3));
    printf("Gigaoptions per second    : %f     \n\n",
           ((double)(2 * OPT_N) * 1E-9) / (gpuTime * 1E-3));

    printf(
        "BlackScholes, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u "
        "options, NumDevsUsed = %u, Workgroup = %u\n",
        (((double)(2.0 * OPT_N) * 1.0E-9) / (gpuTime * 1.0E-3)), gpuTime * 1e-3,
        (2 * OPT_N), 1, 128);
    printf("Shutting down...\n");
#endif
  }
  // If the task is going to be issued for CPU, execute the following code.
  else if (type == CPU) {
#if (DEBUG_ENABLED)
    printf("BlackScholes in CPU .\n\n");
#endif
    // Calculate options values on CPU

    /**********************************************************************/
    // vine_talk additions.
    // Create a task id.
    vine_proc *process_id = vine_proc_get(type, "BlackScholes");

    // Pointers to the shared memory segment containing input data.
    vine_buffer_s inputs[3] = {VINE_BUFFER(h_StockPrice, OPT_SZ),
                               VINE_BUFFER(h_OptionStrike, OPT_SZ),
                               VINE_BUFFER(h_OptionYears, OPT_SZ)};

    vine_buffer_s outputs[2] = {
        VINE_BUFFER(h_CallResultCPU, OPT_SZ),
        VINE_BUFFER(h_PutResultCPU, OPT_SZ),
    };

    // Issue task to accelerator.
    vine_task *task =
        vine_task_issue(accel, process_id, vine_args, 3, inputs, 2, outputs);

    // Wait for task or exit if it fails.
    if (vine_task_wait(task) != task_completed) {
      cerr << "Task failed!" << endl;
      return NULL;
    }
    vine_task_free(task);
  }
#if (DEBUG_ENABLED)
  printf("...releasing CPU memory.\n");
#endif
  free(h_OptionYears);
  free(h_OptionStrike);
  free(h_StockPrice);
  free(h_PutResultGPU);
  free(h_CallResultGPU);
  free(h_PutResultCPU);
  free(h_CallResultCPU);
  sdkDeleteTimer(&hTimer);
  /**************************************************************************/
  vine_accel_release(&accel);
  /**************************************************************************/

  printf("Shutdown done.\n");

  printf("\n[BlackScholes] - Test Summary \n");
#if (DEBUG_ENABLED)
  printf(
      "\nNOTE: The CUDA Samples are not meant for performance measurements. "
      "Results may vary when GPU Boost is enabled.\n\n");
#endif
  printf("Test passed\n");
  vine_talk_exit();
  //exit(EXIT_SUCCESS);
  return gpuTimesArray;
}

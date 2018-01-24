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

/*
 * This sample evaluates fair call price for a
 * given set of European options using Monte Carlo approach.
 * See supplied whitepaper for more explanations.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>

// includes, project
#include <helper_functions.h>  // Helper functions (utilities, parsing, timing)
#include <helper_cuda.h>  // helper functions (cuda error checking and intialization)
#include <multithreading.h>

#include "MonteCarlo_common.h"
#include "vine_talk.h"
#include "vine_pipe.h"
#include "monteCarloArgs.h"
#include "statisticsDefineEnable.h"
#include "vineFunCaller.hpp"

using std::cout;
using std::endl;
using std::cerr;

int *pArgc = NULL;
char **pArgv = NULL;
int NUM_ITERATIONS=1000;
bool Host2GPU(vine_task_msg_s *vine_task, std::vector<void *> &ioHD)
{
    cout<<__FILE__<<__func__<<endl;
    return false;
}

/* Cuda Memcpy from Device to host*/
bool GPU2Host(vine_task_msg_s *vine_task, std::vector<void *> &ioDH)
{
    cout<<__FILE__<<__func__<<endl;
    return false;
}

/* Free Device memory */
bool GPUMemFree(std::vector<void *> &io)
{
    cout<<__FILE__<<__func__<<endl;
    return false;

}



#ifdef WIN32
#define strcasecmp _strcmpi
#endif
# define VERIFICATION_ENABLED (0)
# define DEBUG (0)
////////////////////////////////////////////////////////////////////////////////
// Common functions
////////////////////////////////////////////////////////////////////////////////
float randFloat(float low, float high) {
  float t = (float)rand() / (float)RAND_MAX;
  return (1.0f - t) * low + t * high;
}

/// Utility function to tweak problem size for small GPUs
int adjustProblemSize(int GPU_N, int default_nOptions) {
  int nOptions = default_nOptions;

  // select problem size
  for (int i = 0; i < GPU_N; i++) {
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, i));
    int cudaCores = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
                    deviceProp.multiProcessorCount;

    if (cudaCores <= 32) {
      nOptions = (nOptions < cudaCores / 2 ? nOptions : cudaCores / 2);
    }
  }

  return nOptions;
}

int adjustGridSize(int GPUIndex, int defaultGridSize) {
  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, GPUIndex));
  int maxGridSize = deviceProp.multiProcessorCount * 40;
  return ((defaultGridSize > maxGridSize) ? maxGridSize : defaultGridSize);
}

///////////////////////////////////////////////////////////////////////////////
// CPU reference functions
///////////////////////////////////////////////////////////////////////////////

// Black-Scholes formula for call options
extern "C" void BlackScholesCall(float &CallResult, TOptionData optionData);

////////////////////////////////////////////////////////////////////////////////
// GPU-driving host thread
////////////////////////////////////////////////////////////////////////////////
// Timer
StopWatchInterface *hTimer = NULL;

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
void usage() {
  printf("--method=[threaded,streamed] --scaling=[strong,weak] [--help]\n");
  printf("Method=threaded: 1 CPU thread for each GPU     [default]\n");
  printf(
      "       streamed: 1 CPU thread handles all GPUs (requires CUDA 4.0 or "
      "newer)\n");
  printf("Scaling=strong : constant problem size\n");
  printf(
      "        weak   : problem size scales with number of available GPUs "
      "[default]\n");
}

//int main(int argc, char **argv)
float* vineFunCallerInterface::vine_montecarlo()
{
  char *multiMethodChoice = NULL;
  char *scalingChoice = NULL;
  char *accelChoice = NULL;
  bool use_threads = true;
  bool bqatest = false;
  bool strongScaling = false;
  vine_accel_type_e type;
  float totalGpuTime=0;
  float* gpuTimesArray;
  gpuTimesArray=(float*)malloc(NUM_ITERATIONS*sizeof(float));

  //pArgc = &argc;
  //pArgv = argv;

  vine_talk_init();

  printf("vine_MonteCarloMultiGPU Starting...\n\n");
  /*
  if (checkCmdLineFlag(argc, (const char **)argv, "qatest")) {
    bqatest = true;
  }

  getCmdLineArgumentString(argc, (const char **)argv, "method",
                           &multiMethodChoice);
  getCmdLineArgumentString(argc, (const char **)argv, "scaling",
                           &scalingChoice);
  getCmdLineArgumentString(argc, (const char **)argv, "accelerator",
                           &accelChoice);

  // Ignore help for now
  if (checkCmdLineFlag(argc, (const char **)argv, "h") ||
      checkCmdLineFlag(argc, (const char **)argv, "help")) {
    usage();
    exit(EXIT_SUCCESS);
  }
  */

  if (multiMethodChoice == NULL) {
    use_threads = false;
  } else {
    if (!strcasecmp(multiMethodChoice, "threaded")) {
      use_threads = true;
    } else {
      use_threads = false;
    }
  }

  if (use_threads == false) {
    printf("Using single CPU thread for multiple GPUs\n");
  }

  if (scalingChoice == NULL) {
    strongScaling = false;
  } else {
    if (!strcasecmp(scalingChoice, "strong")) {
      strongScaling = true;
    } else {
      strongScaling = false;
    }
  }

  if (accelChoice == NULL) {
    type = GPU;
    printf("Using GPU accelerator\n");
  } else {
    if (!strcasecmp(accelChoice, "CPU")) {
      type = CPU;
      printf("Using CPU accelerator\n");
    } else if (!strcasecmp(accelChoice, "GPU")) {
      type = GPU;
      printf("Using GPU accelerator\n");
    }
  }

  // GPU number present in the system
  int nOptions = 8 * 1024;

  // select problem size
  int OPT_N = nOptions;
  int PATH_N = 262144;

  // initialize the timers
  // Input data array
  TOptionData *OptionData = new TOptionData[OPT_N];
  // Final GPU MC results
  TOptionValue *CallValue = new TOptionValue[OPT_N];

  // Solver config
  TOptionPlan *optionSolver = new TOptionPlan[1];

  int gridSize;
  int i;

  float time;

  double delta, ref;
  //sumDelta=0, sumRef=0, sumReserve=0;

  printf("MonteCarloMultiGPU\n");
#if (DEBUG)
  printf("==================\n");
  printf("Parallelization method  = %s\n",
         use_threads ? "threaded" : "streamed");
  printf("Problem scaling         = %s\n", strongScaling ? "strong" : "weak");
  printf("Number of GPUs          = 1\n");
  printf("Total number of options = %d\n", OPT_N);
  printf("Number of paths         = %d\n", PATH_N);

  printf("main(): generating input data...\n");
#endif

#ifdef STATISTICS_TIMERS_ENABLED
  /*timer for input creation*/
  std::chrono::time_point<std::chrono::system_clock> startCreating, endCreating,
      startMemCpySHM, stopMemCpySHM;

  startCreating = std::chrono::system_clock::now();
#endif

  srand(123);

  for (i = 0; i < OPT_N; i++) {
    OptionData[i].S = randFloat(5.0f, 50.0f);
    OptionData[i].X = randFloat(10.0f, 25.0f);
    OptionData[i].T = randFloat(1.0f, 5.0f);
    OptionData[i].R = 0.06f;
    OptionData[i].V = 0.10f;
    CallValue[i].Expected = -1.0f;
    CallValue[i].Confidence = -1.0f;
  }
#ifdef STATISTICS_TIMERS_ENABLED
  endCreating = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = endCreating - startCreating;
  cout << endl;
  cout << "--------------------Breakdown Stats----------------------" << endl;
  cout << "Creating 2 Inputs with sizes " << OPT_N << " took: " << elapsed.count() << " sec." << endl;
#endif

  printf("main(): starting 1 host threads...\n");

//  gridSize = adjustGridSize(0, OPT_N);
  gridSize = 512;   // Value chosen after performing the ultimate scientific apporach, "searching the net".

  /**************************************************************************/
  // vine_talk additions.
  vine_accel *accel;  // The accelerator to use

  // Choose an accelerator.
  accel = vine_accel_acquire_type(type);

  // Create a pointer to the vine argument struct.
  monteCarloArgs *arguments;
  // Get the actual data (type cast to vine_argument).
  arguments = (monteCarloArgs *)malloc(sizeof(monteCarloArgs));

  // Pass the appropriate data to the struct.
  arguments->gridSize = gridSize;
  arguments->optionCount = OPT_N;
  arguments->pathN = PATH_N;
  vine_buffer_s vine_args[1] = {VINE_BUFFER(arguments, sizeof(monteCarloArgs))};

  // End of vine_talk additions.
  /**************************************************************************/
  if (type == GPU) {
    if (use_threads || bqatest) {
      cout << "Threads not supported!" << endl;
    }

    if (!use_threads || bqatest) {
      // allocate and initialize an array of stream handles
      void *rngStates;  // curandState *?

      // host-side
      __TOptionData *h_OptionData;
      __TOptionValue *h_CallValue;

      // Create a task id.
      vine_proc *process_id = vine_proc_get(type, "MonteCarlo");

      // Allocate inputs.
      // Allocate states for pseudo random number generators
      // checkCudaErrors(cudaMalloc((void **) &plan->rngStates,
      //                                 plan->gridSize * THREAD_N *
      //                                 sizeof(curandState)));
      rngStates = malloc(gridSize * THREAD_N * sizeof(curandState));
      h_OptionData = (__TOptionData *)malloc(sizeof(__TOptionData) * OPT_N);

      // Allocate outputs.
      h_CallValue =  (__TOptionValue *)malloc(sizeof(__TOptionValue) * OPT_N);

      for (i = 0; i < OPT_N; i++) {
        const double T = OptionData[i].T;
        const double R = OptionData[i].R;
        const double V = OptionData[i].V;
        const double MuByT = (R - 0.5 * V * V) * T;
        const double VBySqrtT = V * sqrt(T);
        h_OptionData[i].S = (real)OptionData[i].S;
        h_OptionData[i].X = (real)OptionData[i].X;
        h_OptionData[i].MuByT = (real)MuByT;
        h_OptionData[i].VBySqrtT = (real)VBySqrtT;
      }

      // Pointers to the shared memory segment containing input data.
      vine_buffer_s inputs[2];
      vine_buffer_s outputs[1];

      /*vine buffer inputs*/
      inputs[0] = VINE_BUFFER(rngStates, gridSize * THREAD_N * sizeof(curandState));
      inputs[1] = VINE_BUFFER(h_OptionData, sizeof(__TOptionData) * OPT_N);

      /*vine buffer outputs*/
      outputs[0] = VINE_BUFFER(h_CallValue, sizeof(__TOptionValue) * OPT_N);

      sdkCreateTimer(&hTimer);
      // Main computations
      for(int index =0; index<NUM_ITERATIONS;index++){
        sdkResetTimer(&hTimer);
        sdkStartTimer(&hTimer);
            vine_task *task =
                vine_task_issue(accel, process_id, vine_args, 2, inputs, 1, outputs);
            // Wait for task or exit if it fails.11
            if (vine_task_wait(task) != task_completed) {
              cerr << "Task failed!" << endl;
              return NULL;
            }
      #if (VERIFICATION_ENABLED)

            for (i = 0; i < OPT_N; i++) {
              const double RT = OptionData[i].R * OptionData[i].T;
              const double sum = h_CallValue[i].Expected;
              const double sum2 = h_CallValue[i].Confidence;
              const double pathN = PATH_N;
              // Derive average from the total sum and discount by riskfree rate
              CallValue[i].Expected = (float)(exp(-RT) * sum / pathN);
              // Standart deviation
              double stdDev =
                  sqrt((pathN * sum2 - sum * sum) / (pathN * (pathN - 1)));
              // Confidence width; in 95% of all cases theoretical value lies within
              // these borders
              CallValue[i].Confidence =
                  (float)(exp(-RT) * 1.96 * stdDev / sqrt(pathN));
            }
      #endif
            vine_task_free(task);
            sdkStopTimer(&hTimer);
            gpuTimesArray[index]=sdkGetTimerValue(&hTimer);
            totalGpuTime+=gpuTimesArray[index];
      }
      cout<<"Total GPU time: "<<totalGpuTime<<"milliseconds\n";
      free(rngStates);
      free(h_CallValue);
      free(h_OptionData);
    }
  }
  /*CPU*/
  else {
    printf("Running CPU MonteCarlo...\n");

    __TOptionData *h_OptionData;
    __TOptionValue *h_CallValue;

    // Create a task id.
    vine_proc *process_id = vine_proc_get(type, "MonteCarlo");

    // Allocate inputs.
    h_OptionData = (__TOptionData *)malloc(sizeof(__TOptionData) * OPT_N);

    // Allocate outputs.
    h_CallValue =  (__TOptionValue *)malloc(sizeof(__TOptionValue) * OPT_N);

    for (i = 0; i < OPT_N; i++) {
        const double T = OptionData[i].T;
        const double R = OptionData[i].R;
        const double V = OptionData[i].V;
        const double MuByT = (R - 0.5 * V * V) * T;
        const double VBySqrtT = V * sqrt(T);
        h_OptionData[i].S = (real)OptionData[i].S;
        h_OptionData[i].X = (real)OptionData[i].X;
        h_OptionData[i].MuByT = (real)MuByT;
        h_OptionData[i].VBySqrtT = (real)VBySqrtT;
    }

    // Pointers to the shared memory segment containing input data.
    vine_buffer_s inputs[1];
    vine_buffer_s outputs[1];

    // vine buffer inputs.
    inputs[0] = VINE_BUFFER(h_OptionData, sizeof(__TOptionData) * OPT_N);
    // vine buffer outputs.
    outputs[0] = VINE_BUFFER(h_CallValue, sizeof(__TOptionValue) * OPT_N);

    // Main computations
    vine_task *task = vine_task_issue(accel, process_id, vine_args, 1, inputs, 1, outputs);

    // Wait for task or exit if it fails.
    if (vine_task_wait(task) != task_completed) {
      cerr << "Task failed!" << endl;
      return NULL;
    }

    cout << "Received results." << endl;

    vine_task_free(task);
    free(h_CallValue);
    free(h_OptionData);

    printf("Shutting down...\n");
  }
  delete[] optionSolver;
  delete[] CallValue;
  sdkDeleteTimer(&hTimer);
  delete[] OptionData;
#if (VERIFICATION_ENABLED)
  //printf("Test Summary...\n");
  //printf("L1 norm        : %E\n", sumDelta / sumRef);
  //printf("Average reserve: %f\n", sumReserve);
#endif
  printf(
      "\nNOTE: The CUDA Samples are not meant for performance measurements. "
      "Results may vary when GPU Boost is enabled.\n\n");

  vine_accel_release(&accel);
  vine_talk_exit();
  printf("Exiting MonteCarloMultiGPU\n");
  //exit(EXIT_SUCCESS);
  return gpuTimesArray;
}

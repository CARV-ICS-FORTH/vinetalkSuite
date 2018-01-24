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
#define ENABLE_TIMERS (0)
#define VERIFICATION_ENABLED (0)
/*
* This sample implements 64-bin histogram calculation
* of arbitrary-sized 8-bit data array
*/
#include <iostream>
#include <chrono>

// CUDA Runtime
#include <cuda_runtime.h>

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

// projece include
#include "histogram_common.h"

// includes for vine_talk
#include "vine_talk.h"
#include "histogramArgs.h"
#include "statisticsDefineEnable.h"

using std::cout;
using std::endl;
using std::cerr;

const static char *sSDKsample = "[histogram]\0";
uint maxPartialHist64Count = 32768;  // is used in initHistogram64
uint partialHistogram256Count = 240;

int main(int argc, char **argv) {
  uchar *h_Data;
  uint *h_HistogramCPU, *h_HistogramGPU, *h_Hist_for_verification;
  StopWatchInterface *hTimer = NULL;
  int PassFailFlag = 1;
  uint byteCount = 64 * 1048576;
  uint uiSizeMult = 1;

  int accelType = 1;         // CPU = 3 and GPU=1
  int histogramVersion = 1;  // 0=64 and 1 =256
  int numRuns = 1;           // number of iteration that kernel is executed

  vine_talk_init();

// set logfile name and start logs
#if (DEBUG_ENABLED)
  printf("[%s] - Starting...\n", sSDKsample);
  printf(
      "******USAGE:*******\n  <sizemult = > : to increase the size of the "
      "image times sizemult.\n  <accelType = 3> : to execute the CPU version "
      "or <accelType=1> for GPU.\n  <histogramVersion = 0> : to execute the 64 "
      "bin histogram version \nor <histogramVersion=1> : to execute 256 "
      "histogram version.\n");
  cout << "*****************************************" << endl;
#endif
  // Optional Command-line multiplier to increase size of array to histogram
  if (checkCmdLineFlag(argc, (const char **)argv, "sizemult")) {
    uiSizeMult = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
    uiSizeMult = MAX(1, MIN(uiSizeMult, 10));
    printf(" Increase the array size by: %u\n", uiSizeMult);
    byteCount *= uiSizeMult;
  }
  // Command line get the accelerator type
  if (checkCmdLineFlag(argc, (const char **)argv, "accelType")) {
    accelType = getCmdLineArgumentInt(argc, (const char **)argv, "accelType");
    if (accelType == 3) {  // CPU
      printf("Accelerator type is CPU.  \n");
    } else if (accelType == 1) {  // GPU
      printf("Accelerator type is GPU.  \n");
    } else {
      printf("Wrong choice in accelerator type!\n");
      printf("The system will exit....");
      return -1;
    }
  }
  // Command line multiplier to specify which histogram version to run
  // 0 is for 64 and 1 is for 256
  if (checkCmdLineFlag(argc, (const char **)argv, "histogramVersion")) {
    histogramVersion =
        getCmdLineArgumentInt(argc, (const char **)argv, "histogramVersion");
    if (histogramVersion == 0) {  // 64
      printf(" Histogram 64 bin version.  \n");
    } else if (histogramVersion == 1) {  // 256
      printf(" Histogram 256 bin version.  \n");
    } else {
      printf("Wrong choice in histogram version!\n");
      printf("The system will exit....");
      return -1;
    }
  }
  // Command line multiplier to specify the number of iterations
  // that the kernel is going to be executed
  if (checkCmdLineFlag(argc, (const char **)argv, "numIterations")) {
    numRuns = getCmdLineArgumentInt(argc, (const char **)argv, "numIterations");
  }
#if (ENABLE_TIMERS)
  sdkCreateTimer(&hTimer);
#endif

#if (DEBUG_ENABLED)
  printf("Initializing data...\n");
#endif

  h_Data = (uchar *)malloc(byteCount);
  h_Hist_for_verification =
      (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
  h_HistogramCPU = (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
  h_HistogramGPU = (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));

#if (DEBUG_ENABLED)
  printf("...generating input data\n");
#endif
#ifdef STATISTICS_TIMERS_ENABLED
  /*timer for input creation*/
  std::chrono::time_point<std::chrono::system_clock> startCreating, endCreating,
      startMemCpySHM, stopMemCpySHM;

  startCreating = std::chrono::system_clock::now();
#endif

  srand(2009);

  for (uint i = 0; i < byteCount; i++) {
    h_Data[i] = rand() % 256;
  }

#ifdef STATISTICS_TIMERS_ENABLED
  endCreating = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = endCreating - startCreating;
  cout << endl;
  cout << "--------------------Breakdown Stats----------------------" << endl;
  cout << "Creating an input array with size: " << byteCount
       << " took: " << elapsed.count() << " sec." << endl;
#endif
  /*common variables for both histogram versions*/
  vine_buffer_s inputs[2];
  vine_buffer_s outputs[1];
  vine_buffer_s inputsCPU[1];
  vine_buffer_s outputsCPU[1];
  vine_accel **accels;
  int accels_count = 0;
  /* The accelerator to use */
  vine_accel *accel = 0;
uint * d_PartialHistograms;
  /*GPU*/
  if (accelType == 1) {
    if (histogramVersion == 0) {  // 64 bin version histogram
#if (DEBUG_ENABLED)
      printf("Initializing 64-bin histogram...\n");
#endif
      d_PartialHistograms = (uint *)malloc(
          maxPartialHist64Count * HISTOGRAM64_BIN_COUNT * sizeof(uint));
      assert(HISTOGRAM64_THREADBLOCK_SIZE % (4 * SHARED_MEMORY_BANKS) == 0);
      // Pointers to the shared memory segment containing input data.
      inputs[0] = VINE_BUFFER(h_Data, byteCount);
      inputs[1] = VINE_BUFFER(
          d_PartialHistograms,
          maxPartialHist64Count * HISTOGRAM64_BIN_COUNT * sizeof(uint));

      outputs[0] =
          VINE_BUFFER(h_HistogramGPU, HISTOGRAM256_BIN_COUNT * sizeof(uint));

#if (DEBUG_ENABLED)
      printf("Running 64-bin GPU histogram for %u bytes (%u runs)...\n\n",
             byteCount, numRuns);
#endif

      /******** 256 bin version histogram *****/
    } else if (histogramVersion == 1) {
      d_PartialHistograms = (uint *)malloc(
          partialHistogram256Count * HISTOGRAM256_BIN_COUNT * sizeof(uint));
#if (DEBUG_ENABLED)
      printf("Initializing 256-bin histogram...\n");
#endif
      // Pointers to the shared memory segment containing input data.
      inputs[0] = VINE_BUFFER(h_Data, byteCount);
      inputs[1] = VINE_BUFFER(
          d_PartialHistograms,
          partialHistogram256Count * HISTOGRAM256_BIN_COUNT * sizeof(uint));

      outputs[0] =
          VINE_BUFFER(h_HistogramGPU, HISTOGRAM256_BIN_COUNT * sizeof(uint));

#if (DEBUG_ENABLED)
      printf("Running 256-bin GPU histogram for %u bytes (%u runs)...\n\n",
             byteCount, numRuns);
#endif
    }

  } else if (accelType == 3) {
#if (DEBUG_ENABLED)
    printf("...allocating CPU memory and copying input data\n\n");
#endif
    // Pointers to the shared memory segment containing input data.
    inputsCPU[0] = VINE_BUFFER(h_Data, byteCount);

    outputsCPU[0] =
        VINE_BUFFER(h_HistogramCPU, HISTOGRAM256_BIN_COUNT * sizeof(uint));

#if (DEBUG_ENABLED)
    if (histogramVersion == 0) {
      printf(
          "Running 64-bin CPU histogram (vinetalk) for %u bytes (%u "
          "runs)...\n\n",
          byteCount, numRuns);
    } else if (histogramVersion == 1) {
      printf(
          "Running 256-bin CPU histogram (vinetalk) for %u bytes (%u "
          "runs)...\n\n",
          byteCount, numRuns);
    }
#endif
  }
  vine_accel_type_e accelerator = (vine_accel_type_e)accelType;

  /******************************* Start hist 64
   * ************************************/

  if (histogramVersion == 0) {  // 64 bin version histogram

    vine_proc *process_id = vine_proc_get(accelerator, "histogram64");

    /* Create a pointer to vine argument struct */
    histogramArgs *d_histogram_args;

    /* Get the actual data (type cast to vine_argument) */
    d_histogram_args = (histogramArgs *)malloc(sizeof(d_histogram_args));

    /* Pass arguments to the struct*/
    d_histogram_args->byteCount = byteCount;
    vine_buffer_s vine_args[1] = {
        VINE_BUFFER(d_histogram_args, sizeof(histogramArgs))};
    /* Choose accelerator randomly */
    accel = vine_accel_acquire_type((vine_accel_type_e)accelType);
    /******* Accelerator is GPU *******/
    if (accelType == 1) {
      sdkResetTimer(&hTimer);
      sdkStartTimer(&hTimer);

      /*memcpy HIGHLY important 2 inputs(d_Data, d_PartialHistogram) and 1
      * output(d_Histogram) */
      vine_task *task =
          vine_task_issue(accel, process_id, vine_args, 2, inputs, 1, outputs);

      if (vine_task_wait(task) == task_completed) {
        cout << "Histogram has been completed succesfully!" << endl;
      } else {
        cout << "Histogram has FAILED!" << endl;
        return -1;
      }

      vine_task_free(task);

#if (ENABLE_TIMERS)
      sdkStopTimer(&hTimer);
      double dAvgSecs =
          1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
#if (DEBUG_ENABLED)
      printf("histogram64() time (average) : %.5f sec, %.4f MB/sec\n\n",
             dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
      printf(
          "histogram64, Throughput = %.4f MB/s, Time = %.5f s, Size = %u "
          "Bytes, NumDevsUsed = %u, Workgroup = %u\n",
          (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1,
          HISTOGRAM64_THREADBLOCK_SIZE);
#endif
#endif

#if (VERIFICATION_ENABLED)
      printf("\nValidating GPU results...\n");

#if (DEBUG_ENABLED)
      printf(" ...histogram64CPU()\n");
#endif

      /* CPU execution */
      histogram64CPU(h_Hist_for_verification, h_Data, byteCount);

#if (DEBUG_ENABLED)
      printf(" ...comparing the results...\n");
#endif
      for (uint i = 0; i < HISTOGRAM64_BIN_COUNT; i++)
        if (h_HistogramGPU[i] != h_Hist_for_verification[i]) {
          PassFailFlag = 0;
        }
#if (DEBUG_ENABLED)
      printf(PassFailFlag ? " ...64-bin histograms match\n\n"
                          : " ***64-bin histograms do not match!!!***\n\n");

      printf("Shutting down 64-bin histogram...\n\n\n");
#endif
#endif

      /***** Accelerator CPU (vinetalk) ****/
    } else if (accelType == 3) {
      vine_task *task = vine_task_issue(accel, process_id, vine_args, 1,
                                        inputsCPU, 1, outputsCPU);
      if (vine_task_wait(task) == task_completed) {
        cout << "Histogram has been completed succesfully!" << endl;
      } else {
        cout << "Histogram has FAILED!" << endl;
        return -1;
      }
      vine_task_free(task);
#if (VERIFICATION_ENABLED)
      printf("\nValidating CPU results...\n");

#if (DEBUG_ENABLED)
      printf(" ...histogram64CPU()\n");
#endif

      /* CPU execution */
      histogram64CPU(h_Hist_for_verification, h_Data, byteCount);

#if (DEBUG_ENABLED)
      printf(" ...comparing the results...\n");
#endif
      for (uint i = 0; i < HISTOGRAM64_BIN_COUNT; i++)
        if (h_HistogramCPU[i] != h_Hist_for_verification[i]) {
          PassFailFlag = 0;
        }
#if (DEBUG_ENABLED)
      printf(PassFailFlag ? " ...64-bin histograms match\n\n"
                          : " ***64-bin histograms do not match!!!***\n\n");

      printf("Shutting down 64-bin histogram...\n\n\n");
#endif
#endif

    } else {
      cerr << "Not supported accelerator" << endl;
    }

    /**************** End 64***********/
    /********************************* 256 Version
     * *********************************/
  }

  else if (histogramVersion == 1) {
    /* Create a task id  */
    vine_proc *process_id = vine_proc_get(accelerator, "histogram256");

    /* Create a pointer to vine argument struct */
    histogramArgs *d_histogram_args;

    /* Get the actual data (type cast to vine_argument) */
    d_histogram_args = (histogramArgs *)malloc(sizeof(d_histogram_args));

    /* Pass arguments to the struct*/
    d_histogram_args->byteCount = byteCount;
    vine_buffer_s vine_args[1] = {
        VINE_BUFFER(d_histogram_args, sizeof(histogramArgs))};

    /* Choose accelerator randomly */
    accel = vine_accel_acquire_type((vine_accel_type_e)accelType);

    /******* Accelerator is GPU *******/
    if (accelType == 1) {
      sdkResetTimer(&hTimer);
      sdkStartTimer(&hTimer);
      /* HIGHLY important 2 inputs(d_Data, d_PartialHistogram) and 1
       * output(d_Histogram) */
      vine_task *task =
          vine_task_issue(accel, process_id, vine_args, 2, inputs, 1, outputs);
      if (vine_task_wait(task) == task_completed) {
        cout << "Histogram has been completed succesfully!" << endl;
      } else {
        cout << "Histogram has FAILED!" << endl;
        return -1;
      }
#if (ENABLE_TIMERS)
      sdkStopTimer(&hTimer);
      double dAvgSecs =
          1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;

#if (DEBUG_ENABLED)
      printf("histogram256() time (average) : %.5f sec, %.4f MB/sec\n\n",
             dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
      printf(
          "histogram256, Throughput = %.4f MB/s, Time = %.5f s, Size = %u "
          "Bytes, "
          "NumDevsUsed = %u, Workgroup = %u\n",
          (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1,
          HISTOGRAM256_THREADBLOCK_SIZE);
#endif
#endif

#if (VERIFICATION_ENABLED)
      printf("\nValidating GPU results...\n");
#if (DEBUG_ENABLED)
      printf(" ...histogram256CPU()\n");
#endif
      /* CPu histogram*/
      histogram256CPU(h_Hist_for_verification, h_Data, byteCount);
#if (DEBUG_ENABLED)
      printf(" ...comparing the results\n");
#endif
      for (uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
        if (h_HistogramGPU[i] != h_Hist_for_verification[i]) {
          PassFailFlag = 0;
        }
#if (DEBUG_ENABLED)
      printf(PassFailFlag ? " ...256-bin histograms match\n\n"
                          : " ***256-bin histograms do not match!!!***\n\n");
      printf("Shutting down 256-bin histogram...\n\n\n");
#endif
#endif
      /***** Accelerator CPU (vinetalk) ****/
    } else if (accelType == 3) {
      vine_task *task = vine_task_issue(accel, process_id, vine_args, 1,
                                        inputsCPU, 1, outputsCPU);
      if (vine_task_wait(task) == task_completed) {
        cout << "Histogram has been completed succesfully!" << endl;
      } else {
        cout << "Histogram has FAILED!" << endl;
        return -1;
      }
      vine_task_free(task);

#if (VERIFICATION_ENABLED)
      printf("\nValidating CPU results...\n");
#if (DEBUG_ENABLED)
      printf(" ...histogram256CPU()\n");
#endif
      /* CPu histogram*/
      histogram256CPU(h_Hist_for_verification, h_Data, byteCount);
#if (DEBUG_ENABLED)
      printf(" ...comparing the results\n");
#endif
      for (uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
        if (h_HistogramGPU[i] != h_Hist_for_verification[i]) {
          PassFailFlag = 0;
        }
#if (DEBUG_ENABLED)
      printf(PassFailFlag ? " ...256-bin histograms match\n\n"
                          : " ***256-bin histograms do not match!!!***\n\n");
      printf("Shutting down 256-bin histogram...\n\n\n");
#endif
#endif

    } else {
      cerr << "Not supported accelerator" << endl;
    }
  }
/***************************  End hist256  *************************/
#if (DEBUG_ENABLED)
  printf("Shutting down...\n");
#endif
#if (ENABLE_TIMERS)
  sdkDeleteTimer(&hTimer);
#endif
  free(h_HistogramGPU);
  free(h_HistogramCPU);
  free(h_Data);
  vine_accel_release(&accel);

#if (VERIFICATION_ENABLED)
  printf("%s - Test Summary\n", sSDKsample);
  // pass or fail (for both 64 bit and 256 bit histograms)
  if (!PassFailFlag) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("Test passed\n");
#endif
  vine_talk_exit();
  exit(EXIT_SUCCESS);
}

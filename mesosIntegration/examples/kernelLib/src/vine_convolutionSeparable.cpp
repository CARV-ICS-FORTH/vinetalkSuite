/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
:q * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
* This sample implements a separable convolution filter
* of a 2D image with an arbitrary kernel.
*/

// CUDA runtime
#include <cuda_runtime.h>
#include <time.h>
#include <chrono>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>

#include "convolutionSeparable_common.h"

// includes for vine_talk
#include "vine_talk.h"
#include "convolutionSeparableArgs.h"
#include "statisticsDefineEnable.h"
#include "vineFunCaller.hpp"

#include <iostream>

using std::cout;
using std::endl;
using std::cerr;
#define ENABLE_TIMERS (0)
#define VERIFICATION_ENABLED (0)
#define STATISTICS_TIMERS_ENABLED (1)
////////////////////////////////////////////////////////////////////////////////
// Reference CPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionRowCPU(float *h_Result, float *h_Data,
                                  float *h_Kernel, int imageW, int imageH,
                                  int kernelR);

extern "C" void convolutionColumnCPU(float *h_Result, float *h_Data,
                                     float *h_Kernel, int imageW, int imageH,
                                     int kernelR);

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
//int main(int argc, char **argv)
int vineFunCallerInterface::vine_convolutionSeparable() 
{
  /* Initialize Vine talk*/
  vine_talk_init();

  // start logs
  printf("vine_ convolutionSeparable - Starting...\n");
  cout << "******USAGE:*******\n <accelType = 3> : to execute the CPU version "
          "or <accelType=1> for GPU.\n" << endl;
  cout << "*****************************************" << endl;
  float *h_Kernel,              /*Input*/
      *h_Input,                 /*Input*/
      *h_OutputCPU,             /*Output*/
      *h_OutputGPU,             /*Output*/
      *h_OutputForVerification, /*Output*/
      *d_Buffer,                /*Partial Buffers for Vinetalk + GPU*/
      *d_Buffer_CPU,            /*Partial Buffers for VineTalk + CPU*/
      *h_Buffer;                /*Partial Buffers for CPU without Vinetalk*/

  int accelType = 1;  // CPU = 3 and GPU=1
  const int imageW = 3072;
  const int imageH = 3072;

  // Command line get the accelerator type
  /*
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
  */
#if (ENABLE_TIMERS)
  StopWatchInterface *hTimer = NULL;
  sdkCreateTimer(&hTimer);
#endif
#if (DEBUG_ENABLED)
  printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
  printf("Allocating and initializing host arrays...\n");
#endif
  /*Inputs*/
  h_Kernel = (float *)malloc(KERNEL_LENGTH * sizeof(float));
  h_Input = (float *)malloc(imageW * imageH * sizeof(float));
  /*Partial Buffers*/
  d_Buffer = (float *)malloc(imageW * imageH * sizeof(float));
  d_Buffer_CPU = (float *)malloc(imageW * imageH * sizeof(float));
  h_Buffer = (float *)malloc(imageW * imageH * sizeof(float));
  /*Outputs*/
  h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
  h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));
  h_OutputForVerification = (float *)malloc(imageW * imageH * sizeof(float));

#ifdef STATISTICS_TIMERS_ENABLED
  /*timer for input creation*/
  //std::chrono::time_point<std::chrono::system_clock> startCreating, endCreating,startMemCpySHM, stopMemCpySHM;
  std::chrono::milliseconds startCreating, endCreating,startMemCpySHM, stopMemCpySHM;
  //startCreating = std::chrono::system_clock::now();
  startCreating= std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch());
#endif
  srand(200);

  for (unsigned int i = 0; i < KERNEL_LENGTH; i++) {
    h_Kernel[i] = (float)(rand() % 16);
  }
  for (unsigned i = 0; i < imageW * imageH; i++) {
    h_Input[i] = (float)(rand() % 16);
  }

#ifdef STATISTICS_TIMERS_ENABLED
  //endCreating = std::chrono::system_clock::now();
  endCreating = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch());

  //std::chrono::duration<double> elapsed = endCreating - startCreating;
  std::chrono::milliseconds elapsed = endCreating - startCreating;
  cout << endl;
  cout << "--------------------Breakdown Stats----------------------" << endl;
  cout << "Creating 2 Inputs with sizes " << KERNEL_LENGTH << " , "
       << imageW *imageH << " took: " << elapsed.count() << " milliseconds " << endl;
#endif
  /* Inputs and Outputs for GPU*/
  vine_buffer_s inputs[3];
  vine_buffer_s outputs[1];

  /* Inputs Outputs for CPU*/
  vine_buffer_s inputsCPU[3];
  vine_buffer_s outputsCPU[1];

  vine_proc *process_id;
  double L2norm;
  if (accelType == 1) {
#if (DEBUG_ENABLED)
    printf("Accelerator is GPU.\n");
#endif

    /*vine buffer inputs*/
    inputs[0] = VINE_BUFFER(h_Input, imageW * imageH * sizeof(float));
    inputs[1] = VINE_BUFFER(d_Buffer, imageW * imageH * sizeof(float));
    inputs[2] = VINE_BUFFER(h_Kernel, KERNEL_LENGTH * sizeof(float));

    outputs[0] = VINE_BUFFER(h_OutputGPU, imageW * imageH * sizeof(float));

  } else if (accelType == 3) {
#if (DEBUG_ENABLED)
    printf("Accelerator is CPU\n");
#endif
    inputsCPU[0] = VINE_BUFFER(h_Input, imageW * imageH * sizeof(float));
    inputsCPU[1] = VINE_BUFFER(d_Buffer_CPU, imageW * imageH * sizeof(float));
    inputsCPU[2] = VINE_BUFFER(h_Kernel, KERNEL_LENGTH * sizeof(float));

    outputsCPU[0] = VINE_BUFFER(h_OutputCPU, imageW * imageH * sizeof(float));
  }

  vine_accel_type_e accelerator = (vine_accel_type_e)accelType;
  process_id = vine_proc_get(accelerator, "convolution");

  /* Create a pointer to vine argument struct */
  convolutionArgs *conv_args;

  /* Get the actual data (type cast to vine_argument) */
  conv_args = (convolutionArgs *)malloc(sizeof(conv_args));

  /* Pass arguments to the struct*/
  conv_args->imageW = imageW;
  conv_args->imageH = imageH;
  vine_buffer_s vine_args[1] = {
      VINE_BUFFER(conv_args, sizeof(convolutionArgs))};

  /* The accelerator to use */
  vine_accel *accel = vine_accel_acquire_type(accelerator);

  if (accelType == 1) {
    printf("Running GPU convolution ( Vine Talk ) ...\n\n");
  } else if (accelType == 3) {
    printf("Running CPU convolution ( Vine Talk ) ...\n\n");
  }
  if (accelType == 1) {
#if (ENABLE_TIMERS)
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
#endif
#if (DEBUG_ENABLED)
    cout << "Vine talk issue..." << endl;
#endif
    vine_task *task =
        vine_task_issue(accel, process_id, vine_args, 3, inputs, 1, outputs);
    if (vine_task_wait(task) == task_completed) {
      cout << "Convolution has been completed succesfully!" << endl;
    } else {
      cout << "Convolution has FAILED!" << endl;
      return -1;
    }

    vine_task_free(task);

#if (ENABLE_TIMERS)
    sdkStopTimer(&hTimer);
    double gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
#if (DEBUG_ENABLED)
    printf(
        "convolutionSeparable, Throughput = %.4f MPixels/sec, Time = %.5f s, "
        "Size = %u Pixels, NumDevsUsed = %i, Workgroup = %u\n",
        (1.0e-6 * (double)(imageW * imageH) / gpuTime), gpuTime,
        (imageW * imageH), 1, 0);
#endif
#endif

#if (DEBUG_ENABLED)
    printf("\nReading back GPU results...\n\n");
#endif

#if (VERIFICATION_ENABLED)
#if (DEBUG_ENABLED)
    printf("Checking the results...\n");
    printf(" ...running convolutionRowCPU()\n");
#endif
    convolutionRowCPU(h_Buffer, h_Input, h_Kernel, imageW, imageH,
                      KERNEL_RADIUS);
#if (DEBUG_ENABLED)
    printf(" ...running convolutionColumnCPU()\n");
#endif
    convolutionColumnCPU(h_OutputForVerification, h_Buffer, h_Kernel, imageW,
                         imageH, KERNEL_RADIUS);
#if (DEBUG_ENABLED)
    printf(" ...comparing the results\n");
#endif
    double sum = 0, delta = 0;

    for (unsigned i = 0; i < imageW * imageH; i++) {
      delta += (h_OutputGPU[i] - h_OutputForVerification[i]) *
               (h_OutputGPU[i] - h_OutputForVerification[i]);
      sum += h_OutputForVerification[i] * h_OutputForVerification[i];
    }

    L2norm = sqrt(delta / sum);
    printf(" ...Relative L2 norm: %E\n\n", L2norm);
#endif

  } else if (accelType == 3) {
    conv_args->kernelR = KERNEL_RADIUS;
#if (DEBUG_ENABLED)
    cout << "Vine talk issue CPU..." << endl;
#endif
    vine_task *task = vine_task_issue(accel, process_id, vine_args, 3,
                                      inputsCPU, 1, outputsCPU);
#if (DEBUG_ENABLED)
    cout << "Wait for the result" << endl;
#endif
    if (vine_task_wait(task) == task_completed) {
      cout << "Convolution has been completed succesfully!" << endl;
    } else {
      cout << "Convolution has FAILED!" << endl;
      return -1;
    }
    vine_task_free(task);
#if (DEBUG_ENABLED)
    printf("\nReading back CPU results...\n\n");
#endif

#if (VERIFICATION_ENABLED)
#if (DEBUG_ENABLED)
    printf("Checking the results...\n");
    printf(" ...running convolutionRowCPU()\n");
#endif
    convolutionRowCPU(h_Buffer, h_Input, h_Kernel, imageW, imageH,
                      KERNEL_RADIUS);
#if (DEBUG_ENABLED)
    printf(" ...running convolutionColumnCPU()\n");
#endif
    convolutionColumnCPU(h_OutputForVerification, h_Buffer, h_Kernel, imageW,
                         imageH, KERNEL_RADIUS);
#if (DEBUG_ENABLED)
    printf(" ...comparing the results\n");
#endif
    double sum = 0, delta = 0;

    for (unsigned i = 0; i < imageW * imageH; i++) {
      delta += (h_OutputCPU[i] - h_OutputForVerification[i]) *
               (h_OutputCPU[i] - h_OutputForVerification[i]);
      sum += h_OutputForVerification[i] * h_OutputForVerification[i];
    }

    L2norm = sqrt(delta / sum);
    printf(" ...Relative L2 norm: %E\n\n", L2norm);
#endif

  } else {
    cout << "Wrong accelerator type." << endl;
  }

#if (DEBUG_ENABLED)
  printf("Shutting down...\n");
#endif
  /*Outputs*/
  free(h_OutputGPU);
  free(h_OutputCPU);
  free(h_OutputForVerification);
  /*Partial*/
  free(d_Buffer_CPU);
  free(d_Buffer);
  free(h_Buffer);
  /*Inputs*/
  free(h_Input);
  free(h_Kernel);
#if (VERIFICATION_ENABLED)
  if (L2norm > 1e-6) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }
  printf("Test passed\n");
#endif

  /* Close Vinetalk*/
  vine_talk_exit();
  //exit(EXIT_SUCCESS);
  printf("Exiting Test\n");
  return elapsed.count()<0 ? -1 : elapsed.count()  ;
}

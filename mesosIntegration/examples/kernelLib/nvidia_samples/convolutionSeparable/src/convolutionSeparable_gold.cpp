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

#include "../include/convolutionSeparable_common.h"

#include <iostream>
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <vine_talk.h>
#include <vine_pipe.h>
#include "VineLibUtilsCPU.h"
#include "../include/convolutionSeparableArgs.h"
using std::cout;
using std::endl;
float *h_Buffer;

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Kernel,
                                  int imageW, int imageH, int kernelR) {
  for (int y = 0; y < imageH; y++)
    for (int x = 0; x < imageW; x++) {
      float sum = 0;

      for (int k = -kernelR; k <= kernelR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW)
          sum += h_Src[y * imageW + d] * h_Kernel[kernelR - k];
      }

      h_Dst[y * imageW + x] = sum;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionColumnCPU(float *h_Dst, float *h_Src,
                                     float *h_Kernel, int imageW, int imageH,
                                     int kernelR) {
  for (int y = 0; y < imageH; y++)
    for (int x = 0; x < imageW; x++) {
      float sum = 0;

      for (int k = -kernelR; k <= kernelR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH)
          sum += h_Src[d * imageW + x] * h_Kernel[kernelR - k];
      }

      h_Dst[y * imageW + x] = sum;
    }
}

/* ****** Convolution function call ****** */
extern "C" void convolutionCPU(float *h_OutputCPU, float *h_Input,
                               float *h_Kernel, int imageW, int imageH,
                               int kernelR) {
  /* Call the first kernel */
  convolutionRowCPU(h_Buffer, h_Input, h_Kernel, imageW, imageH, kernelR);

  /* Call the second kernel */
  convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Kernel, imageW, imageH,
                       kernelR);
}

vine_task_state_e hostCodeCPU(vine_task_msg_s *vine_task) {
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::vector<void *> ioVector;

  cout << "Convolution execution in CPU." << endl;

  convolutionArgs *conv_args;
  conv_args = (convolutionArgs *)vine_data_deref(vine_task->args.vine_data);
#if (DEBUG_ENABLED)
  cout << "host2cpu" << endl;
#endif
  Host2CPU(vine_task, ioVector);

  // Wrong !!! h_Buffer = (float *)ioVector[1];
  h_Buffer = (float *)vine_data_deref(vine_task->io[1].vine_data);
#if (DEBUG_ENABLED)
  cout << "Call the kernel" << endl;
#endif

/*Old usage*/
//  convolutionCPU((float *) vine_data_deref(vine_task->io[3]) , (float *)
//  vine_data_deref(vine_task->io[0]),  (float
//  *)vine_data_deref(vine_task->io[2]), conv_args->imageW, conv_args->imageH,
//  conv_args->kernelR);
#ifdef TIMERS_ENABLED
  start = std::chrono::system_clock::now();
#endif

  convolutionCPU((float *)ioVector[3], (float *)ioVector[0],
                 (float *)ioVector[2], conv_args->imageW, conv_args->imageH,
                 conv_args->kernelR);

#ifdef TIMERS_ENABLED

  end = std::chrono::system_clock::now();

  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::chrono::duration<double> elapsed_seconds = end - start;

  cout << "Convolution CPU kernel execution time: " << elapsed_seconds.count()
       << " secs." << endl;
#endif

#if (DEBUG_ENABLED)
  cout << "Get results back" << endl;
#endif
  CPU2Host(vine_task, ioVector);

#if (DEBUG_ENABLED)
  cout << "Free" << endl;
#endif
  CPUMemFree(ioVector);

  return vine_task_stat(vine_task, 0);
}

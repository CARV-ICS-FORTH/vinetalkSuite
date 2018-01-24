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
#include <iostream>
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <vine_talk.h>
#include <vine_pipe.h>
#include "histogram_common.h"
#include "VineLibUtilsCPU.h"
#include "histogramArgs.h"


using std::cout;
using std::endl;

extern "C" void histogram64CPU(uint *h_Histogram, void *h_Data,
                               uint byteCount) {
#if (DEBUG_ENABLED)
  cout << "CPU kernel Histogram 64 in " << __FILE__ << endl;
#endif
  for (uint i = 0; i < HISTOGRAM64_BIN_COUNT; i++) h_Histogram[i] = 0;

  assert(sizeof(uint) == 4 && (byteCount % 4) == 0);

  for (uint i = 0; i < (byteCount / 4); i++) {
    uint data = ((uint *)h_Data)[i];
    h_Histogram[(data >> 2) & 0x3FU]++;
    h_Histogram[(data >> 10) & 0x3FU]++;
    h_Histogram[(data >> 18) & 0x3FU]++;
    h_Histogram[(data >> 26) & 0x3FU]++;
  }
}

extern "C" void histogram256CPU(uint *h_Histogram, void *h_Data,
                                uint byteCount) {

#if (DEBUG_ENABLED)
  cout << "CPU kernel Histogram 256" << __FILE__ << endl;
#endif

  for (uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++) h_Histogram[i] = 0;

  assert(sizeof(uint) == 4 && (byteCount % 4) == 0);

  for (uint i = 0; i < (byteCount / 4); i++) {
    uint data = ((uint *)h_Data)[i];
    h_Histogram[(data >> 0) & 0xFFU]++;
    h_Histogram[(data >> 8) & 0xFFU]++;
    h_Histogram[(data >> 16) & 0xFFU]++;
    h_Histogram[(data >> 24) & 0xFFU]++;
  }
}

vine_task_state_e hostCode64CPU(vine_task_msg_s *vine_task) {
  std::vector<void *> ioVector;
  std::chrono::time_point<std::chrono::system_clock> start, end;

  cout << "Histogram 64 execution in CPU." << endl;

  histogramArgs *args_cuda64;
  args_cuda64 = (histogramArgs *)vine_data_deref(vine_task->args.vine_data);
  Host2CPU(vine_task, ioVector);
#ifdef TIMERS_ENABLED
  start = std::chrono::system_clock::now();
#endif
  histogram64CPU((uint *)vine_data_deref(vine_task->io[1].vine_data),
                 (void *)vine_data_deref(vine_task->io[0].vine_data),
                 args_cuda64->byteCount);

#ifdef TIMERS_ENABLED

  end = std::chrono::system_clock::now();

  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::chrono::duration<double> elapsed_seconds = end - start;

  cout << "Histogram 64 CPU kernel execution time: " << elapsed_seconds.count()
       << " secs." << endl;
#endif

  CPU2Host(vine_task, ioVector);

 
  CPUMemFree(ioVector);

  return vine_task_stat(vine_task, 0);
}

vine_task_state_e hostCode256CPU(vine_task_msg_s *vine_task) {
  std::vector<void *> ioVector;
  std::chrono::time_point<std::chrono::system_clock> start, end;

  cout << "Histogram 256 execution in CPU." << endl;

  histogramArgs *args_cuda256;
  args_cuda256 = (histogramArgs *)vine_data_deref(vine_task->args.vine_data);
  Host2CPU(vine_task, ioVector);

#ifdef TIMERS_ENABLED
  start = std::chrono::system_clock::now();
#endif

  histogram256CPU((uint *)vine_data_deref(vine_task->io[1].vine_data),
                  (void *)vine_data_deref(vine_task->io[0].vine_data),
                  args_cuda256->byteCount);

#ifdef TIMERS_ENABLED

  end = std::chrono::system_clock::now();

  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::chrono::duration<double> elapsed_seconds = end - start;

  cout << "Histogram 256 CPU kernel execution time: " << elapsed_seconds.count()
       << " secs." << endl;
#endif

  CPU2Host(vine_task, ioVector);

  CPUMemFree(ioVector);

  return vine_task_stat(vine_task, 0);
}

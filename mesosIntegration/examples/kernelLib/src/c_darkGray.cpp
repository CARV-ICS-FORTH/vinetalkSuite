/*
 * Copyright 2018 Foundation for Research and Technology - Hellas
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at 
 * http://www.apache.org/licenses/LICENSE-2.0 [1] [1] 
 * 
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
 * See the License for the specific language governing permissions and 
 * limitations under the License. 
 * 
 * Links: 
 * 
 * [1] http://www.apache.org/licenses/LICENSE-2.0 [1] 
 */
#include <iostream>
#include <chrono>
/* Include utilities for CPU accel */
#include "VineLibUtilsCPU.h"
/* Include c_darkGray lib */
#include "c_darkGray.h"
/* include darkgray arguments*/
#include "darkGrayArgs.h"

using std::cout;
using std::endl;

/* Kernel to be executed in CPU */
void darkGray(const int width, const int height,
              const unsigned char *inputImage, unsigned char *darkGrayImage) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float grayPix = 0.0f;
      float r = static_cast<float>(inputImage[(y * width) + x]);
      float g =
          static_cast<float>(inputImage[(width * height) + (y * width) + x]);
      float b = static_cast<float>(
          inputImage[(2 * width * height) + (y * width) + x]);
      grayPix = ((0.3f * r) + (0.59f * g) + (0.11f * b));
      grayPix = (grayPix * 0.6f) + 0.5f;
      darkGrayImage[(y * width) + x] = static_cast<unsigned char>(grayPix);
    }
  }
}
/* Create a function to register in the dynamic libraries */
vine_task_state_e hostCode(vine_task_msg_s *vine_task) {
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::vector<void *> ioVector;
  darkGrayArgs *dArgs;
  /* Get the actual arguments and typecast them */
  dArgs = (darkGrayArgs *)vine_data_deref(vine_task->args.vine_data);

  cout << "c_darkgray execution in CPU." << endl;

  /* Transfer data to the accelerator's memory
   * Warning for CPU it does nothing ! (used for homogenity)
   * There is not device memory to copy them.
   */
  Host2CPU(vine_task, ioVector);
/* Call the actual kernel.
 * In case of CPU use it as bellow with vine_data_deref.
*/
#ifdef TIMERS_ENABLED
  start = std::chrono::system_clock::now();
#endif
  darkGray(dArgs->width, dArgs->height,
           (unsigned char *)vine_data_deref(vine_task->io[0].vine_data),
           (unsigned char *)vine_data_deref(vine_task->io[1].vine_data));

#ifdef TIMERS_ENABLED

  end = std::chrono::system_clock::now();

  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::chrono::duration<double,std::nano> elapsed_seconds = end - start;
  cout << "DarkGray kernel execution time: " << elapsed_seconds.count()
       << " nanosec." << endl;
#endif

  /* Transfer data from the accelerator's memory
   * Warning for CPU it does nothing ! (used for homogenity)
   *    There is not device memory to copy them.
  */
  CPU2Host(vine_task, ioVector);

  /* Free accelerator's memory
   *  Warning for CPU it does nothing ! (used for homogenity)
   *    There is not device memory to free them.
  */
  CPUMemFree(ioVector);

  /* Return completed or failed*/
return vine_task_stat(vine_task, 0);
}
/* Create a table according to VineLibMgr to store the functions exist in one
 * .so*/
VINE_PROC_LIST_START()
VINE_PROCEDURE("darkGray", CPU, hostCode, sizeof(darkGrayArgs))
VINE_PROC_LIST_END()

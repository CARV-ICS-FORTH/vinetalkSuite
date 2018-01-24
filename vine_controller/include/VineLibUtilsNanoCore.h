/*
 * Copyright 2018 Foundation for Research and Technology - Hellas
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0 [1] [1]
 *
 * Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 *  implied.
 * See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * Links:
 *  ------
 * [1] http://www.apache.org/licenses/LICENSE-2.0 [1] 
*/

#ifndef VINE_LIB_NANO_CORE_UTILS_HEADER
#define VINE_LIB_NANO_CORE_UTILS_HEADER

#include <vector>

using namespace::std;

#include <vine_pipe.h>
#include <VineLibMgr.h>
#ifdef LIBRARY_BUILD
/// @brief  Transfers data from host memory to NanoCore.
///
/// @param  vine_task   Vineyard task information.
/// @param  ioHD        Data to transfer.
///
/// @retval true    The transmission was successful.
/// @retval false   The transmission failed.
///
extern bool Host2NanoCore(vine_task_msg_s *vine_task, vector<void *> &ioHD);

/// @brief  Transfers data from NanoCore to host memory.
///
/// @param  vine_task   Vineyard task information.
/// @param  ioHD        Data to transfer.
///
/// @retval true    The transmission was successful.
/// @retval false   The transmission failed.
///
extern bool NanoCore2Host(vine_task_msg_s *vine_task, vector<void *> &ioDH);

/// @brief  Releases memory allocated on NanoCore.
///
/// @param  io  Address of the memory space on NanoCore to free.
///
/// @retval true    Freeing was successful.
/// @retval false   Freeing failed.
///
extern bool NanoCoreMemFree(vector<void *> &io);
#endif
#endif // !defined(VINE_LIB_UTILS_HEADER)

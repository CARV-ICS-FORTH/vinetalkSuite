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

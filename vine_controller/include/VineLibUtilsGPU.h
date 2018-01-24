#ifndef VINE_LIB_GPU_UTILS_HEADER
#define VINE_LIB_GPU_UTILS_HEADER

#include <vector>
#include "vine_pipe.h"
#include "VineLibMgr.h"
#ifdef LIBRARY_BUILD
extern bool Host2GPU(vine_task_msg_s *vine_task, std::vector<void *> &ioHD);

/* Cuda Memcpy from Device to host*/
extern bool GPU2Host(vine_task_msg_s *vine_task, std::vector<void *> &ioDH);

/* Free Device memory */
extern bool GPUMemFree(std::vector<void *> &io);
#endif
#endif

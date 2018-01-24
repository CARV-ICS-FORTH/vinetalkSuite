#ifndef VINE_LIB_CPU__UTILS_HEADER
#define VINE_LIB_CPU_UTILS_HEADER

#include <vector>
#include "vine_pipe.h"
#include "VineLibMgr.h"
#ifdef LIBRARY_BUILD
extern void Host2CPU(vine_task_msg_s *vine_task, std::vector<void *> &ioHD);

/* Cuda Memcpy from Device to host*/
extern void CPU2Host(vine_task_msg_s *vine_task, std::vector<void *> &ioDH);

/* Free Device memory */
extern void CPUMemFree(std::vector<void *> &io);
#endif
#endif

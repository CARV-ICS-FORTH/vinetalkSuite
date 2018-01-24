#ifndef VINE_LIB_SDA_UTILS_HEADER
#define VINE_LIB_SDA_UTILS_HEADER

#include <vector>
#include "vine_pipe.h"
#include "VineLibMgr.h"

#ifdef LIBRARY_BUILD
#include "xcl.h"
extern bool Host2SDA(vine_task_msg_s *vine_task, vector<void *> &ioHD,  xcl_world world, cl_kernel krnl);

extern bool SDA2Host(vine_task_msg_s *vine_task, vector<void *> &ioDH,xcl_world world, cl_kernel krnl);

extern bool SDAMemFree(vector<void *> &io);
#endif
#endif
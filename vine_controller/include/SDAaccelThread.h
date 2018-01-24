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
#ifndef SDA_ACCELTHREAD
#define SDA_ACCELTHREAD
#include <pthread.h>
#include <map>
#include "timers.h"
#include "xcl.h"
class SDAaccelThread;

#include "accelThread.h"

class SDAaccelThread : public accelThread {
    public:
		SDAaccelThread(vine_pipe_s * v_pipe, AccelConfig & conf);
        ~SDAaccelThread();
        virtual bool acceleratorInit(); /* Function that initializes a SDA accelerator */
        virtual void acceleratorRelease(); /* Function that resets a SDA accelerator */
        virtual void executeHostCode(void* functor, vine_task_msg_s* task);
    private:
        CpuSet prof_thread_cpus;			/* Pin Xilinx profiling thread*/
        std::string vendor;
        std::string dev_addr_str;
        std::string xclbin;				/** xcl bin file */
        std::map<std::string,cl_kernel> kernels;/** Available kernels */
        xcl_world world;
};
#endif

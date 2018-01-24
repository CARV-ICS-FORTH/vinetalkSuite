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

#ifndef CPU_ACCELTHREAD
#define CPU_ACCELTHREAD
#include <pthread.h>
#include <vector>
#include "timers.h"
class CPUaccelThread;

#include "accelThread.h"

class CPUaccelThread : public accelThread {
    public:
		CPUaccelThread(vine_pipe_s * v_pipe, AccelConfig &conf);
        ~CPUaccelThread();
        virtual bool acceleratorInit(); /* Function that initializes a CPU accelerator */
        virtual void acceleratorRelease(); /* Function that resets a CPU accelerator */
};
#endif

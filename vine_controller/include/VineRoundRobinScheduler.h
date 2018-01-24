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
#ifndef VINE_ROUND_ROBIN_SCHEDULER
#define VINE_ROUND_ROBIN_SCHEDULER
#include "Scheduler.h"
#include <map>
#include <vector>

class VineRoundRobinScheduler : public Scheduler {
    public:
		VineRoundRobinScheduler(std::string args);
        virtual ~VineRoundRobinScheduler();
        /*Pick a Virtual Accelerator Queue(VAQ) from all avaliable VAQs that exist in the system*/
        virtual utils_queue_s *selectVirtualAcceleratorQueue( accelThread *threadPerAccel );

        /*Select a task from all the VAQs that exist in the system  */
        virtual vine_task_msg_s *selectTask( accelThread *threadPerAccel );

    private:
        //Array with all VAQ types
        int acceleratorIndex[VINE_ACCEL_TYPES];
        map<vine_accel_s *, int> virtualQueueIndex;
};

#endif

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
#ifndef VINE_SCHEDULER
#define VINE_SCHEDULER

class Scheduler;

#include "Config.h"
#include "Factory.h"
#include "accelThread.h"
class Scheduler {
    public:
        Scheduler(std::string args);
		void setGroup(GroupConfig * group);
        virtual ~Scheduler() ;
        virtual utils_queue_s  *selectVirtualAcceleratorQueue(accelThread *th);
        virtual vine_task_msg_s *selectTask(accelThread *th) = 0;
		virtual void postTaskExecution(vine_task_msg_s *task);
        virtual void loadConfigurationFile() ;
    protected:
		GroupConfig * group;
};

extern Factory<Scheduler,std::string> schedulerFactory;

/**
 * Helper function to register Accelerator Threads
 * Must be put in a cpp file.
 */
#define REGISTER_SCHEDULER(CLASS)\
    static Registrator<Scheduler,CLASS,std::string> reg(schedulerFactory);
#endif

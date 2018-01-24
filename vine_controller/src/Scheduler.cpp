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
#include <iostream>
#include "Scheduler.h"
Scheduler::Scheduler(std::string args) {}

utils_queue_s  *Scheduler::selectVirtualAcceleratorQueue(accelThread *th)
{
	std::cerr << "Using " << __func__ << "although not implemented in scheduler.";
	throw std::runtime_error("Unimplemented function");
	return 0;
}

void Scheduler :: setGroup(GroupConfig * group)
{
	this->group = group;
}

Scheduler::~Scheduler() {}

void Scheduler::postTaskExecution(vine_task_msg_s *task)
{
}

void Scheduler::loadConfigurationFile()
{
    cout<<"Load configiration file dummy!!"<<endl;
}


Factory<Scheduler,std::string> schedulerFactory;

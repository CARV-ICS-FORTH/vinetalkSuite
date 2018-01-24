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
using namespace::std;
#include "VineRoundRobinScheduler.h"
VineRoundRobinScheduler::VineRoundRobinScheduler(std::string args) : Scheduler(args), acceleratorIndex{0} {}

VineRoundRobinScheduler::~VineRoundRobinScheduler() {}

/**
 * Used to select a task from all the Virtual Accelerator Queues in the system.
 * Selects a virtual accelerator queue from which the accelThread is going to pop tasks
 * Inputs: Vector with all virtual accelerator queues for a specific physical accelerator
 *        and the physical accelerator for that VAQs.
 * Outputs: a utils_queue_s from which selectTask is going to pop.
 **/
utils_queue_s *VineRoundRobinScheduler::selectVirtualAcceleratorQueue(accelThread *th)
{
    /*init variables*/
    vine_accel ** arrayAllVAQs = th->getAllVirtualAccels();
    vine_accel_s *phys = th->getAccelConfig().vine_accel;
    int numOfVAQs = th->getNumberOfVirtualAccels();

    size_t index;
    index = ++virtualQueueIndex[phys];
    return vine_vaccel_queue((vine_vaccel_s*)arrayAllVAQs[index%numOfVAQs]);
}
/**
 * Select task from the array returned from vine_accel_list
 * this array contains All the Virtual Accelerator Queues that exist in the system.
 * Inputs: Array with all Virtual Accelerators Queues of the system.
 * Outputs: A vine_task_msg_s (i.e. vine_task).
 **/
vine_task_msg_s *VineRoundRobinScheduler::selectTask(accelThread *th)
{
    int allVaqs;
    /*Task poped from that VAQ*/
    vine_task_msg_s *vine_task;
    /*VAQ that the task is going to be poped*/
    utils_queue_s *selectedVAQ;
    //cout<<"Array size"<< arraySize<<endl;
    for (allVaqs = 0; allVaqs != th->getNumberOfVirtualAccels(); allVaqs++)
    {
        /*Call the select VAQ to find the VAQ that we are going to pop task*/
        selectedVAQ = selectVirtualAcceleratorQueue(th);
        /*If there is no VAQ*/
        if (!selectedVAQ)
            continue;
        /*take the task from that queue*/
        vine_task = (vine_task_msg_s *)utils_queue_pop(selectedVAQ);
        /*Pop a task*/
        if (vine_task)
            return vine_task;
    }
    return 0;
}

REGISTER_SCHEDULER(VineRoundRobinScheduler)

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
#include "CPUaccelThread.h"
#include "definesEnable.h"
#include "utils/timer.h"
using namespace std;

CPUaccelThread::CPUaccelThread(vine_pipe_s * v_pipe, AccelConfig & conf):accelThread(v_pipe, conf) {}
CPUaccelThread::~CPUaccelThread() {}
/*initializes the CPU accelerator*/
bool CPUaccelThread::acceleratorInit(){
    cout<<"CPU initalization done."<<endl;
    cout<<"=============================="<<endl;
    return true;
}
/*Releases the CPU accelerator*/
void CPUaccelThread::acceleratorRelease(){
    cout<<"CPU released."<<endl;
}

/**
 * Transfer Function Implementations
 */

/**
 * TODO: UGLY HACK TO AVOID API change PLZ FIXME
 */
extern vine_pipe_s *vpipe_s;

void Host2CPU(vine_task_msg_s *vine_task, vector<void *> &ioHD)
{
	int in;

	/*Start timer for Host2CPU*/
	utils_breakdown_advance(&(vine_task->breakdown),"MemCpy_H2C");

	#ifdef BREAKDOWNS_CONTROLLER
	/*meassure time*/
	chrono::time_point<chrono::system_clock> startH2D, endH2D;
	/*start timer*/
	startH2D = chrono::system_clock::now();
	#endif

	#ifdef DATA_TRANSFER
	cout<<"Number of inputs: "<<vine_task->in_count<<endl;
	cout<<"Number of outputs: "<<vine_task->out_count<<endl;
	#endif

	for (in = 0; in < vine_task->in_count; in++)
	{
		vine_data *vinedata = vine_task->io[in].vine_data;
		ioHD.push_back(vine_data_deref(vinedata));
		#ifdef DATA_TRANSFER
		cout<<"Size of input " <<in<<" is: "<< vine_data_size(vine_task->io[in].vine_data)<<endl;
		#endif
	}
	for (int out = in; out < vine_task->out_count + in; out++)
	{
		vine_data *vinedata = vine_task->io[out].vine_data;
		ioHD.push_back(vine_data_deref(vinedata));
		#ifdef DATA_TRANSFER
		cout<<"Size of output " <<out<<" is: "<< vine_data_size(vine_task->io[out].vine_data)<<endl;
		#endif

	}
	#ifdef BREAKDOWNS_CONTROLLER
	/*stop timer*/
	endH2D = chrono::system_clock::now();
	/*duration*/
	cout << "---------------Breakdown inside Controller-----------------" << endl;
	chrono::duration<double, nano> elapsed_H2D = endH2D - startH2D;
	cout << "HOST2CPU took :" << elapsed_H2D.count() << " nanosec." << endl;
	#endif

	/* Stop timer for Host to CPU - Start timer for kernel execution in CPU*/
	utils_breakdown_advance(&(vine_task->breakdown), "Kernel Execution_CPU");
}
/* Cuda Memcpy from Device to host*/
void CPU2Host(vine_task_msg_s *vine_task, vector<void *> &ioDH)
{
	/*Stop timer for kernel execution in CPU - Start timer for CPU to Host*/
	utils_breakdown_advance(&(vine_task->breakdown), "MemCpy_C2H");

	#ifdef BREAKDOWNS_CONTROLLER
	/*meassure time*/
	chrono::time_point<chrono::system_clock> startD2H, endD2H;
	/*start timer*/
	startD2H = chrono::system_clock::now();
	#endif

	for (int out = vine_task->in_count; out < vine_task->out_count + vine_task->in_count; out++)
	{
		if (out == vine_task->out_count + vine_task->in_count-1)
		{
			/*Mark the state as completed*/
			vine_task->state = task_completed;

			/*Stop timer for CPU to Host - Start timer for memFree*/
			utils_breakdown_advance(&(vine_task->breakdown), "MemFree");
		}
		vine_data_mark_ready(vpipe_s, vine_task->io[out].vine_data);
	}
	utils_timer_set(vine_task->stats.task_duration,stop);
	#ifdef BREAKDOWNS_CONTROLLER
	/*stop timer*/
	endD2H = chrono::system_clock::now();
	/*duration*/
	chrono::duration<double, nano> elapsed_D2H = endD2H - startD2H;
	cout << "CPU2HOST took :" << elapsed_D2H.count() << " nanosec." << endl;
	#endif
}

/* Free Device memory */
void CPUMemFree(vector<void *> &io)
{
	#ifdef BREAKDOWNS_CONTROLLER
	/*meassure time*/
	chrono::time_point<chrono::system_clock> startFree, endFree;
	/*start timer*/
	startFree = chrono::system_clock::now();
	/*stop timer*/
	endFree = chrono::system_clock::now();
	/*duration*/
	chrono::duration<double,nano> elapsed_Free = endFree - startFree;
	cout << "Free took :" << elapsed_Free.count() << " nanosec" << endl;
	cout << "---------------End Breakdown----------------" << endl;
	#endif

}

REGISTER_ACCEL_THREAD(CPUaccelThread)

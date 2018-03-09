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
#include <pthread.h>
#include <iostream>
#include <vector>
#include "vine_pipe.h"
#include "accelThread.h"
#include "VineLibMgr.h"
#include "definesEnable.h"
#include <chrono>
#include <ctime>
#include <unistd.h>
#include <mutex>

using namespace ::std;

void *workFunc(void *thread);

//Constructor
accelThread::accelThread(vine_pipe_s * v_pipe, AccelConfig & accelConf):
	enable_mtx(PTHREAD_MUTEX_INITIALIZER), v_pipe(v_pipe), accelConf(accelConf)
{
	allVirtualAccels = 0;
	numOfConcurrentVAQs = 0;
	revision = 0;
	stopExec = 0;
}
/*Returns all VAQs of the system (with ANY type)*/
vine_accel ** accelThread::getAllVirtualAccels()
{
	return allVirtualAccels;
}

/*Returns the number of all VAQs in the system*/
int accelThread::getNumberOfVirtualAccels()
{
	return numOfConcurrentVAQs;
}

/*Returns the current configuration*/
AccelConfig & accelThread::getAccelConfig()
{
	return accelConf;
}


/*Executes the kernel*/
void accelThread::executeHostCode(void* functor, vine_task_msg_s* vine_task)
{
	(*((VineFunctor **)functor))(vine_task);
}

AccelConfig::JobPreference accelThread::getJobPreference()
{
	return accelConf.job_preference;
}

AccelConfig::JobPreference accelThread::getInitialJobPreference()
{
	return accelConf.initial_preference;
}

void accelThread::setJobPreference(AccelConfig::JobPreference jpref)
{
	accelConf.job_preference = jpref;
	if(jpref == AccelConfig::NoJob)
		disable();
	else
		enable();
}

void accelThread::enable()
{
	pthread_mutex_unlock(&enable_mtx);
}

void accelThread::disable()
{
	pthread_mutex_trylock(&enable_mtx);
}

/*Spawn threads and sprecifies in which core an accel thread is going to run*/
void accelThread::spawn()
{
	pthread_create(&pthread, NULL, workFunc, this);
	/*specify in which core (from the .config file) the acceleThread is going to run*/
	pthread_setaffinity_np(pthread, sizeof(cpu_set_t), accelConf.affinity.getSet());

}

/*Stops the execution*/
void accelThread::terminate()
{
	stopExec = 1;
	/*Add a fake task to sleep the current thread */
	vine_pipe_add_task(v_pipe,accelConf.vine_accel->type);
	setJobPreference(AccelConfig::AnyJob);
}

void accelThread::joinThreads()
{
	pthread_join(pthread, 0);
}

vine_pipe_s * accelThread :: getPipe()
{
	return v_pipe;
}

void accelThread::printOccupancy() {
	/*Used for GPUs only*/
}

/*Destructor*/
accelThread::~accelThread()
{
	free (allVirtualAccels);
}
void printAcceleratorType(GroupConfig * group)
{
        map<int,int> acceleratorCount;
        for(auto accel : group->getAccelerators())
        {
                acceleratorCount[accel.second->accelthread->getJobPreference()]++;
                //cout<<"Accel type: "<<accel.second->accelthread->getJobPreference()<<endl;
        }
        for (int i=0; i<4;i++)
        {
                cout<<" "<<acceleratorCount[i];
        }
	cout<<endl;
}

void my_gettime(struct timeval * tp,struct timezone *tz)
{
        static timeval start;
        gettimeofday(tp,tz);

        if(!start.tv_sec)
                start = *tp;
        tp->tv_sec -= start.tv_sec;
        tp->tv_usec -= start.tv_usec;
}

#define gettimeofday my_gettime

/*Function that handles the Virtual accelerator queues */
void *workFunc(void *thread)
{
#ifdef POWER_TIMER
	/*used for power consumption*/
	struct timeval power_tv1, power_tv2;
#endif
	accelThread *th = (accelThread *)thread;
	vine_accel_type_e mod_counter;	// Type of modified counter by updateVirtAccels

	/*create a vine task pointer*/
	vine_task_msg_s *vine_task;

	/*The scheduling policy to be used*/
	Scheduler * selectedScheduler;
	selectedScheduler = th->accelConf.group->getScheduler();

#ifdef BREAKDOWNS_CONTROLLER
	/*meassure time*/
	chrono::time_point<chrono::system_clock> startExecT, endExecT, startSchTask, endSchTask, start, end;
	double sumDuration; /* Summary of time spend in  task scheduling decisions*/
#endif

	/*Initiliaze the accelerator*/
	if (!th->acceleratorInit())
	{
		cerr <<"Exit!!"<<endl;
		return 0;
	}

	/*Iterate until ctrl+c is pressed*/
	while (!th->stopExec)
	{
		if(th->accelConf.job_preference == AccelConfig::NoJob)
			pthread_mutex_lock(&(th->enable_mtx));

		mod_counter = th->updateVirtAccels();

#ifdef BREAKDOWNS_CONTROLLER
		startSchTask = chrono::system_clock::now();
#endif
		/*iterate in the Virtual accelerator vector, which contains all the VAQs*/
		vine_task = selectedScheduler->selectTask(th);

#ifdef BREAKDOWNS_CONTROLLER
		endSchTask = chrono::system_clock::now();
		chrono::duration<double> elapsedSchTaskDec = endSchTask - startSchTask;
#endif

		//cout<<"Accelerator job preference : "<<th->getJobPreference();
		/*If there is a vine_task_msg */
		if (vine_task)
		{
			/*Name of task*/
			char *taskName = ((vine_object_s) ( ((vine_proc_s*)(vine_task->proc))->obj)).name;
			/*Type of the accelerator thread*/
			vine_accel_type_e accelThreadType = (vine_accel_type_e)(th->accelConf).vine_accel->type;
			/*kernel of the selected task*/
			vine_proc_s *proc;
			//			cerr<<"!!!!!"<<(void*)vine_task<<endl;
			//			cerr<<"!!!!!"<<(vine_vaccel_s*)(vine_task->accel)<<endl;
			/*If VAQ is ANY*/
			if (((vine_vaccel_s*)(vine_task->accel))->type == ANY)
			{
				proc = (vine_proc_s *) vine_proc_get(accelThreadType, taskName);
			}
			else
			{
				proc = (vine_proc_s *)vine_task->proc;
			}
#ifdef BREAKDOWNS_CONTROLLER
			sumDuration = sumDuration + elapsedSchTaskDec.count();
#endif
			/* Call the appropriate kernel from the specified .so*/
			if (!proc)
			{//if there does NOT exist the specified kernel
				cerr << __FILE__ << " The function does not exist" << endl;
				cerr << "check in application in <vine_proc_get> the accelType and "
					"function name" << endl;
				vine_task->state = task_failed;
			}
			else
			{//If there exists the kernel that is asked from the task
				if (vine_task->accel == 0)
				{//Check if the accelerator of the task is NULL
					cerr <<__FILE__<<"-"<< "Accelerator is NULL! In function " << proc->obj.name << endl;
				}
				else
				{//The kernel exists and the accelerator is NOT NULL
					utils_breakdown_instance_set_paccel(&(vine_task->breakdown),th->accelConf.vine_accel);
#ifdef BREAKDOWNS_CONTROLLER
					/*start timer*/
					startExecT = chrono::system_clock::now();
#endif
#ifdef POWER_TIMER
					vine_vaccel_s * execVAQ = (vine_vaccel_s *)vine_task->accel;
					//job priority : userfacing = 1, batch = 0
					int jobPriority = vine_vaccel_get_job_priority(execVAQ);
					gettimeofday(&power_tv1, NULL);
					long int powerStartedAt = power_tv1.tv_sec * 1000000 +  power_tv1.tv_usec ;
#endif
					vine_object_ref_inc(&(vine_task->obj));
					/*Execute the kernel specified from the task*/
					th->executeHostCode(vine_proc_get_code(proc, 0),vine_task);
					selectedScheduler->postTaskExecution(vine_task);
					vine_object_ref_dec(&(vine_task->obj));

#ifdef SM_OCCUPANCY
					th->printOccupancy();
#endif

#ifdef POWER_TIMER
					gettimeofday(&power_tv2, NULL);

					long int powerEndedAt = power_tv2.tv_sec * 1000000 + power_tv2.tv_usec ;
					cout<<"TIME (POWER): start : "<<powerStartedAt<<" , end : "<<powerEndedAt<<" id: "<<vine_vaccel_get_cid(execVAQ);
					printAcceleratorType(th->accelConf.group);

#endif
#ifdef BREAKDOWNS_CONTROLLER
					/*stop timer*/
					endExecT = chrono::system_clock::now();
					/*duration*/
					chrono::duration<double, nano> elapsed_ExecT = endExecT - startExecT;
					cerr << "Overall execution time (inside controller): " << elapsed_ExecT.count() << " nanosec."<< endl;
					cerr << "-----------------------------------------------" << endl;
#endif
				}
			}
		}
		else
		{	// Scheduler did not pick any task, put back the reference.
			vine_pipe_add_task(th->v_pipe,mod_counter);
			// Note: This might be a good place to add a back-off time.
		}
	}
	/*Release the accelerator*/
	th->acceleratorRelease();
	return 0;
}
/*Search for new VAQs with ANY queues*/
vine_accel_type_e accelThread::updateVirtAccels()
{
	//	if(allVirtualAccels)
	//		vine_accel_list_free(allVirtualAccels);
	/*Temporary vine_accel arrays*/
	vine_accel **iterator, **elementHolder;
	/*Reduced semaphore of a specific accelarator type*/
	vine_accel_type_e reducedSemAccel;
	/*Sleep untill a task arrives*/
	reducedSemAccel = vine_pipe_wait_for_task_type_or_any(v_pipe, accelConf.vine_accel->type);
	//usleep(10000);
	/*Get VAQs with type = ANY (i.e. all VAQs that exist in the system)*/
	numOfConcurrentVAQs = vine_accel_list(ANY, 0, &allVirtualAccels);
	int off = 0;
	for (int i=0; i<numOfConcurrentVAQs; i++)
	{
		if(utils_queue_used_slots(vine_vaccel_queue((vine_vaccel_s*)allVirtualAccels[i])))
		{
			allVirtualAccels[off] = allVirtualAccels[i];
			off++;
		}
	}
	numOfConcurrentVAQs = off;
	/*Initialize tmp arrays with all VAQs that exist in the system*/
	iterator = allVirtualAccels;
	elementHolder = allVirtualAccels;

	/*Number of VAQs of type ANY and accelthread type*/
	int anyVAQCount = 0;
	/*
	 * Iterate through all &allVirtualAccels, select VAQS of type ANY
	 * and VAQs for the current accelThread Type
	 */

	//cout<<this<<"  Num of All VAQs: "<<numOfConcurrentVAQs<<" , VAQ type: "<< ((vine_vaccel_s*)*iterator)->type<<endl;
	for (int q = 0 ; q < numOfConcurrentVAQs ; q++)
	{
		/*Type of VAQ is ANY or this VAQ can be executed from the current accel thread*/
		if ( (((vine_vaccel_s*)*iterator)->type == reducedSemAccel) )
		{
			/*Store the relevant VAQs type (i.e. ANY and accelThread type)*/
			*elementHolder = *iterator;
			/*go to the next cell of the temporary array*/
			elementHolder++;
			/*increase the number of VAQs that are returned*/
			anyVAQCount++;
		}
		/*Go to the next cell of tmp1*/
		iterator++;
		//cout <<"Number of VAQs "<<anyVAQCount<<" for thread type: "<<accelConf.vine_accel->type<<endl;
	}
	numOfConcurrentVAQs = anyVAQCount;

	return reducedSemAccel;
}

Factory<accelThread,vine_pipe_s *, AccelConfig &> threadFactory;

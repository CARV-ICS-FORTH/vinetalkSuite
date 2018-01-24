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
#ifndef ACCELTHREAD
#define ACCELTHREAD
using namespace::std;
#include <pthread.h>
#include <vector>
#include "vine_pipe.h"
#include "timers.h"
#include <typeinfo>
#include <map>
#include <ostream>
class accelThread;
#include "Config.h"
#include "Factory.h"
#include "Scheduler.h"
#define VECTOR_WITH_DURATIONS_SIZE 30
class accelThread {
	public:
		accelThread(vine_pipe_s * v_pipe, AccelConfig & conf);
		void spawn();
		void terminate();
		void joinThreads();
		vine_pipe_s * getPipe();
		virtual ~accelThread();
		/* Function that initializes an accelerator */
		virtual bool acceleratorInit() = 0;
		/* Function that resets an accelerator */
		virtual void acceleratorRelease() = 0;
		virtual void printOccupancy();
		/* signal to exit*/
		volatile int stopExec;
		/*Getters*/
		/*returns all VAQs (without considering type) of the system*/
		vine_accel ** getAllVirtualAccels();
		/*returns the number of all VAQs (without considering type) of the system*/
		int getNumberOfVirtualAccels();
		/*returns Accelarator configuration specified from the configuration file*/
		AccelConfig & getAccelConfig();
		/*executes a task*/
		virtual void executeHostCode(void* functor, vine_task_msg_s* task);
		AccelConfig::JobPreference getJobPreference();
		AccelConfig::JobPreference getInitialJobPreference();
		void setJobPreference(AccelConfig::JobPreference jpref);
		void enable();
		void disable();
	private:
		pthread_mutex_t enable_mtx;
		/* thread*/
		pthread_t pthread;
		/* VineTalk Pipe */
		vine_pipe_s * v_pipe;
		/* Accelerator configuration*/
		AccelConfig &accelConf;
		/*function to check if there are new VA queues*/
		vine_accel_type_e updateVirtAccels();
		size_t revision;
		/*function that performs the execution per thread*/
		friend void *workFunc(void *thread);
		/*vector with timers*/
		vector<Timers_s> statsVector;
		/*Number of virtual accelerators for a specific physical accelerator returned
		 * from vine_accel_list()*/
		int numOfConcurrentVAQs;
		/*Array with all VAQs per accelerator type (returned from vine_accel_list)*/
		vine_accel **allVirtualAccels;
};

extern Factory<accelThread,vine_pipe_s *, AccelConfig &> threadFactory;

/**
 * Helper function to register Accelerator Threads
 * Must be put in a cpp file.
 */
#define REGISTER_ACCEL_THREAD(CLASS)\
	static Registrator<accelThread,CLASS,vine_pipe_s *, AccelConfig &> reg(threadFactory);
#endif

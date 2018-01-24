/*
 * Copyright 2018 Foundation for Research and Technology - Hellas
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at 
 * http://www.apache.org/licenses/LICENSE-2.0 [1] [1] 
 * 
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
 * See the License for the specific language governing permissions and 
 * limitations under the License. 
 * 
 * Links: 
 * 
 * [1] http://www.apache.org/licenses/LICENSE-2.0 [1] 
 */
#ifndef VINE_TALK_TYPES_HEADER
#define VINE_TALK_TYPES_HEADER
#include <sys/time.h>
#include <time.h>
/**
 * vine_accel: Accelerator descriptor.
 */
typedef void vine_accel;

/**
 * vine_proc: Process descriptor.
 */
typedef void vine_proc;

/**
 * Location of a vine_accel.
 */
typedef struct vine_accel_loc {
	/**< To be filled */
} vine_accel_loc_s;

/**
 * Vine Pipe instance
 */
typedef struct vine_pipe vine_pipe_s;

/**
 * Accelerator Statistics
 */
typedef struct vine_accel_stats {} vine_accel_stats_s;

typedef struct utils_timer_s
{
	struct timespec start;
	struct timespec stop;
}utils_timer_s;

/**
 * Accelerator State enumeration.
 */
typedef enum vine_accel_state {
	accel_failed, /**< Accelerator has failed. */
	accel_idle, /**< Accelerator is idle. */
	accel_busy /**< Accelerator is busy. */
} vine_accel_state_e;

/**
 * Vineyard Task Descriptor
 */
typedef void vine_task;

/**
 * Vine Task State enumeration.
 */
typedef enum vine_task_state_e {
	task_failed, /**< Task execution failed. */
	task_issued, /**< Task has been issued. */
	task_completed /**< Task has been completed. */
} vine_task_state_e;

/**
 * Vine Task Statistics
 */
typedef struct vine_task_stats {
	int task_id; /**< Unique among tasks of this instance */
	utils_timer_s task_duration;
} vine_task_stats_s;

/**
 * Accelerator type enumeration.
 * NOTE: If updated update types_map variable in vine_accel_types.c
 */
typedef enum vine_accel_type {
	ANY       = 0,   /**< Let Scheduler Decide                 */
	GPU       = 1,   /**< Run on GPU with CUDA                 */
	GPU_SOFT  = 2,   /**< Run on CPU with software CUDA        */
	CPU       = 3,   /**< Run Native x86 code                  */
	SDA       = 4,   /**< Xilinx SDAaccel                      */
	NANO_ARM  = 5,   /**< ARM accelerator core from NanoStream */
	NANO_CORE = 6,   /**< NanoStreams FPGA accelerator         */
	VINE_ACCEL_TYPES /** End Marker                            */
} vine_accel_type_e;

typedef struct arch_alloc_s arch_alloc_s;

typedef struct vine_buffer_s vine_buffer_s;

typedef struct arch_alloc_s arch_alloc_s;

typedef struct async_meta_s async_meta_s;
#endif

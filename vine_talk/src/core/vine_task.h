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
#ifndef VINE_TASK_HEADER
#define VINE_TASK_HEADER
#include "core/vine_object.h"
#include "core/vine_proc.h"
#include "core/vine_buffer.h"

/**
 * Vineyard Task message.
 */
typedef struct vine_task_msg {
	vine_object_s     obj;
	vine_accel        *accel; /**< Accelerator responsible for this task */
	vine_proc         *proc; /**< Process id */
	vine_buffer_s     args; /**< Packed process arguments */
	int               in_count; /**< Number of input buffers */
	int               out_count; /**< Number of output buffers */
	vine_task_state_e state;
	vine_task_stats_s stats;
	vine_accel_type_e type;		/** Type of task at issue */
	utils_breakdown_instance_s breakdown;
	vine_buffer_s     io[]; /**< in_count+out_count pointers
	*                       to input and output
	* buffers*/
} vine_task_msg_s;

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

	vine_task_msg_s * vine_task_alloc(vine_pipe_s *vpipe,int ins,int outs);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif

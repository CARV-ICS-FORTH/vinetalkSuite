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
#include "vine_task.h"
#include "vine_pipe.h"
#include "utils/breakdown.h"
#include <stdlib.h>

vine_task_msg_s * vine_task_alloc(vine_pipe_s *vpipe,int ins,int outs)
{
	vine_task_msg_s * task;

	task = (vine_task_msg_s *)vine_object_register( &(vpipe->objs),
													VINE_TYPE_TASK,"Task",
				sizeof(vine_task_msg_s)+sizeof(vine_buffer_s)*(ins+outs));

	if(!task)
		return 0;

	task->in_count = ins;
	task->out_count = outs;
	task->args.vine_data = 0;

	utils_breakdown_instance_init(&(task->breakdown));

	return task;
}

VINE_OBJ_DTOR_DECL(vine_task_msg_s)
{
	vine_task_msg_s *_task = (vine_task_msg_s *)obj;
	vine_data_s * prev;
	int cnt;
	utils_breakdown_advance(&(_task->breakdown),"TaskFree");

	qsort(_task->io,_task->in_count+_task->out_count,sizeof(vine_buffer_s),vine_buffer_compare);
	prev = 0;
	for(cnt = 0 ; cnt < _task->in_count+_task->out_count ; cnt++)
	{
		if(prev != _task->io[cnt].vine_data)
		{
			prev = _task->io[cnt].vine_data;
 			vine_object_ref_dec(&(prev->obj));
		}
	}

	#ifdef BREAKS_ENABLE
	if(_task->breakdown.stats)
		utils_breakdown_end(&(_task->breakdown));
	#endif

	if(_task->args.vine_data)
		vine_object_ref_dec(_task->args.vine_data);

	// This check is necessary as some unit tests leave stats null
	// TODO: Fix this properly
	arch_alloc_free(obj->repo->alloc,obj);

}

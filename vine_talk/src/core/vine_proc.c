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
#include "vine_proc.h"
#include <string.h>

vine_proc_s* vine_proc_init(vine_object_repo_s *repo, const char *name,
							vine_accel_type_e type, const void *code,
							size_t code_size)
{
	vine_proc_s *proc =
	(vine_proc_s*)vine_object_register(repo, VINE_TYPE_PROC, name,
									   sizeof(vine_proc_s)+code_size);

	if(!proc)
		return 0;

	proc->type     = type;
	proc->users    = 0;
	proc->bin_size = code_size;
	utils_breakdown_init_stats(&(proc->breakdown));
	memcpy(proc+1, code, code_size);
	return proc;
}

size_t vine_proc_calc_size(const char *name, size_t code_size)
{
	return sizeof(vine_proc_s)+code_size;
}

int vine_proc_match_code(vine_proc_s *proc, const void *code, size_t code_size)
{
	if (code_size != proc->bin_size)
		return 0;
	return !memcmp(code, proc+1, code_size);
}

void* vine_proc_get_code(vine_proc_s *proc, size_t *code_size)
{
	if (code_size)
		*code_size = proc->bin_size;
	return proc+1;
}

int vine_proc_mod_users(vine_proc_s *proc, int delta)
{
	return __sync_fetch_and_add(&(proc->users), delta);
}

VINE_OBJ_DTOR_DECL(vine_proc_s)
{
}

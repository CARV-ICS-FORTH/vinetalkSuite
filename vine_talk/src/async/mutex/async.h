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
#ifndef ASYNC_HEADER
#define ASYNC_HEADER
#include "core/vine_object.h"
#include "pthread.h"
#include <semaphore.h>

struct async_meta_s
{
	#ifndef __cplusplus
	char padd;
	#endif
};

typedef struct
{
	pthread_mutex_t mutex;
	pthread_mutexattr_t attr;
}
async_completion_s;

typedef struct
{
	sem_t sem;
}async_semaphore_s;

typedef struct
{
	async_completion_s mutex;
	pthread_cond_t condition;
	pthread_condattr_t c_attr;
}async_condition_s;

#include "async_api.h"
#endif

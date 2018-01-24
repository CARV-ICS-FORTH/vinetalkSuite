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
#include "arch/alloc.h"
#include <stdlib.h>

arch_alloc_s arch_alloc_init(void *shm, size_t size)
{
	return NULL;
}

void* arch_alloc_allocate(arch_alloc_s alloc, size_t size)
{
	return malloc(size);
}

void arch_alloc_free(arch_alloc_s alloc, void *mem)
{
	free(mem);
}

void arch_alloc_exit(arch_alloc_s alloc) {}

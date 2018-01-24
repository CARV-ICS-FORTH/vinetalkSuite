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
#ifndef VINE_BUFFER_HEADER
#define VINE_BUFFER_HEADER

#include <string.h>

struct vine_buffer_s
{
	void * user_buffer;
	size_t user_buffer_size;
	void * vine_data;
};

#include "core/vine_data.h"

/**
 * Define a Vineyard buffer.
 * @param USER_POINTER Pointer to a valid memory location used for data.
 * @param BUFFER_SIZE Size of memory pointer by USER_POINTER in bytes.
 * @param copy Copy user data to buffer.
 */
#define VINE_BUFFER(USER_POINTER,BUFFER_SIZE)								\
{USER_POINTER,BUFFER_SIZE,0}

void vine_buffer_init(vine_buffer_s * buffer,void * user_buffer,size_t user_buffer_size,void * vine_data,int copy);

/**
 * Compare two vine_buffer_s objcets \c a \c b in terms of their vine_data pointer.
 *
 * @return Difference of the two objects, 0 if equal.
 */
int vine_buffer_compare(const void * a,const void * b);
#endif

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
/**
 * @brief
 * The Dynamic circular work-stealing deque WITHOUT the dynamic part ;)
 * (https://dl.acm.org/citation.cfm?id=1073974)
 *
 * NOTE: Also WITHOUT pop-front
 *
 * Removes the need for dynamic reallocation and constantly increasing
 * bottom and top
 *
 * @author  Foivos Zakkak <zakkak@ics.forth.gr>
 */

#include "queue.h"
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <limits.h>

#define COMPILER_BARRIER() asm volatile ("" : : : "memory")

#ifdef __GNUC__
#define UNLIKELY(cond) __builtin_expect(cond, 0)
#define LIKELY(cond)   __builtin_expect(cond, 1)
#else /* ifdef __GNUC__ */
#define UNLIKELY(cond) (cond)
#define LIKELY(cond)   (cond)
#endif /* ifdef __GNUC__ */

utils_queue_s* utils_queue_init(void *buff)
{
	assert( !( UTILS_QUEUE_CAPACITY & (UTILS_QUEUE_CAPACITY - 1) ) );

	/* Zero memory */
	memset( buff, 0, sizeof(struct queue) );

	return (utils_queue_s*)buff;
}

unsigned int utils_queue_used_slots(utils_queue_s *q)
{
	register int used_slots;

	used_slots = q->bottom - q->top;

	if (used_slots < 0)
		used_slots += UINT16_MAX+1;

	return (unsigned int)used_slots;
}

void* utils_queue_pop(utils_queue_s *q)
{
	register uint16_t t, b;
	register int      i;
	void              *ret_val = 0;

	assert(q);

	/* Only one thief can succeed in the following critical section */
	t = q->top;
	b = q->bottom;

	/* If it is empty */
	if (b == t)
		return 0;

	/* Get the top element */
	i       = t & (UTILS_QUEUE_CAPACITY - 1);
	ret_val = q->entries[i];
	if ( __sync_bool_compare_and_swap(&q->top, t, t + 1) )
		return ret_val;

	return 0;
}

void* utils_queue_push(utils_queue_s *q, void *data)
{
	uint16_t b, t;
	int      i;

	assert(data);
	assert(q);

	b = q->bottom;
	t = q->top;

	int used_slots = b - t;

	if (used_slots < 0)
		used_slots += UINT16_MAX+1;

	/* If there is no more space */
	if (used_slots == UTILS_QUEUE_CAPACITY)
		return 0;

	i             = b & (UTILS_QUEUE_CAPACITY - 1);
	q->entries[i] = data;
	__sync_synchronize();
	q->bottom = b + 1;
	/* printf("b=%u t=%u\n", ++b, t);
	 * assert(((b >> 7) == (t >> 7)) || ((b & (UTILS_QUEUE_CAPACITY-1)) <= (t & (UTILS_QUEUE_CAPACITY)))); */

	return data;
}

void* utils_queue_peek(utils_queue_s *q)
{
	register uint16_t t, b;
	register int      i;

	/* Only one thief can succeed in the following critical section */
	t = q->top;
	b = q->bottom;

	/* If it is empty */
	if (b == t)
		return 0;

	i = t & (UTILS_QUEUE_CAPACITY - 1);

	return q->entries[i];

}

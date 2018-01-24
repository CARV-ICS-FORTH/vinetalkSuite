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
#ifndef UTILS_SPINLOCK_HEADER
#define UTILS_SPINLOCK_HEADER
#include <assert.h>

#if __x86_64__
typedef volatile uint64_t utils_spinlock;

#else /* if __x86_64__ */
typedef volatile uint32_t utils_spinlock;

#endif /* if __x86_64__ */

/**
 * Initialize \c lock as unlocked.
 *
 * \param lock utils_spinlock to be initialized as unlocked.
 */
static inline void utils_spinlock_init(utils_spinlock *lock)
{
	*lock = 0;
}

/**
 * Lock \c lock.
 *
 * Will attempt to lock \c lock.
 * Will spin until succesfull.
 *
 * \param lock utils_spinlock instance to be locked.
 */
static inline void utils_spinlock_lock(utils_spinlock *lock)
{
	do {
		while (*lock)
			; /* Maybe add relax()? */
		if ( __sync_bool_compare_and_swap(lock, 0, 1) )
			break; /* We got it */
	} while (1); /* Try again */
}

/**
 * Will unlock \c lock that was previously locked.
 * \note Calling utils_spinlock_unlock on an unlocked utils_spinlock
 * instance is an error.
 * \param lock utils_spinlock instance to be unlocked.
 */
static inline void utils_spinlock_unlock(utils_spinlock *lock)
{
	assert(*lock); /* Attempting to unlock twice */
	__sync_fetch_and_and(lock, 0);
}

#endif /* ifndef UTILS_SPINLOCK_HEADER */

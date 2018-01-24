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
#ifndef UTILS_COMPAT_HEADER
#define UTILS_COMPAT_HEADER

/**
 * Placeholder struct to be used instead of an empty struct.
 *
 * This struct is necessary as a c struct is 0 bytes*, whereas a
 * c++ struct is 1 byte big**, resulting in problematic c<->c++ interaction.
 *
 * https://gcc.gnu.org/onlinedocs/gcc/Empty-Structures.html#Empty-Structures
 * ** Could not find reference, but seems to be defined as 'non zero'.
 */
typedef struct
{
	unsigned long nothing;
}utils_compat_empty_s;

#endif

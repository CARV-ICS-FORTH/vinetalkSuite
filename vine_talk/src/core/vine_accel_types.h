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
#ifndef VINE_ACCEL_TYPES_HEADER
#define VINE_ACCEL_TYPES_HEADER
#include "vine_talk_types.h"
#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/**
 * Convert a vine_accel_type_e value to a human readable string.
 * If \c type not a valid vine_accel_type_e value NULL is returned.
 * NOTE: This function should not be used in critical paths!
 *
 * @return A cahracter representation for the given \c type,NULL on error.
 */
const char * vine_accel_type_to_str(vine_accel_type_e type);

/**
 * Convert a string to the matching vine_accel_type_e value.
 * \c type will be compared ignoring capitalization with the string in
 * types_map variable in vine_accel_types.c.
 *
 * NOTE: This function should not be used in critical paths!
 *
 * @return A value from vine_accel_type_e, if no match is found returns
 * VINE_ACCEL_TYPES
 */
vine_accel_type_e vine_accel_type_from_str(const char * type);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif

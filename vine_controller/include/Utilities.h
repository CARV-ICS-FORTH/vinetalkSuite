/*
 * Copyright 2018 Foundation for Research and Technology - Hellas
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0 [1] [1]
 *
 * Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 *  implied.
 * See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * Links:
 *  ------
 * [1] http://www.apache.org/licenses/LICENSE-2.0 [1] 
*/
#ifndef VINE_CONTROLLER_UTILITIES_HEADER_FILE
	#define VINE_CONTROLLER_UTILITIES_HEADER_FILE
	#include <map>
	#include <string>

	/**
	 * Convert arguement list of the following format to a map:
	 *
	 * key1:val1,key2:val2
	 *
	 * Resulting map has keys and values in lowercase.
	 * Keys and values are assumed to be words
	 * (i.e a key/value can not contain whitespace)
	 */
	std::map<std::string,std::string> decodeArgs(std::string args);

	/*
	 * Trim whitespace from word contained in \c word.
	 *
	 * Removes whitespace from the begining and end of \c word.
	 */
	std::string trim(const std::string &s);
#endif

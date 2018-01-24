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
#include "Utilities.h"
#include <sstream>
#include <algorithm>

std::map<std::string,std::string> decodeArgs(std::string args)
{
	std::string k,v;
	std::istringstream iss(args);
	std::map<std::string,std::string> kv;

	do
	{
		std::getline(iss,k,':');
		std::getline(iss,v,',');
		if(iss)
		{
			k = trim(k);
			std::transform(k.begin(), k.end(), k.begin(), ::tolower);
			v = trim(v);
			kv[k] = v;
		}
	}while(iss);

	return kv;
}

std::string trim(const std::string &s)
{
	auto wsfront=std::find_if_not(s.begin(),s.end(),[](int c){return std::isspace(c);});
	auto wsback=std::find_if_not(s.rbegin(),s.rend(),[](int c){return std::isspace(c);}).base();
	return (wsback<=wsfront ? std::string() : std::string(wsfront,wsback));
}

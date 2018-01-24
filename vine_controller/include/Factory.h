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
#ifndef FACTORY_HEADER
using namespace::std;
#define FACTORY_HEADER
#include <cxxabi.h>
#include <ostream>
#include <iostream>
#include <map>

template <class T,typename... CTOR_ARGS>
class Factory
{
	typedef T * (Constructor)(CTOR_ARGS...);
	public:
		Factory()
		{
		}
		void registerType(const type_info & type,Constructor * ctor)
		{
			char * type_c = abi::__cxa_demangle(type.name(),0,0,0);
			prototypes[type_c] = (void*)ctor;
			free(type_c);
		}
		T * constructType(string type,CTOR_ARGS... args)
		{
			char * type_c = abi::__cxa_demangle(typeid(T).name(),0,0,0);
			type += type_c;
			free(type_c);
			if(prototypes.count(type) == 0)
				return 0;
			try
			{
				return ((Constructor*)(prototypes)[type])(args...);
			}catch(exception & e)
			{
				return 0;
			}
		}
		std::vector<string> getTypes()
		{
			std::vector<string> types;
			for(auto itr = prototypes.begin() ; itr != prototypes.end() ; itr++)
			{
				types.push_back(itr->first);
			}

			return types;
		}
	private:
		map<string,void * > prototypes;
};

template <class B,class T,typename... CTOR_ARGS>
class Registrator
{
	public:
		Registrator(Factory<B, CTOR_ARGS... > & factory)
		{
			factory.registerType(typeid(T),Registrator<B,T,CTOR_ARGS...>::construct);
		}

		static B * construct(CTOR_ARGS... args)
		{
			return new T(args...);
		}
};

#endif

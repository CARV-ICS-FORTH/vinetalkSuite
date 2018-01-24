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
#include "Config.h"
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iostream>
#include <thread>
#include <algorithm>
#include <regex>
#include "../include/Formater.h"
#include "../include/Scheduler.h"

string trim(string &str)
{
    size_t start = str.find_first_not_of("\t ");
    size_t end = str.find_last_not_of("\t ");
    if(start == string::npos)
        start = 0;
    if(end != string::npos)
        end += 1;
    return str.substr(start,end);
}

template <typename M, typename V>

void MapToVec( const  M & m, V & v )
{
    for( typename M::const_iterator it = m.begin(); it != m.end(); ++it )
    {
        v.push_back( it->second );
    }
}

int GroupConfig::groupCount = 0;

GroupConfig::GroupConfig(std::string name,Scheduler * sched)
: name(name), scheduler(sched)
{
	scheduler->setGroup(this);
    groupId = groupCount;
    groupCount ++;
}

int GroupConfig :: getCount()
{
	return groupCount;
}

std::string GroupConfig :: getName() const
{
	return name;
}

int GroupConfig :: getID() const
{
	return groupId;
}

void GroupConfig :: addAccelerator(AccelConfig * accel)
{
	accelerators[accel->name] = accel;
	accel->group = this;
}

const map<string,AccelConfig*> & GroupConfig :: getAccelerators() const
{
	return accelerators;
}

Scheduler * GroupConfig :: getScheduler()
{
	return scheduler;
}

    AccelConfig :: AccelConfig()
: line(0) , type(VINE_ACCEL_TYPES), vine_accel(0), accelthread(0), group(0)
{
}

CpuSet :: CpuSet()
{
    CPU_ZERO(&cpu_set);
}

void CpuSet :: setAll()
{
    for(unsigned int core = 0 ; core < std::thread::hardware_concurrency() ; core++)
    {
        CPU_SET(core,&cpu_set);
    }
}

void CpuSet :: clearAll()
{
    CPU_ZERO(&cpu_set);
}

void CpuSet :: set(int core)
{
    if(core >= 0)
    {
        if(core >= (int)std::thread::hardware_concurrency())
            throw new runtime_error("Setting non existing core "+ std::to_string(core));
        CPU_SET(core,&cpu_set);
    }
    else
        setAll();
}

void CpuSet :: clear(int core)
{
    if(core >= 0)
    {
        if(core >= (int)std::thread::hardware_concurrency())
            throw new runtime_error("Clearing non existing core "+ std::to_string(core));
        CPU_CLR(core,&cpu_set);
    }
    else
        clearAll();
}

cpu_set_t * CpuSet :: getSet()
{
    return &cpu_set;
}

Config :: Config(string config_file)
{
    ifstream ifs(config_file);
    string line;
    string token;
    int cline = 0;
	bool comment_mode = false;
    if(!ifs)
        throw new runtime_error("Config file \""+config_file+"\" could not be read!");

    /*read the configuration file*/
    while(getline(ifs,line))
    {
        line = trim(line);
        istringstream iss(line);
        cline++;
        iss >> token;

		if(line[0] == '#' || !line[0] || (line[0] == '/' && line[1] =='/'))
            continue;	// Comment or empty

		if(line[0] == '/' && line[1] =='*')
		{
			comment_mode = true;
		}

		if(comment_mode)
		{
			if(line[0] == '*' && line[1] =='/')
				comment_mode = false;
			continue;
		}

        /*Path of .so files*/
        if(!token.compare(0,4,"path"))
        {
			/* read space separated paths*/
			while(iss)
			{
				iss >> token;
				if(iss)
					paths.push_back(token);
			}
        }
        /*Accelerators of the system*/
        else
            if(!token.compare(0,5,"accel"))
            {
                AccelConfig accel_conf;
                accel_conf.line = cline;
                accel_conf.config = this;
                iss >> token;	// Type

                accel_conf.type = vine_accel_type_from_str(token.c_str());
                accel_conf.type_str = token;
                if(accel_conf.type == VINE_ACCEL_TYPES)
                    throw new runtime_error("Unknown accelerator type \'"+token+"\' at line " + to_string(cline));

                iss >> accel_conf.name;
				iss >> accel_conf.affinity;
				iss >> token;	// Read JobPreference

				std::transform(token.begin(), token.end(), token.begin(), ::tolower);

				if(token == "nojob")
				{
					accel_conf.job_preference = AccelConfig::NoJob;	
					accel_conf.initial_preference = AccelConfig::NoJob;
				}
				else if(token == "userjob")
				{
					accel_conf.job_preference = AccelConfig::UserJob;
					accel_conf.initial_preference = AccelConfig::UserJob;
				}
				else if(token == "batchjob")
				{
					accel_conf.job_preference = AccelConfig::BatchJob;
					accel_conf.initial_preference = AccelConfig::BatchJob;
				}
				else if(token == "anyjob")
				{
					accel_conf.job_preference = AccelConfig::AnyJob;
					accel_conf.initial_preference = AccelConfig::AnyJob;
				}
				else
					throw new runtime_error("Parse error at line " + to_string(cline) + " expected a JobPreference(NoJob,UserJob,BatchJob,AnyJob)");

                if(!iss)
                    throw new runtime_error("Parse error at line " + to_string(cline));

                getline(iss,accel_conf.init_params);

                getAccelerator(accel_conf.name) = new AccelConfig(accel_conf);
            }
        /*Selected scheduler*/
            else
                if(!token.compare(0,5,"sched"))
                {
                    string sname;
                    string args;
                    iss >> sname;		// Scheduler name
                    iss >> token;		// Scheduler class
                    getline(iss,args);	// Scheduler Args

                    if(schedulers.count(token))
                        throw new runtime_error("Scheduler "+token+ "redeclared at line " + to_string(cline));
					schedulers[sname] = schedulerFactory.constructType(token,args);
                    if(!schedulers[sname])
                        throw new runtime_error("Could not construct a "+sname+" "+token+ " at line " + to_string(cline));
                    cout<<"Name of scheduler: "<<sname<<"  "<<(void*)schedulers[sname]<<endl;
                }
        /*Accelerator's group*/
                else
                    if(!token.compare(0,5,"group"))
                    {
                        GroupConfig * group_conf;
                        string gname;
                        iss >> gname; // Group name
                        if(groups.count(gname))
                            throw new runtime_error("Duplicate definition of group "+token+ " line " + to_string(cline));

                        iss >> token;
						Scheduler * sched = schedulers.at(token);

						if(!sched)
                            throw new runtime_error("Group scheduler "+token+" not declared, line " + to_string(cline));

						group_conf = new GroupConfig(gname,sched);
						groups[gname] = group_conf;

						while(iss >> token)
                        {
							std::vector<AccelConfig*> avec = getAccelerators(token);
							if(avec.size() == 0)
								throw new runtime_error("Expression "+token+" does not match any accelerator, used in group " + gname + " line " + to_string(cline));
							std::for_each(avec.begin(),avec.end(),
										  [group_conf](AccelConfig* accel_conf){group_conf->addAccelerator(accel_conf);});


                        }
                    }
                    else
                        cerr << "Ignoring \'" << line << "\'" << endl;
    }
    // Sanity checks
    string orphans = "";
    for(map<std::string,AccelConfig*>::iterator itr = accelerators.begin() ; itr != accelerators.end() ; itr++)
        if(!itr->second->group)
            orphans += "\t" + itr->first + " declared at line "+ to_string(itr->second->line) + "\n";
    if(orphans != "")
        throw new runtime_error("The following accelerators are not assigned to any group:\n"+orphans);
}

const vector<string> Config :: getPaths() const
{
    return paths;
}

const AccelConfig * Config :: getAccelerator(std::string accel) const
{
	return accelerators.at(accel);
}

AccelConfig *& Config :: getAccelerator(std::string accel)
{
	return accelerators[accel];
}

const vector<AccelConfig*> Config :: getAccelerators() const
{
    vector<AccelConfig*> vec;
    MapToVec(accelerators,vec);
    return vec;
}

#define MATCH [regex](const AccelConfig* p){return std::regex_match(p->name,regex);}

const vector<AccelConfig*> Config :: getAccelerators(std::string name) const
{
	std::vector<AccelConfig*> vec;
	MapToVec(accelerators,vec);
	std::vector<AccelConfig*> ret;

	std::regex regex(name);
	ret.resize(std::count_if(vec.begin(),vec.end(),MATCH));
	std::copy_if(vec.begin(),vec.end(),ret.begin(),MATCH);
	return ret;
}

const vector<Scheduler*> Config :: getSchedulers() const
{
    vector<Scheduler*> vec;
    MapToVec(schedulers,vec);
    return vec;
}

const vector<GroupConfig*> Config :: getGroups() const
{
	vector<GroupConfig*> vec;
	MapToVec(groups,vec);
	return vec;
}

Config :: ~Config()
{
    /*Free all maps*/
    for (map<string,AccelConfig*>::iterator it=accelerators.begin(); it!=accelerators.end(); ++it)
    {
        delete it->second;
    }

    for (map<string,Scheduler*>::iterator it=schedulers.begin(); it!=schedulers.end(); ++it)
    {
        delete it->second;
    }

    for (map<string,GroupConfig*>::iterator it=groups.begin(); it!=groups.end(); ++it)
    {
        delete it->second;
    }

}

istream & operator>>(istream & is,CpuSet & cpu_set)
{
    std::string fmt;
    int core;

    is >> fmt;
    std::replace( fmt.begin(), fmt.end(), ',', ' ');

    istringstream iss(fmt);

    do
    {
        iss >> core;
        if(iss)
            cpu_set.set(core);
    }while(iss);

    return is;
}

ostream & operator<<(ostream & os,CpuSet & cpu_set)
{
    std::vector<int> cores;

    for(unsigned int core = 0 ; core < std::thread::hardware_concurrency() ; core++)
    {
        if(CPU_ISSET(core,cpu_set.getSet()))
            cores.push_back(core);
    }

    os << Formater<int>("",",","",cores);
    return os;
}

ostream & operator<<(ostream & os,const AccelConfig * conf)
{
	CpuSet temp = conf->affinity;
	if(conf)
		os << vine_accel_type_to_str(conf->type) << ", " << conf->name << ", " << temp << ", \"" << conf->init_params << "\"";
	else
		os << "Missing AccelConfig specification!" << endl;
	return os;
}

ostream & operator<<(ostream & os,const GroupConfig * conf)
{
	std::string sep = "";
	os << conf->getName() << "{";
	for(auto accel : conf->getAccelerators())
	{
		os << sep << accel.first;
		sep = ",";
	}
	os << "}";
	return os;
}

ostream & operator<<(ostream & os,const Config & conf)
{
    os << "Repository Paths:" << endl;
    os << Formater<string>("\t","\n\t","\n",conf.getPaths()) << endl;
	os << "Accelerators:" << endl;
	os << Formater<AccelConfig*>("\t","\n\t","\n",conf.getAccelerators());
	os << "Groups:" << endl;
	os << Formater<GroupConfig*>("\t","\n\t","\n",conf.getGroups());
	return os;
}

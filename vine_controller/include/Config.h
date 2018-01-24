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
#ifndef CONFIG_FILE_HEADER
#define CONFIG_FILE_HEADER
using namespace::std;
#include <string>
#include <vector>
#include <map>
#include <vine_pipe.h>

struct accelThread;
struct GroupConfig;
class Config;

/*
 * CpuSet class
 */
class CpuSet
{
	public:
		CpuSet();
		void setAll();
		void clearAll();
		void set(int core);
		void clear(int core);
		cpu_set_t * getSet();
	private:
		cpu_set_t cpu_set;
};

/*
 * Struct that describes the accelerators of one node
 * 	the accelerators are described in a configuration file
 */
struct AccelConfig {
	typedef enum
	{
		NoJob,
		UserJob,
		BatchJob,
		AnyJob
	}JobPreference;
	
	AccelConfig();
	int line;			// Configure line
	string name;		// Accelerator Name
	string init_params;	// For Gpus PCI id
	string type_str;		// type string will fix this.
	vine_accel_type_e type;		// Accelerator type GPU, CPU, FPGA
	CpuSet affinity;		// Thread affinity
	//	int thread_core;		// Thread affinity
	vine_accel_s *vine_accel;	// Vinetalk accelerator object
	accelThread *accelthread;	// Accelerator Thread
	GroupConfig * group;		// Group where this accelerator belongs
	Config * config;        // System configuration
	JobPreference job_preference;	// What jobs this accelerator accepts
	JobPreference initial_preference;	// initial job preference that accelerator accepts
};

#include "Scheduler.h"

/*
 * Struct that describes a group of accelerators a group contains accelerators
 * with similar specs that are handled from the same Scheduler
 */
class GroupConfig
{
	public:
		GroupConfig (std::string name,Scheduler * sched);
		static int getCount();
		std::string getName() const;
		int getID() const;
		void addAccelerator(AccelConfig * accel);
		const map<string,AccelConfig*> & getAccelerators() const;
		Scheduler * getScheduler();
	private:
		std::string name;
		Scheduler * scheduler;					// Scheduler handling this group
		map<string,AccelConfig*> accelerators;	// Accelerators in this group
		/*Id per group*/
		int groupId;
		static int groupCount;
};

/*
 * Classs that describes the configuration file with
 *  1. The paths of .so files
 *  2. The accelerators
 *  3. The schedulers
 *  4. The groups of accels
 */
class Config
{
	public:
		Config(string config_file);
		string getRepository();
		const vector<string> getPaths() const;
		const AccelConfig * getAccelerator(std::string accel) const;
		AccelConfig *& getAccelerator(std::string accel);
		const vector<AccelConfig*> getAccelerators() const;
		const vector<AccelConfig*> getAccelerators(std::string regex) const;
		const vector<Scheduler*> getSchedulers() const;
		const vector<GroupConfig*> getGroups() const;
		~Config();
	private:
		vector<string>paths;				// Paths to load libraries from
		map<string,AccelConfig*> accelerators;	// Accelerators defined in config
		map<string,Scheduler*> schedulers;		// Schedulers defined in config
		map<string,GroupConfig*> groups;		// Groups defined in config
};

istream & operator>>(istream & is,CpuSet & cpu_set);
ostream & operator<<(ostream & os,CpuSet & cpu_set);
ostream & operator<<(ostream & os,const Config & conf);
ostream & operator<<(ostream & os,const AccelConfig * conf);
ostream & operator<<(ostream & os,const GroupConfig * conf);
#endif

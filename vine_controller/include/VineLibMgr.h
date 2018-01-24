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
#ifndef VINE_LIB_MGR_HEADER
#define VINE_LIB_MGR_HEADER
#include <string>
#include <set>
#include <map>
#include <thread>
#include "vine_pipe.h"
#include "accelThread.h"

typedef map<string, void *> Str2VpMap;
typedef set<vine_proc *> ProcSet;

class VineLibMgr {
    public:
        VineLibMgr();
        bool loadFolder(string lib_path,bool silent = false);
        bool loadLibrary(string lib_file);
		void unloadLibraries(vine_pipe_s *pipe);
		void startRunTimeLoader();
        ~VineLibMgr();

    private:
		static void rtld_handler(VineLibMgr * libmgr);
		std::set<std::string> lib_paths;
		std::map<int,std::string> lib_fd_paths;
		Str2VpMap libs;
        ProcSet procs;
		std::thread rtld;	// Run time loader
		int inotify_fd;
		bool run;
};

/**
 * Receives arguments and inputs/outputs.
 * Performs argument marshalling and task issue to accelerator.
 */
typedef vine_task_state_e(VineFunctor)(vine_task_msg_s *);

// typedef vine_task_state_e(VineFunctor)(vineTaskAndTimers &);

struct VineProcedureDefinition {
    const char *name;
    vine_accel_type_e type;
    vine_accel_type_e max_type; /** Always set to VINE_ACCEL_TYPES */
    VineFunctor *functor;
    size_t arg_size;
};

#define VINE_PROC_LIST_START() \
    extern "C" {                 \
        struct VineProcedureDefinition vine_proc_defs[] = {
#define VINE_PROCEDURE(NAME, TYPE, FUNCTOR, ARG_SIZE) \
            { NAME, TYPE, VINE_ACCEL_TYPES, FUNCTOR, ARG_SIZE } \
            ,
#define VINE_PROC_LIST_END() \
            { 0 }                      \
        }                          \
        ;                          \
    }
#endif

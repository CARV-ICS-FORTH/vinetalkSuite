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
#include <iostream>
#include <cstring>
#include <cerrno>
#include <sys/types.h>
#include <dirent.h>
#include <dlfcn.h>
#include "VineLibMgr.h"
#include <sys/inotify.h>
#include <fstream>
#include <unistd.h>

#define ESC_CHR(CHR) (char)27 << "[1;" << (int)CHR << "m"

VineLibMgr::VineLibMgr()
: run(true)
{
    inotify_fd = inotify_init();
}

bool VineLibMgr::loadFolder(string lib_path,bool silent) {
    string lib;
    DIR *dir;
    dirent *itr;

    if(lib_paths.count(lib_path))
        return true;

    dir = opendir(lib_path.c_str());

    if (!dir) {
        if(!silent)
            cerr << "Path \'" << lib_path << "\' could not be opened." << endl;
        return false;
    }

    lib_fd_paths[inotify_add_watch(inotify_fd,lib_path.c_str(),IN_CLOSE_WRITE)] = lib_path;

    lib_paths.insert(lib_path);

    do {
        itr = readdir(dir);
        if (itr)
        {
            if (itr->d_name[0] == '.')
                continue;

            lib = lib_path + "/" + itr->d_name;
            if (!loadFolder(lib,true))
                loadLibrary(lib);
        }
    } while (itr);

    closedir(dir);
    return true;
}
bool VineLibMgr::loadLibrary(string lib_file)
{
	if(lib_file.rfind(".so")==string::npos)
		return false;

	if(libs.count(lib_file))
		return true;

	void *handle = dlopen(lib_file.c_str(), RTLD_NOW); /* Fail now */
	VineProcedureDefinition *defs;

	if (!handle) {
        cerr << __func__ << "(" << lib_file << ")" << dlerror() << endl;
        return false;
    }

    /* We have a library, but is it a Vineyard lib? */
    defs = (VineProcedureDefinition *)dlsym(handle, "vine_proc_defs");

    if (!defs)
    {
        cerr << __func__ << "(" << lib_file << "):Not a Vineyard library"<< endl;
        dlclose(handle);
        return false;
    }
    libs[lib_file] = handle;

    while (defs->name)
    {
        if (defs->max_type > VINE_ACCEL_TYPES)
            cerr << "Warning: " << defs->name << "() in " << lib_file
                << " compiled with future version!" << endl;
        if (defs->type >= VINE_ACCEL_TYPES)
        {
            cerr << "Error: " << defs->name << "() in " << lib_file
                << " targeting for unknown accelerator,skiping!" << endl;
            defs++;
            continue;
        }
        void *pntr = (void *)(defs->functor);
        vine_proc *proc;

        proc = vine_proc_register(defs->type, defs->name, (void *)(&pntr),sizeof(void *));
        if(proc)
        {
            procs.insert(proc);
            cerr << "Registered " << vine_accel_type_to_str(defs->type) << "::" << defs->name
                << "():" << ESC_CHR(32) << "Successful!" << ESC_CHR(0) << endl;
        }
        else
        {
            cerr << "Registration of " << vine_accel_type_to_str(defs->type)
                << "::" << defs->name << "():" << ESC_CHR(31) << "Failed!" << ESC_CHR(0) << endl;
        }
        defs++;
    }

    return true;
}

void VineLibMgr::unloadLibraries(vine_pipe_s *pipe)
{

    for (ProcSet::iterator itr = procs.begin(); itr != procs.end(); itr++) {
        cerr << "Unregistering: " << vine_accel_type_to_str(((vine_proc_s *)*itr)->type)
            << "::" << ((vine_proc_s *)*itr)->obj.name << "()" << endl;
        vine_pipe_delete_proc(pipe, (vine_proc_s *)*itr);
    }
    procs.clear();
    for (Str2VpMap::iterator itr = libs.begin(); itr != libs.end(); itr++)
        dlclose(itr->second);
    libs.clear();
}

void VineLibMgr::startRunTimeLoader() {
    rtld = std::thread(rtld_handler,this);
}

union InotifyEvent
{
    struct inotify_event event;
    char padd[NAME_MAX + 1];
};

void VineLibMgr::rtld_handler(VineLibMgr * libmgr) {
    while(libmgr->run)
    {
        InotifyEvent event;

        read(libmgr->inotify_fd,&event,sizeof(event));

        libmgr->loadLibrary(libmgr->lib_fd_paths[event.event.wd] +"/" + event.event.name);
    }
}

VineLibMgr::~VineLibMgr() {
    run = false;
	// Create a fake library file 'run.false' to wakeup rtld thread
    {std::ofstream signal(lib_fd_paths.begin()->second+"/run.false");}
    rtld.join();
    if(procs.size() || libs.size())
        cerr << "VineLibMgr still has " << procs.size() << " procedures"
            "and " << libs.size() << " libraries loaded!" << endl;

}

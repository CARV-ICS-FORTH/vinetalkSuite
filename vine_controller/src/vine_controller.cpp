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
#include <fstream>
#include <csignal>
#include <signal.h>
#include <iomanip>
#include <stdio.h>
#include <cstring>
#include <stdlib.h>
#include <sstream>
#include "vine_pipe.h"
#include <vector>
#include <cstdlib>
#include <pthread.h>
#include <unistd.h>
#include "VineLibMgr.h"
#include "accelThread.h"
#include "Scheduler.h"
#include "definesEnable.h"
#include "Formater.h"
#include <chrono>
#include <ctime>
#include <sched.h>

using namespace ::std;

vine_pipe_s *vpipe; /* get a pointer to the vine pipe */
void *shm = 0;      /* get a pointer to the share memory segment */
volatile bool shouldExit = false;

/* Define the function to be called when ctrl-c (SIGINT) signal is sent to
 * process*/
void signal_callback_handler(int signum) {

#if (DEBUG_ENABLED)
    cout << "Caught signal " << signum << endl;
#endif
    shouldExit = true;
}

/*Deregister accelerators at termination*/
void deRegisterAccel(vine_pipe_s *vpipe_s,
        vector<AccelConfig*> &accelSystemList) {
    cout << "Deregister accelerators" << endl;
    vector<AccelConfig*>::iterator it;
    for (it = accelSystemList.begin(); it != accelSystemList.end(); ++it) {
		vine_accel_release((vine_accel**)&((*it)->vine_accel));
    }
}

/*Creates one thread per physical accelerator*/
bool spawnThreads(vine_pipe_s *vpipe_s,vector<AccelConfig*> &accelSystemList) {
    /*iterate to the list with the accels from config file*/
    vector<AccelConfig*>::iterator itr;
    string msg;
    accelThread *thread;

    for (itr = accelSystemList.begin(); itr != accelSystemList.end(); ++itr) {
        // Initialize accelerator
        (*itr)->vine_accel = vine_accel_init(vpipe_s , (char *)((*itr)->name.c_str()), (*itr)->type);
        if ((*itr)->vine_accel == 0)
        {
            msg = "Failed to perform initialization";
            goto FAIL;
        }

        // Create thread
        thread = threadFactory.constructType((*itr)->type_str,vpipe_s,*(*itr));
        if(!thread)
        {
            msg = "Could not create thread";
            goto FAIL;
        }
        (*itr)->accelthread = thread;
        // Spawn thread
        thread->spawn();
    }
    return true;
FAIL:
    cerr << "While spawning line " << (*itr)->line << ": \'" << *itr << "\'\n" << msg << " for " << (*itr)->type_str << endl;
    return false;
}

/*Deregister threads */
void deleteThreads(vector<AccelConfig*> &accelSystemList) {
    /*iterate to the list with the accels from config file*/
    vector<AccelConfig*>::iterator itr;
    for (itr = accelSystemList.begin(); itr != accelSystemList.end(); ++itr) {
        /*creates one thread per line from config file*/
        (*itr)->accelthread->terminate();
    }
    for (itr = accelSystemList.begin(); itr != accelSystemList.end(); ++itr) {
        (*itr)->accelthread->joinThreads();
        delete (*itr)->accelthread;
    }
}

vine_pipe_s *vpipe_s;

/*Main function*/
int main(int argc, char *argv[]) {
    /* get the directories that .so exist*/
    if (argc < 2)
    {
        cerr << "Usage:\n\t" << argv[0] << " config_file" << endl;
        return 0;
    }

    /*Create a config instance*/
    try
    {
		/*Config gets as arguments a file with the appropriate info*/
		Config config(argv[1]);

		/*Vector with the paths that libraries are located*/
		vector<string> paths  = config.getPaths();

        if(paths.size() == 0)
        {
            cerr << "No library paths declared in config file" << endl;
            return 0;
        }

        cerr << "Library paths: " << Formater<string>("\n\t","\n\t","\n",paths) << endl;

        /*get the share mem segment*/
		vpipe_s = vine_talk_init();

        /* Load folder with .so*/
		VineLibMgr *vineLibMgr = new VineLibMgr();
        for (vector<string>::iterator itr = paths.begin() ; itr != paths.end(); itr++)
        {
            if(!vineLibMgr->loadFolder(*itr))
                return -1;
        }

        vineLibMgr->startRunTimeLoader();

        cerr << "Supported threads: "
            << Formater<string>("\n\t","\n\t","\n",threadFactory.getTypes()) << endl;
        cerr << "Supported Schedulers: "
            << Formater<string>("\n\t","\n\t","\n",schedulerFactory.getTypes()) << endl;

        cerr << config << endl;

        /*create a vine task pointer*/
        vector<AccelConfig*> vectorOfAccels;
        vectorOfAccels = config.getAccelerators();

        signal(SIGINT, signal_callback_handler);

        /*create threads*/
        if(!spawnThreads(vpipe_s,vectorOfAccels))
        {
            cerr << "Failed to spawn threads, exiting" << endl;
            return 0;
        }

        while (!shouldExit)
        {
            usleep(1000);
            if (shouldExit)
            {
                cout << "Ctr+c is pressed EXIT!" << endl;
                break;
            }
        }

        deleteThreads(vectorOfAccels);
        deRegisterAccel(vpipe_s, vectorOfAccels);
        /*Delete libraries*/
		vineLibMgr->unloadLibraries(vpipe_s);
        delete vineLibMgr;

    }catch(exception * e)
    {
        cerr << "Error:\n\t" << e->what() << endl;
    }

    vine_talk_exit();
    return 0;
}

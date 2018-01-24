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
#ifndef TIMERS_H
#define TIMERS_H
#include <chrono>
#include <ctime>

using namespace::std;

/*Timers per job*/
struct Timers_s {
    int jobId;
    int taskId;
    double elapsed_H2D;  // time for copying data from Host2Device
    chrono::duration<double>
        elapsed_D2H;  // time for copying data from Device2Host
    double
        elapsed_Malloc;  // time for allocating memory to device (inputs+outputs)
    chrono::duration<double> elapsed_Free;  // time for free resources
    chrono::duration<double> elapsed_ExecT;  // time for excution time
    chrono::duration<double>
        elapsed_mapping_VAQ_2_PA;  // time for scheduling decsion
    double avgTaskSchedulingDuration;  // AVG scheduling decision per task
    chrono::time_point<chrono::system_clock>
        startTimeTask;  // point of time that a task has started
    chrono::time_point<chrono::system_clock>
        endTimeTask;  // point of time that a task has ended
};
#if 0
/*Printing results from timers */
static void printStats(vector<Timers> stats, map<string, Timers>::iterator timersIt) {
    cout << "-----------Printing results--------------"<< endl << endl;
    for (timersIt = stats.begin(); timersIt != stats.end(); ++timersIt) {
        cout << "JobID:                 "<< timersIt->first << endl 
            << "TaskID:                "<< timersIt->second.taskId << endl
            << "Data transfer H2D:     "<< timersIt->second.elapsed_H2D.count() << endl 
            << "Data transfer D2H:     "<< timersIt->second.elapsed_D2H.count() << endl
            << "Free resources:        "<< timersIt->second.elapsed_D2H.count() << endl
            << "Execution time:        "<< timersIt->second.elapsed_ExecT.count() << endl    
            << "Mapping VAQ2 P.A.:     "<< timersIt->second.elapsed_mapping_VAQ_2_PA.count() << endl
            << "AVG task scheduling:   "<< timersIt->second.avgTaskSchedulingDuration << endl;
        //<< "Start time:            "<< timersIt->second.startTimeTask.count() << endl
        //<< "End time:              "<< timersIt->second.endTimeTask.count() << endl;
    }
    cout << endl << "-----------End printing results--------------" << endl;
}

#endif

#endif

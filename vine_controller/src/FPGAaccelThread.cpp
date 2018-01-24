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
#include "FPGAaccelThread.h"
using namespace std;

FPGAaccelThread::FPGAaccelThread(vine_pipe_s * v_pipe, AccelConfig & conf): accelThread(v_pipe, conf) {}
FPGAaccelThread::~FPGAaccelThread() {}

/*initializes the FPGA accelerator*/
bool FPGAaccelThread::acceleratorInit()
{
    cout << "FPGA initalization done." << endl;
    return true;
}
/*Releases the CPU accelerator*/
void FPGAaccelThread::acceleratorRelease()
{
    cout << "FPGA released." << endl;
}

REGISTER_ACCEL_THREAD(FPGAaccelThread)

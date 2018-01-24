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
#include <stdio.h>
#include <iostream>
#include <fstream>
using namespace ::std;

int main() {
    ifstream fin;
    fin.open("/proc/cpuinfo", ios::in);

    ofstream fout;
    fout.open("new.txt", ios::out);
    char ch;
    char *model_name, *cpu_cores;
    char line[75];
    while (fin.get(ch)) 
    {
        fin.get(line, 75, '\n');
        model_name = strstr(line, "model name");
        cpu_cores = strstr(line, "cpu cores");

        if (model_name != NULL) 
        {
            fout << "Accelerator type is CPU \n" << model_name << endl;
        } else if (cpu_cores != NULL) {
            fout << cpu_cores << endl << "--------------------" << endl;
        }
    }
    // Number of CUDA devices
    int devCount;
    cudaGetDeviceCount(&devCount);
    if (devCount > 0) 
    {
        fout << "Accelerator type is NVIDIA GPU" << endl;
        fout << "Number of NVIDIA GPUs: " << devCount << endl;
        // Iterate through devices
        for (int i = 0; i < devCount; ++i) 
        {
            cudaDeviceProp devProp;
            cudaGetDeviceProperties(&devProp, i);
            fout << "model name       " << devProp.name << endl;
            fout << "Multi-Processors " << devProp.multiProcessorCount << endl;
            cudaSetDevice(i);
        }
    } else 
    {
        cout << "There is no CUDA device" << endl;
    }
    fin.close();
    return 0;
}

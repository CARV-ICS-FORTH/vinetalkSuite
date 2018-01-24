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
#include <cuda.h>
#include <cuda_runtime_api.h>

/*
 Explicitly destroys and cleans up all resources associated with the current device in the current process.
 Any subsequent API call to this device will reinitialize the device.
 Note that this function will reset the device immediately.
 It is the caller's responsibility to ensure that the device is not being accessed by any other host threads from the process when this function is called.
*/

void colorPrint(bool expr,std::string msg[2])
{
	if(expr)
		std::cerr << (char)27 << '[' << 32 << 'm';
	else
		std::cerr << (char)27 << '[' << 31 << 'm';
	std::cerr << msg[expr];
	std::cerr << (char)27 << '[' << 0 << 'm';
}
int main(int argc,char * argv[])
{
	int devCount;
	std::string sorf[2];
	sorf[0] = "Failed";;
	sorf[1] = "Successfull";
	cudaGetDeviceCount(&devCount);
	std::cerr << "Devices found:" << devCount << std::endl;
	if(argc == 1)
	{
		std::cerr << "Usage:\n\t" << argv[0] << "GPU_ID0 ... GPU_IDN" << std::endl;
		return -1;
	}
	for(int gpu_id = 1 ; gpu_id < argc ; gpu_id++)
	{
		int gpu = atoi(argv[gpu_id]);
		if(gpu < devCount)
		{
			cudaError_t err;
			cudaSetDevice (gpu);
			err = cudaDeviceReset ();
			std::cerr << "Reseting GPU #" << gpu << " ";
			colorPrint((err==cudaSuccess),sorf);
			std::cerr << std::endl;
		}
		else
		{
			sorf[0] = "Wrong GPU #"+std::to_string(gpu);
			colorPrint(0,sorf);
			std::cerr << std::endl;
		}
	}
	return 0;
}

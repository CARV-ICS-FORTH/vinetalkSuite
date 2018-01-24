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
#include "stdio.h"
#include <iostream>
#include "../include/prepareGPU.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>
//#include <nvml.h>

using namespace std;

/*Find the avaliable CUDA devices in the system*/
int numberOfCudaDevices() {
	// Number of CUDA devices
	int devCount;
	cudaGetDeviceCount(&devCount);
	if (devCount > 0) 
	{
		// Iterate through devices
		for (int i = 0; i < devCount; ++i) 
		{
			cudaDeviceProp devProp;
			cudaGetDeviceProperties(&devProp, i);
			//deviceSpecs(devCount);
		}
		return devCount;
	} else {
		cout << "There is no CUDA device" << endl;
		return 0;
	}
}
/*GPUs specifications*/
void deviceSpecs(int deviceCount){
	int dev;
	for (dev = 0; dev < deviceCount; ++dev)
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

		// Console log
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

		// This is supported in CUDA 5.0 (runtime API device properties)
		printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
		printf("  Memory Bus Width:                              %d-bit\n",   deviceProp.memoryBusWidth);

		if (deviceProp.l2CacheSize)
		{
			printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
		}
		printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
				deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
				deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
		printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
				deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
		printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
				deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);

		printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
		printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);	
		printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
		printf("  Warp size:                                     %d\n", deviceProp.warpSize);
		printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
		printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
		printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
				deviceProp.maxThreadsDim[0],
				deviceProp.maxThreadsDim[1],
				deviceProp.maxThreadsDim[2]);
		printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
				deviceProp.maxGridSize[0],
				deviceProp.maxGridSize[1],
				deviceProp.maxGridSize[2]);
		printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
		printf("  Number of Concurrent kernels :       %d\n", deviceProp.concurrentKernels);
		printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
		printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
		const char *sComputeMode[] =
		{
			"Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
			"Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
			"Prohibited (no host thread can use ::cudaSetDevice() with this device)",
			"Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
			"Unknown",
			NULL
		};
		printf("  Compute Mode:\n");
		printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
	}
}

/*Initiliazes CUDA devices*/
bool initCUDADevice(int id) {
	if (cudaSetDevice(id) != cudaSuccess) {
		cout<<"failed to set Device with ID: "<< id <<endl;
		return false;
	}
	return true;
}

/*Performs a cudaMalloc and Free inorder to prepare the device*/
bool prepareCUDADevice() {
	float *d_a;
	int size = sizeof(int);
	cudaError_t error;
	error = cudaMalloc((void**)&d_a,size);

	if (error != cudaSuccess){
		//cout <<" First CUDA Malloc Failed at "<< __LINE__ << " .  Error " <<cudaGetErrorString(error)<< " with code  " << error <<endl;
		return false;
	}

	if (cudaFree(d_a) != cudaSuccess) {
		cout<<"Free failed"<<endl;
		return false;
	}
	return true;

}

/*
   bool firstCUDAMalloc(){
   float *d_a;
   if (cudaMalloc((void**)&d_a,1)!= cudaSuccess){
   return false;
   }
   return true;
   }
 */

/*Reset CUDA devices*/
bool resetCUDADevice() {
	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	if (cudaDeviceReset() != cudaSuccess) {
		return false;
	}
	return true;
}

unsigned int  monitorGPU()
{
	/*
	nvmlInit();
	int devicePCI ;
	cudaGetDevice(&devicePCI);
	nvmlDevice_t device;
	unsigned int powerConsumption;

	nvmlDeviceGetHandleByIndex(devicePCI, &device);
	nvmlDeviceGetPowerUsage(device, &powerConsumption);
	return powerConsumption;
	*/
	return 0;
}

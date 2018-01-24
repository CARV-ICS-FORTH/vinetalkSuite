#ifndef GPUACCEL_CUH
#define GPUACCEL_CUH

bool initCUDADevice(int id);
int gpuGetMaxGflopsDeviceId();
int numberOfCudaDevices();
bool resetCUDADevice();
bool prepareCUDADevice();
void deviceSpecs(int number);
unsigned int  monitorGPU();
#endif

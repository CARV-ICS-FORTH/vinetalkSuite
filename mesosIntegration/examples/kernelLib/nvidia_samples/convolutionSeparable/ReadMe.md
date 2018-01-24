Sample: CUDA Separable Convolution
Minimum spec: SM 2.0

This sample implements a separable convolution filter of a 2D signal with a gaussian kernel.

Key concepts:
Image Processing
Data Parallel Algorithms
This example runs in two accelerator types, CPU and GPU, via `vine_talk`.
Execute histogram with images with bigger size. if one add simzemult =2 the image is doubled.
LD_PRELOAD=./lib/cuda2vineTalk.s ./histogram -sizemult=2

## Folder layout
* **include** - Header files 
	- **histogram_common.h**: Header file that contains the function definition of 
	histogram that is executed in the CPU accelerator.

	- **histogramArgs.h**: Header file that contains the arguments needed by 
	  the histogram function. These arguments are transferred from the 
	  application to the controller through vine\_talk. 

* **src** -  Source code
	- **histogram_gold.cpp**: Contains the kernel that is going to be 
	executed in CPU, and the approproate function calls to prepare for this 
	execution. It is used to create the `.so` file.
  	- **histogram256.cu**: contains the kernel for histogram256 that is going to be executed in 
  	GPU, and the approproate function calls to prepare for this execution 
  	(cudaMalloc, cudaMemcpy, cudaFree). It is used to create the `.so` file.
  	- **histogram64.cu**: contains the kernel for histogram64 that is going to be executed in 
  	GPU, and the approproate function calls to prepare for this execution 
  	(cudaMalloc, cudaMemcpy, cudaFree). It is used to create the `.so` file.
	- **main.cpp**: contains the call that creates a histogram task 
	that is going to be executed in the controller.
    - **cuda2vineTalk.c**:contains the transformation from cuda fucntions to vinetalk calls
    
## Requirements
This benchmark requires [vine_talk](https://carvgit.ics.forth.gr/vineyard/vine_talk 
"Vine Talk") and [vine_controller](https://carvgit.ics.forth.gr/vineyard/vine_controller
"Vine Controller") in order to run. Download them from the provided links. Then 
proceed with downloading vine\_applications, placing in the same folder as the 
previous two packages. 

CUDA applications also need the `vinecudalib.so` library create it by type make in the current dir. 

We recommend that the folders are structured as follows: 

	vineyard_dir:
		- vine_talk
		- vine_controller
		- vine-applications
		
Otherwise, you must change the paths in every Makefile in order for the code to 
compile correctly.  

## Building
* First, build vine\_talk according to its instructions.
* Then, build vine\_controller, again according to its instructions.
* Finally, build histogram by typing `make`. This is going to create two 
folders: 
	- **lib**, that contains one .so file, with the kernels for both CPU and GPU . 
	
## Running  
* Run `vine_controller` according to the information provided in the README file
there. The executable requires as an argument a library folder. Please, provide 
the `lib` folder created after building histogram. 
(./bin/vine_controller ../vine-applications/image_proc_samples_NVIDIA/3_Imaging/histogram/lib)
	- histogram creates one .so filew, that  contains the kernel for the GPU and the CPU.  
	The controller is going to load the libraries, so that it can execute the code required by the task. 
* To run the application on CPU or GPU, type respectively:
```
LD_PRELOAD=./lib/cuda2vineTalk.so ./histogram histogramVersion=0 accelType=1
histogramVersion = 0 for histogram64 
                          or 1 for histogram256
accetype = 1 for GPU
                  0 for CPU
``` 

or with make run : for histogram 256 and GPU (default values)
	make run_gpu        : for histogram 256 and GPU
        make run_cpu        : for histogram 256 and CPU
	make run_gpu_64     : for histogram 64 and GPU
	make run_gpu_256    : for histogram 256 and GPU

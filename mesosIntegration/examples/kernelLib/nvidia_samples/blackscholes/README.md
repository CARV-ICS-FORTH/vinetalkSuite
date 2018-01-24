# Black-Scholes

BlackScholes is an implementation of the Black-Scholes formula, that evaluates 
fair call and puts prices for a given set of European options. This benchmark 
was created by NVidia and was taken from the samples in CUDA toolkit. We ported
it in order to work over vine talk. 

This example runs in two accelerator types, CPU and GPU, via `vine_talk`.

## Folder layout
* **include** - Header files 
	- **c_blackScholes.h**: Header file that contains the function definition of 
	blackScholes that is executed in the CPU accelerator.
	- **BlackScholes_kernel.cuh**: Header file that contains the function 
	definition	of blackScholes that is executed in the GPU accelerator. 
	- **blackScholesArgs.h**: Header file that contains the arguments needed by 
	  the blackScholes function. These arguments are transferred from the 
	  application to the controller through vine\_talk. 

* **src** -  Source code
	- **BlackScholes_gold.cpp**: Contains the kernel that is going to be 
	executed in CPU, and the approproate function calls to prepare for this 
	execution. It is used to create the `.so` file.
  	- **BlackScholes.cu**: contains the kernel that is going to be executed in 
  	GPU, and the approproate function calls to prepare for this execution 
  	(cudaMalloc, cudaMemcpy, cudaFree). It is used to create the `.so` file.
	- **BlackScholes.cpp**: contains the call that creates a blackScholes task 
	that is going to be executed in the controller.

## Requirements
This benchmark requires [vine_talk](https://carvgit.ics.forth.gr/vineyard/vine_talk 
"Vine Talk") and [vine_controller](https://carvgit.ics.forth.gr/vineyard/vine_controller
"Vine Controller") in order to run. Download them from the provided links. Then 
proceed with downloading vine\_applications, placing in the same folder as the 
previous two packages. 

CUDA applications also need the `vinecudalib.so` , it is produced by make.

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
* Finally, build BlackScholes by typing `make`. This is going to create two 
folders: 
	- **lib**, that contains the two .so files, a kernel for CPU and GPU each. 
	- **bin**, that contains the executable. We may refer to this executable 
	either as application or as procuder. 

## Running  
* Run `vine_controller` according to the information provided in the README file
there. The executable requires as an argument a library folder. Please, provide 
the `lib` folder created after building BlackScholes. 
	- BlackScholes creates two .so files, one containing the kernel for the GPU 
	the other containing the kernel for the CPU. The controller is going to load 
	the libraries, so that it can execute the code required by the task. 
* To run the application on CPU or GPU, type respectively:
```
make run_cpu
``` 
or 
```
make run_gpu
```



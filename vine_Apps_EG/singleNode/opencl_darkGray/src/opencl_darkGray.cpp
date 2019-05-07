#include "../include/opencl_darkGray.h"
#include "../include/darkGrayArgs.h"

#include "VineLibUtilsOpenCL.h"
#include "err_code.h"

#include <chrono>
#include <cmath>


#define TIMERS_ENABLED 1


/* Function that calls the kernel*/
bool opencl_darkGray(
	const int width,
	const int height,
	cl::Buffer *inputImageDev,
	cl::Buffer *darkGrayImageDev) {
	size_t global_work_size[2];
	// size_t local_work_size[2];

	// int wBlock = static_cast<unsigned int>(ceil(width / static_cast<float>(32)));
	// int hBlock = static_cast<unsigned int>(ceil(height / static_cast<float>(16)));

	global_work_size[0] = width;
	global_work_size[1] = height;
	// local_work_size[0] = wBlock;
	// local_work_size[1] = hBlock;
	try {
		cl::NDRange global(global_work_size[0], global_work_size[1]);
		// cl::NDRange local(local_work_size[0], local_work_size[1]);

		cl::Kernel kernel = OpenCLGetKernel("rgb_gray");
		cl::CommandQueue defaultCommandQueue = getDefaultCommandQueue();

		kernel.setArg(0, width);
		kernel.setArg(1, height);
		kernel.setArg(2, *inputImageDev);
		kernel.setArg(3, *darkGrayImageDev);

		// local size is not used, because I can't find out something that consistently works.

		// defaultCommandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
		defaultCommandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
		defaultCommandQueue.finish();

	} catch (cl::Error err) {
		cout << "OpenCL Error: " << err.what() << " returned " << err_code(err.err()) << endl;
		return false;
	}

	return true;
}

/* Contains the code that is executed in Host*/
vine_task_state_e hostCode(vine_task_msg_s *vine_task) {
#ifdef TIMERS_ENABLED
	std::chrono::time_point<std::chrono::system_clock> start, end;
#endif
	std::vector<void *> ioVector;
	darkGrayArgs *argsCuda;

	cout << "opencl_darkgray execution in OpenCL." << endl;

	/* Get the actual arguments*/
	argsCuda = (darkGrayArgs *)vine_data_deref(vine_task->args.vine_data);

	/* Allocate memory in the device and transfer data */
	if (!Host2OpenCL(vine_task, ioVector)) {
		cerr << "Host2OpenCL" << endl;
		cerr << __FILE__ << " Failed at " << __LINE__ << endl;
		return (task_failed);
	}

#ifdef TIMERS_ENABLED
	start = std::chrono::system_clock::now();
#endif
	/* Call the function that calls the kernel */
	bool success = opencl_darkGray(
		argsCuda->width,
		argsCuda->height,
		(cl::Buffer *)ioVector[0],
		(cl::Buffer *)ioVector[1]);
	if (!success) {
		vine_task->state = task_failed;
		return vine_task_stat(vine_task, 0);
	}
#ifdef TIMERS_ENABLED

	end = std::chrono::system_clock::now();

	std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	std::chrono::duration<double, std::nano> elapsed_seconds = end - start;

	cout << "DarkGray kernel execution time: " << elapsed_seconds.count()
		 << " nanosecs." << endl;
#endif
	/* Check for cuda errors*/
	// cudaError_t err = cudaGetLastError();
	// if (err != cudaSuccess) {
	// 	cerr << __FILE__ << " Failed at " << __LINE__ << endl;
	// 	printf("Error: %s\n", cudaGetErrorString(err));
	// 	return (task_failed);
	// }

	/* Copy back the result  from OpenCL*/
	if (!(OpenCL2Host(vine_task, ioVector))) {
		cerr << "OpenCL2Host" << endl;
		cerr << __FILE__ << " Failed at " << __LINE__ << endl;
		return (task_failed);
	}

	/* Free device memory*/
	if (!OpenCLMemFree(ioVector)) {
		cerr << "OpenCLMemFree" << endl;
		cerr << __FILE__ << " Failed at " << __LINE__ << endl;
		return (task_failed);
	}
	/* Execution was successful*/
	return vine_task_stat(vine_task, 0);
}
/* register the function to the array for this .so*/
VINE_PROC_LIST_START()
VINE_PROCEDURE("darkGray", OPEN_CL, hostCode, sizeof(darkGrayArgs))
VINE_PROC_LIST_END()

#include "NanoCoreaccelThread.h"

using namespace::std;
#include <iostream>

/// @brief  Creates a new accelerator thread for NanoCore.
///
NanoCoreaccelThread::NanoCoreaccelThread(vine_pipe_s * v_pipe, AccelConfig & conf) : accelThread(v_pipe, conf) {}

/// @brief  Destroys the accelerator thread.
///
NanoCoreaccelThread::~NanoCoreaccelThread() {}

/// @brief  Initializes NanoCore accelerator.
///
bool NanoCoreaccelThread::acceleratorInit() {
    cout << "NanoCore initalization done." << endl;
    return true;
}

/// @brief  Releases NanoCore accelerator.
///
void NanoCoreaccelThread::acceleratorRelease() {
    cout << "NanoCore released." << endl;
}

/**
 * Transfer Function Implementations
 */

/**
 * TODO: UGLY HACK TO AVOID API change PLZ FIXME
 */
extern vine_pipe_s *vpipe_s;

bool Host2NanoCore(vine_task_msg_s *vine_task, vector<void *> &ioHD) {
	int in;

	/*Start timer for Host2CPU*/
	utils_breakdown_advance(&(vine_task->breakdown),"MemCpy_H2NanoCore");

	#ifdef DATA_TRANSFER
	cout<<"Number of inputs: "<<vine_task->in_count<<endl;
	cout<<"Number of outputs: "<<vine_task->out_count<<endl;
	#endif

	for (in = 0; in < vine_task->in_count; ++in)
	{
		vine_data_s *vinedata = static_cast<vine_data_s *>(vine_task->io[in].vine_data);
		ioHD.push_back(vine_data_deref(vinedata));
	}

	#ifdef DATA_TRANSFER
	cout<<"Size of input " <<in<<" is: "<< vine_data_size(vine_task->io[in].vine_data)<<endl;
	#endif

	for (int out = 0; out < vine_task->out_count + in; ++out)
	{
		vine_data_s *vinedata = static_cast<vine_data_s *>(vine_task->io[out].vine_data);
		ioHD.push_back(vine_data_deref(vinedata));
		#ifdef DATA_TRANSFER
		cout<<"Size of output " <<out<<" is: "<< vine_data_size(vine_task->io[out].vine_data)<<endl;
		#endif
	}

	/* Stop timer for Host to CPU - Start timer for kernel execution in CPU*/
	utils_breakdown_advance(&(vine_task->breakdown), "Kernel Execution_NanoCore");

	return true;
}

/// @brief  Transfers data from NanoCore to host memory.
///
/// @param  vine_task   Vineyard task information.
/// @param  ioHD        Data to transfer.
///
/// @retval true    The transmission was successful.
/// @retval false   The transmission failed.
///
bool NanoCore2Host(vine_task_msg_s *vine_task, vector<void *> &ioDH)
{
	/*Stop timer for kernel execution in CPU - Start timer for CPU to Host*/
	utils_breakdown_advance(&(vine_task->breakdown), "MemCpy_NanoCore2H");

	int out_limit = vine_task->out_count + vine_task->in_count;

	for (int out = vine_task->in_count; out < out_limit; out++)
	{
		if (out == vine_task->out_count + vine_task->in_count-1)
		{
			/*Mark the state as completed*/
			vine_task->state = task_completed;

			/*Stop timer for CPU to Host - Start timer for memFree*/
			utils_breakdown_advance(&(vine_task->breakdown), "MemFreeNano");
		}
		vine_data_mark_ready(vpipe_s, vine_task->io[out].vine_data);
	}
	return true;
}

/// @brief  Releases memory allocated on NanoCore.
///
/// @param  io  Address of the memory space on NanoCore to free.
///
/// @retval true    Freeing was successful.
/// @retval false   Freeing failed.
///
bool NanoCoreMemFree(vector<void *> &io) {
	return true;
}

REGISTER_ACCEL_THREAD(NanoCoreaccelThread)

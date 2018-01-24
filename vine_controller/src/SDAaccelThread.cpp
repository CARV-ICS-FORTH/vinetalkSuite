#include <iostream>
#include <sstream>
#include "SDAaccelThread.h"
#include "xcl.h"
#include <iostream>
#include <set>
#include <chrono>
#include <ctime>
#include "definesEnable.h"
#include "timers.h"

using namespace std;

SDAaccelThread::SDAaccelThread(vine_pipe_s * v_pipe, AccelConfig & conf)
: accelThread(v_pipe, conf)
{
    std::istringstream iss(conf.init_params);
    std::string kernel_name;
    iss >> prof_thread_cpus;
    if(!iss)
        throw new runtime_error("SDA accelerator incorrect arguments:Missing Profiling thread cpumask");
    iss >> vendor;	// Xilinx
    if(!iss)
        throw new runtime_error("SDA accelerator incorrect arguments:Missing vendor");
    iss >> dev_addr_str;	// xilinx:adm-pcie-ku3:1ddr:3.0
    if(!iss)
        throw new runtime_error("SDA accelerator incorrect arguments:Missing device address");
    iss >> xclbin;
    if(!iss)
        throw new runtime_error("SDA accelerator incorrect arguments:Missing xclbin file");
    while(iss)
    {
        iss >> kernel_name;
        if(iss)
        {
            kernels[kernel_name] = 0;	/* Will create it at init */
            std::cerr << "SDA kernel: " << kernel_name << std::endl;
        }
    }

    if(!kernels.size())
        throw new runtime_error("SDA accelerator incorrect arguments:Missing kernel string(s)");
}
SDAaccelThread::~SDAaccelThread() {}

/*initializes the SDA accelerator*/
bool SDAaccelThread::acceleratorInit()
{
    /* Profiling thread will be spawned here, set its affinity */
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), prof_thread_cpus.getSet());
    world = xcl_world_single(CL_DEVICE_TYPE_ACCELERATOR, vendor.c_str(), dev_addr_str.c_str());
    for( auto &krnl : kernels)
    {
        krnl.second  = xcl_import_binary(world, xclbin.c_str(), krnl.first.c_str());
        cout<< "Registered " << krnl.first << ": " << (void*)kernels[krnl.first] << std::endl;
    }
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), getAccelConfig().affinity.getSet());
    cout << "SDA initalization done." << endl;
    return true;
}
/*Releases the CPU accelerator*/
void SDAaccelThread::acceleratorRelease()
{
    cl_int err;
    for( auto krnl : kernels)
    {
        cout << "Releasing " << krnl.first << ":";

        err = clReleaseKernel(krnl.second);

        if( err == CL_SUCCESS )
            cout << "Success\n";
        else
            cout << "Fail(" << (int)err << ")\n";
    }

    xcl_release_world(world);
    cout << "SDA released." << endl;

}

typedef void (SDAFunctor)(vine_task_msg_s *,xcl_world, cl_kernel);

void SDAaccelThread::executeHostCode(void* functor, vine_task_msg_s* task)
{
    std::string kname = ((vine_object_s*)task->proc)->name;
    (*(SDAFunctor**)(functor))(task,world,kernels["krnl_"+kname]);
}

/**
 * TODO: UGLY HACK TO AVOID API change PLZ FIXME
 */
extern vine_pipe_s *vpipe_s;

bool Host2SDA(vine_task_msg_s *vine_task, vector<void *> &ioHD,  xcl_world world, cl_kernel krnl)
{
	void *tmpIn;
	bool completed = true;

	/*Map vinedata with sda data*/
	map<vine_data *, void *> vineData2SDA;
	#ifdef BREAKDOWNS_VINETALK
	utils_breakdown_advance(&(vine_task->breakdown), "cudaMalloc_Inputs");
	#endif

	#ifdef DATA_TRANSFER
	cout<<"Number of inputs: "<<vine_task->in_count<<endl;
	cout<<"Number of outputs: "<<vine_task->out_count<<endl;
	#endif
	int mallocIn;
	for (mallocIn = 0; mallocIn < vine_task->in_count; mallocIn++)
	{
		/* Iterate till the number of inputs*/
		if (((((vine_data_s *)vine_task->io[mallocIn].vine_data)->place) & (Both)) == HostOnly)
		{
			ioHD.push_back(vine_data_deref(vine_task->io[mallocIn].vine_data));
			continue;
		}

		tmpIn = xcl_malloc(world, CL_MEM_READ_ONLY,  vine_data_size(vine_task->io[mallocIn].vine_data));
		clSetKernelArg(krnl, mallocIn, sizeof(cl_mem), &tmpIn);
		/*map between vinedata and cuda alloced data*/
		vineData2SDA[vine_task->io[mallocIn].vine_data] = tmpIn;
	}

	#ifdef BREAKDOWNS_VINETALK
	/*End Malloc input -  Start cudaMemcpy Inputs Host to Device*/
	utils_breakdown_advance(&(vine_task->breakdown), "cudaMemCpy_H2G");
	#endif

	int memCpyIn;
	for (memCpyIn = 0; memCpyIn < vine_task->in_count; memCpyIn++)
	{
		tmpIn=vineData2SDA[vine_task->io[memCpyIn].vine_data];

		/* Copy inputs to the device */
		xcl_memcpy_to_device(world,(cl_mem)tmpIn,vine_data_deref(vine_task->io[memCpyIn].vine_data),vine_data_size(vine_task->io[memCpyIn].vine_data));

		#ifdef DATA_TRANSFER
		cout<<"Size of input " <<memCpyIn<<" is: "<< vine_data_size(vine_task->io[memCpyIn].vine_data)<<endl;
		#endif

		ioHD.push_back(tmpIn);
	}
	int out;
	void *tmpOut;

	#ifdef BREAKDOWNS_VINETALK
	/*End cudaMemCpy Host to Device - Start cudaMalloc for outputs*/
	utils_breakdown_advance(&(vine_task->breakdown), "cudaMalloc_Outputs");
	#endif

	/*Alocate memory for the outputs */
	for (out = mallocIn; out < vine_task->out_count + mallocIn; out++)
	{
		if (((vine_data_s *)(vine_task->io[out].vine_data))->flags & VINE_INPUT)
		{
			tmpOut = vineData2SDA[vine_task->io[out].vine_data];
		} else
		{
			tmpOut = xcl_malloc(world, CL_MEM_WRITE_ONLY,  vine_data_size(vine_task->io[out].vine_data));
		}
		clSetKernelArg(krnl, mallocIn, sizeof(cl_mem), &tmpOut);

		/*End cudaMalloc for outputs - Start Kernel Execution time*/
		ioHD.push_back(tmpOut);
	}

	#ifdef BREAKDOWNS_VINETALK
	/*End cudaMalloc for outputs - Start Kernel Execution time*/
	utils_breakdown_advance(&(vine_task->breakdown), "Kernel_Execution_SDA");
	#endif

	return completed;
}

/* Cuda Memcpy from Device to host*/
bool SDA2Host(vine_task_msg_s *vine_task, vector<void *> &ioDH,xcl_world world, cl_kernel krnl)
{
	int out;
	bool completed = true;

	#ifdef BREAKDOWNS_VINETALK
	/*Stop Kernel Execution time - Start cudaMemCpy for outputs Device to Host*/
	utils_breakdown_advance(&(vine_task->breakdown), "cudaMemCpy_G2H");
	#endif

	for (out = vine_task->in_count; out < vine_task->out_count + vine_task->in_count; out++)
	{

		#ifdef DATA_TRANSFER
		cout<<"Size of output " <<out<<" is: "<< vine_data_size(vine_task->io[out].vine_data)<<endl;
		#endif

		xcl_memcpy_from_device(world, vine_data_deref(vine_task->io[out].vine_data), (cl_mem)ioDH[out], vine_data_size(vine_task->io[out].vine_data));
		completed = true;
		if (out==vine_task->out_count + vine_task->in_count-1)
		{
			vine_task->state = task_completed;
			#ifdef BREAKDOWNS_VINETALK
			/*End cudaMemCpy for outputs Device to Host - Start cudaMemFree*/
			utils_breakdown_advance(&(vine_task->breakdown), "cudaMemFree");
			#endif
		}
		vine_data_mark_ready(vpipe_s,vine_task->io[out].vine_data);
	}
	return completed;
}

/* Free Device memory */
bool SDAMemFree(vector<void *> &io)
{
	bool completed = true;
	set<void *> unique_set(io.begin(), io.end());
	for (set<void *>::iterator itr = unique_set.begin(); itr != unique_set.end(); itr++)
	{
		clReleaseMemObject((cl_mem)*itr);
	}
	return completed;
}

REGISTER_ACCEL_THREAD(SDAaccelThread)

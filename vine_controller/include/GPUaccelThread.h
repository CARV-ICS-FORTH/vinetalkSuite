#ifndef GPU_ACCELTHREAD
#define GPU_ACCELTHREAD
#include <pthread.h>
#include <vector>
#include "timers.h"
class GPUaccelThread;

#include "accelThread.h"

class GPUaccelThread : public accelThread {
    public:
        GPUaccelThread(vine_pipe_s * v_pipe, AccelConfig &conf);
        ~GPUaccelThread();
        virtual bool acceleratorInit(); /* Function that initializes a GPU accelerator */
        virtual void acceleratorRelease(); /* Function that resets a GPU accelerator */
		virtual void printOccupancy();
    private:
        int pciId;
};
#endif

#ifndef CPU_ACCELTHREAD
#define CPU_ACCELTHREAD
#include <pthread.h>
#include <vector>
#include "timers.h"
class CPUaccelThread;

#include "accelThread.h"

class CPUaccelThread : public accelThread {
    public:
		CPUaccelThread(vine_pipe_s * v_pipe, AccelConfig &conf);
        ~CPUaccelThread();
        virtual bool acceleratorInit(); /* Function that initializes a CPU accelerator */
        virtual void acceleratorRelease(); /* Function that resets a CPU accelerator */
};
#endif

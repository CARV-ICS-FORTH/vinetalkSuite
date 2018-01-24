#ifndef FPGA_ACCELTHREAD
#define FPGA_ACCELTHREAD
#include <pthread.h>
#include <vector>
#include "timers.h"
class FPGAaccelThread;

#include "accelThread.h"

class FPGAaccelThread : public accelThread {
    public:
		FPGAaccelThread(vine_pipe_s * v_pipe, AccelConfig & conf);
        ~FPGAaccelThread();
        virtual bool acceleratorInit(); /* Function that initializes a FPGA accelerator */
        virtual void acceleratorRelease(); /* Function that resets a FPGA accelerator */
};
#endif

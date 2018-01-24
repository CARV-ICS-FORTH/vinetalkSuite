#ifndef NANO_CORE_ACCELTHREAD
#define NANO_CORE_ACCELTHREAD

#include <vector>
#include <pthread.h>

#include "timers.h"
#include "accelThread.h"

class NanoCoreaccelThread : public accelThread {
    public:
        /// @brief  Creates a new accelerator thread for NanoCore.
        ///
		NanoCoreaccelThread(vine_pipe_s * v_pipe, AccelConfig &conf);

        /// @brief  Destroys the accelerator thread.
        ///
        ~NanoCoreaccelThread();

        /// @brief  Initializes NanoCore accelerator.
        ///
        virtual bool acceleratorInit();

        /// @brief  Releases NanoCore accelerator.
        ///
        virtual void acceleratorRelease();
};

#endif // !defined(NANO_CORE_ACCELTHREAD)

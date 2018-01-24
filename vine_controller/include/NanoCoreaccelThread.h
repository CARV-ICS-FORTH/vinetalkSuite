/*
 * Copyright 2018 Foundation for Research and Technology - Hellas
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0 [1] [1]
 *
 * Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 *  implied.
 * See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * Links:
 *  ------
 * [1] http://www.apache.org/licenses/LICENSE-2.0 [1] 
*/
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

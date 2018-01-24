#ifndef VINE_ROUND_ROBIN_SCHEDULER
#define VINE_ROUND_ROBIN_SCHEDULER
#include "Scheduler.h"
#include <map>
#include <vector>

class VineRoundRobinScheduler : public Scheduler {
    public:
		VineRoundRobinScheduler(std::string args);
        virtual ~VineRoundRobinScheduler();
        /*Pick a Virtual Accelerator Queue(VAQ) from all avaliable VAQs that exist in the system*/
        virtual utils_queue_s *selectVirtualAcceleratorQueue( accelThread *threadPerAccel );

        /*Select a task from all the VAQs that exist in the system  */
        virtual vine_task_msg_s *selectTask( accelThread *threadPerAccel );

    private:
        //Array with all VAQ types
        int acceleratorIndex[VINE_ACCEL_TYPES];
        map<vine_accel_s *, int> virtualQueueIndex;
};

#endif

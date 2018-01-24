#ifndef VINE_SCHEDULER
#define VINE_SCHEDULER

class Scheduler;

#include "Config.h"
#include "Factory.h"
#include "accelThread.h"
class Scheduler {
    public:
        Scheduler(std::string args);
		void setGroup(GroupConfig * group);
        virtual ~Scheduler() ;
        virtual utils_queue_s  *selectVirtualAcceleratorQueue(accelThread *th);
        virtual vine_task_msg_s *selectTask(accelThread *th) = 0;
		virtual void postTaskExecution(vine_task_msg_s *task);
        virtual void loadConfigurationFile() ;
    protected:
		GroupConfig * group;
};

extern Factory<Scheduler,std::string> schedulerFactory;

/**
 * Helper function to register Accelerator Threads
 * Must be put in a cpp file.
 */
#define REGISTER_SCHEDULER(CLASS)\
    static Registrator<Scheduler,CLASS,std::string> reg(schedulerFactory);
#endif

#include <iostream>
#include "Scheduler.h"
Scheduler::Scheduler(std::string args) {}

utils_queue_s  *Scheduler::selectVirtualAcceleratorQueue(accelThread *th)
{
	std::cerr << "Using " << __func__ << "although not implemented in scheduler.";
	throw std::runtime_error("Unimplemented function");
	return 0;
}

void Scheduler :: setGroup(GroupConfig * group)
{
	this->group = group;
}

Scheduler::~Scheduler() {}

void Scheduler::postTaskExecution(vine_task_msg_s *task)
{
}

void Scheduler::loadConfigurationFile()
{
    cout<<"Load configiration file dummy!!"<<endl;
}


Factory<Scheduler,std::string> schedulerFactory;

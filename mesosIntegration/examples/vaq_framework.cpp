// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

#include <chrono>
#include <boost/lexical_cast.hpp>

#include <mesos/resources.hpp>
#include <mesos/scheduler.hpp>
#include <mesos/type_utils.hpp>

#include <stout/check.hpp>
#include <stout/exit.hpp>
#include <stout/flags.hpp>
#include <stout/numify.hpp>
#include <stout/option.hpp>
#include <stout/os.hpp>
#include <stout/path.hpp>
#include <stout/stringify.hpp>

#include "logging/flags.hpp"
#include "logging/logging.hpp"

using namespace mesos;

using boost::lexical_cast;

using std::cerr;
using std::cout;
using std::endl;
using std::flush;
using std::string;
using std::vector;

using mesos::Resources;

const int32_t CPUS_PER_TASK = 1;
const int32_t MEM_PER_TASK = 128;
const int32_t VAC_RESOURCES_PER_TASK=1;

std::chrono::milliseconds framework_start_time = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch());
// A simple struct that is used for keeping information
// about duration of a task
struct TaskTimeInfo
{
  int taskId;
  std::chrono::milliseconds start_time;
  std::chrono::milliseconds end_time;
  std::chrono::milliseconds duration;
  std::chrono::milliseconds inQueue;
  long int vineInExecutor;
  float gpuTime;
  TaskTimeInfo(){};
};

vector<struct TaskTimeInfo> tasksDurationVector;
class TestScheduler : public Scheduler
{
public:
  TestScheduler(
      bool _implicitAcknowledgements,
      const ExecutorInfo& _executor,
      const string& _role)
    : implicitAcknowledgements(_implicitAcknowledgements),
      executor(_executor),
      role(_role),
      tasksLaunched(0),
      tasksFinished(0),
      totalTasks(1) {}

  virtual ~TestScheduler() {}

  virtual void registered(SchedulerDriver*,
                          const FrameworkID&,
                          const MasterInfo&)
  {
    cout << "Registered!" << endl;
  }

  virtual void reregistered(SchedulerDriver*, const MasterInfo& masterInfo) {}

  virtual void disconnected(SchedulerDriver* driver) {}

  virtual void resourceOffers(SchedulerDriver* driver,
                              const vector<Offer>& offers)
  {
    foreach (const Offer& offer, offers) {
      cout << "Received offer " << offer.id() << " with " << offer.resources()
           << endl;

           Resources taskResources = Resources::parse(
             "vac_resource:" + stringify(VAC_RESOURCES_PER_TASK) +
             //";gpus:"  + stringify(GPUS_PER_TASK) +
             ";cpus:" + stringify(CPUS_PER_TASK) +
             ";mem:"  + stringify(MEM_PER_TASK)
             //";vac_resource:"+stringify(VAC_RESOURCES_PER_TASK)
           ).get();
      taskResources.allocate(role);


      Resources remaining = offer.resources();

      // Launch tasks.
      vector<TaskInfo> tasks;
      while (tasksLaunched < totalTasks &&
             remaining.toUnreserved().contains(taskResources)) {
        int taskId = tasksLaunched++;

        cout <<"--------Launching task: " << taskId << " -------- \n----using offer: "
             << offer.id()<<" ----" << endl;


        TaskTimeInfo tmp = TaskTimeInfo();
        tmp.taskId= taskId;
        tmp.start_time=std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch());
        tmp.inQueue = tmp.start_time - framework_start_time;
        tasksDurationVector.push_back(tmp);

        TaskInfo task;
        task.set_name("Task " + lexical_cast<string>(taskId));
        task.mutable_task_id()->set_value(lexical_cast<string>(taskId));
        task.mutable_slave_id()->MergeFrom(offer.slave_id());
        task.mutable_executor()->MergeFrom(executor);




        Option<Resources> resources = [&]() {
          if (role == "*") {
            return remaining.find(taskResources);
          }

          Resource::ReservationInfo reservation;
          reservation.set_type(Resource::ReservationInfo::STATIC);
          reservation.set_role(role);

          return remaining.find(taskResources.pushReservation(reservation));
        }();

        CHECK_SOME(resources);
        task.mutable_resources()->MergeFrom(resources.get());
        remaining -= resources.get();
        //cout<<"Remaining recourses are: "<<remaining<<endl;

        tasks.push_back(task);
      }
        driver->launchTasks(offer.id(), tasks);

    }
  }

  virtual void offerRescinded(SchedulerDriver* driver,
                              const OfferID& offerId) {}

  virtual void statusUpdate(SchedulerDriver* driver, const TaskStatus& status)
  {

    int taskId = lexical_cast<int>(status.task_id().value());

    if (status.state() == TASK_FINISHED) {
      std::chrono::milliseconds end_time = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch());
      for(unsigned int i=0; i<tasksDurationVector.size(); i++){
        if(tasksDurationVector[i].taskId==taskId){
          tasksDurationVector[i].end_time=end_time;
          tasksDurationVector[i].duration = end_time - tasksDurationVector[i].start_time;
          cout << "Task " << taskId << " Executor Time: " << tasksDurationVector[i].duration.count()<<" milliseconds." << endl;
          cout << "Task " << taskId << " VineTask Time: " << tasksDurationVector[i].vineInExecutor<<" milliseconds." << endl;
          cout << "Task " << taskId << " GPU Time:      " << tasksDurationVector[i].gpuTime << " milliseconds." <<endl;
          cout << "Task " << taskId << " Queue Time:    " << tasksDurationVector[i].inQueue.count()<<" milliseconds." << endl;
          cout << "Task " << taskId << " took: " << tasksDurationVector[i].duration.count() + tasksDurationVector[i].inQueue.count()<<" milliseconds to comlete since the beggining of the framework" << endl;
          break;
        }
      }
      tasksFinished++;
    }

    if (status.state() == TASK_LOST ||
        status.state() == TASK_KILLED ||
        status.state() == TASK_FAILED) {
      cout<<"TASK ABORTED"<<endl;
      cout << "Aborting because task " << taskId
           << " is in unexpected state " << status.state()
           << " with reason " << status.reason()
           << " from source " << status.source()
           << " with message '" << status.message() << "'"
           << endl;
      driver->abort();
    }

    if (!implicitAcknowledgements) {
      driver->acknowledgeStatusUpdate(status);
    }

    if (tasksFinished == totalTasks) {
      driver->stop();
      //Calculate average execution time of tasks.
      int totalTaskTime,averageTaskTime,totalVineTime,averageVineTime,totalGpuTime,averageGpuTime;

      totalTaskTime=0;
      totalVineTime=0;
	    totalGpuTime=0;
      for(unsigned int i=0;i<tasksDurationVector.size();i++)
      {
        totalTaskTime+=tasksDurationVector[i].duration.count();
	      totalVineTime+=tasksDurationVector[i].vineInExecutor;
        totalGpuTime+=tasksDurationVector[i].gpuTime;
      }
      averageTaskTime=(double)totalTaskTime/tasksDurationVector.size();
      averageVineTime=(double)totalVineTime/tasksDurationVector.size();
      averageGpuTime=(double)totalGpuTime/tasksDurationVector.size();
      cout << "======================================================"<< endl;
      cout << "Framework analytics" <<endl;
      cout << "Average execution time for each executor is: "<< averageTaskTime << " milliseconds."<<endl;
      cout << "Execution for all the executors is: " << totalTaskTime << " milliseconds." <<endl;
      cout << "Executor analytics" <<endl;
      cout << "Average execution time for each VINE task is: "<< averageVineTime << " milliseconds."<<endl;
      cout << "Average execution time in the GPU is: "<< averageGpuTime << " milliseconds."<<endl;
      cout << "Execution for all the VINE tasks is:" << totalVineTime << " milliseconds." <<endl;
      cout << "Total Execution time in the GPU:" << totalGpuTime << " milliseconds." <<endl;
      cout << "======================================================"<< endl;

        //Write FrameworkAnalytics.txt
        /*
        std::fstream fwAnal("/home/gelagotis/Desktop/FrameworkAnalytics.txt");
        if(fwAnal.is_open()){
            for(unsigned int i=0;i<tasksDurationVector.size();i++){
              struct TaskTimeInfo tmp=tasksDurationVector[i];
              fwAnal << tmp.taskId<< " " << tmp.cudaInExecutor<< " " <<tmp.vineInExecutor << " "<<tmp.duration.count()<< " " << tmp.inQueue.count()<<endl;
            }
            fwAnal<<"sum "<< sumEx<< " "<< sum<<endl;
            fwAnal<<"avg "<<averageEx<< " "<< average<<endl;
            fwAnal.close();
        }
        else cout << "Unable to open file\n";
        */
    }
  }

  virtual void frameworkMessage(SchedulerDriver* driver,
                                const ExecutorID& executorId,
                                const SlaveID& slaveId,
                                const string& data) {

    string buffer;
    std::stringstream ss(data);
    ss >> buffer;
    int taskId= std::stoi(buffer,NULL);
    ss >> buffer;
    long int vineduration=std::stoi(buffer,NULL);
    ss >> buffer;
    float gpuTime=std::stof(buffer,NULL);
    for(unsigned int index=0 ; index<tasksDurationVector.size() ; index++){
      if(tasksDurationVector[index].taskId==taskId){
        tasksDurationVector[index].vineInExecutor=vineduration;
        tasksDurationVector[index].gpuTime = gpuTime;
        break;
      }
    }
    cout<<"TaskId:" << taskId << " Vine Duration: "<< vineduration<< " GPU Time: "<< gpuTime <<endl;
  }

  virtual void slaveLost(SchedulerDriver* driver, const SlaveID& sid) {}

  virtual void executorLost(SchedulerDriver* driver,
                            const ExecutorID& executorID,
                            const SlaveID& slaveID,
                            int status) {}

  virtual void error(SchedulerDriver* driver, const string& message)
  {
    cout << message << endl;
  }

private:
  const bool implicitAcknowledgements;
  const ExecutorInfo executor;
  string role;
  int tasksLaunched;
  int tasksFinished;
  int totalTasks;
};


void usage(const char* argv0, const flags::FlagsBase& flags)
{
  cerr << "Usage: " << Path(argv0).basename() << " [...]" << endl
       << endl
       << "Supported options:" << endl
       << flags.usage();
}


class Flags : public virtual mesos::internal::logging::Flags
{
public:
  Flags()
  {
    add(&Flags::role, "role", "Role to use when registering", "*");
    add(&Flags::master, "master", "ip:port of master to connect");
    add(&Flags::exec_config,"exec_config","Choose which executor to run");
  }

  string role;
  Option<string> master;
  Option<string> exec_config;
};


int main(int argc, char** argv)
{
  // Find this executable's directory to locate executor.

  Option<string> value = os::getenv("MESOS_HELPER_DIR");
  string path;
  string uri;
  string executable;
  path= os::realpath(Path(argv[0]).dirname()).get();
  cout<< "Path to directory is:" << path<<endl;
  //executable="test-executor";
  //uri=path::join(path,executable);
  //cout<<"Path to executable is:"<< uri<<endl;


  Flags flags;

  Try<flags::Warnings> load = flags.load(None(), argc, argv);
  //TODO If exec_config is missing should I stay or should I go?
  if (load.isError()) {
    cerr << load.error() << endl;
    usage(argv[0], flags);
    exit(EXIT_FAILURE);
  } else if (flags.master.isNone()) {
    cerr << "Missing --master" << endl;
    usage(argv[0], flags);
    exit(EXIT_FAILURE);
  }
  /*
    Pick the executor to run.
    If no one is picked the default test-executor is picked
  */
  if(flags.exec_config.isSome()){
    executable= flags.exec_config.get();
    uri=path::join(path,executable);
    cout<<"Path to executable is:"<< uri<<endl;
  }else{
    cout<< "Missing exec_config"<<endl<<"Picked default test-executor"<<endl;
    executable="test-executor";
    uri=path::join(path,executable);
    cout<<"Path to executable is:"<< uri<<endl;
  }


  internal::logging::initialize(argv[0], flags, true); // Catch signals.

  // Log any flag warnings (after logging is initialized).
  foreach (const flags::Warning& warning, load->warnings) {
    LOG(WARNING) << warning.message;
  }

  ExecutorInfo executor;
  executor.mutable_executor_id()->set_value("default");
  executor.mutable_command()->set_value(uri);
  executor.set_name("Test Executor (C++)");
  executor.set_source("cpp_test");

  cout<<"Executor was set successfully"<<endl;

  FrameworkInfo framework;
  framework.set_user(""); // Have Mesos fill in the current user.
  framework.set_name("Test Framework (C++)");
  framework.set_role(flags.role);
  framework.add_capabilities()->set_type(
      FrameworkInfo::Capability::RESERVATION_REFINEMENT);
  framework.add_capabilities()->set_type(
  	    FrameworkInfo::Capability::GPU_RESOURCES);

  value = os::getenv("MESOS_CHECKPOINT");
  if (value.isSome()) {
    framework.set_checkpoint(
        numify<bool>(value.get()).get());
  }

  bool implicitAcknowledgements = true;
  if (os::getenv("MESOS_EXPLICIT_ACKNOWLEDGEMENTS").isSome()) {
    cout << "Enabling explicit acknowledgements for status updates" << endl;

    implicitAcknowledgements = false;
  }

  MesosSchedulerDriver* driver;
  TestScheduler scheduler(implicitAcknowledgements, executor, flags.role);

  if (os::getenv("MESOS_AUTHENTICATE_FRAMEWORKS").isSome()) {
    cout << "Enabling authentication for the framework" << endl;

    value = os::getenv("DEFAULT_PRINCIPAL");
    if (value.isNone()) {
      EXIT(EXIT_FAILURE)
        << "Expecting authentication principal in the environment";
    }

    Credential credential;
    credential.set_principal(value.get());

    framework.set_principal(value.get());

    value = os::getenv("DEFAULT_SECRET");
    if (value.isNone()) {
      EXIT(EXIT_FAILURE)
        << "Expecting authentication secret in the environment";
    }

    credential.set_secret(value.get());

    driver = new MesosSchedulerDriver(
        &scheduler,
        framework,
        flags.master.get(),
        implicitAcknowledgements,
        credential);
  } else {
    framework.set_principal("test-framework-cpp");

    driver = new MesosSchedulerDriver(
        &scheduler,
        framework,
        flags.master.get(),
        implicitAcknowledgements);
  }

  int status = driver->run() == DRIVER_STOPPED ? 0 : 1;

  // Ensure that the driver process terminates.
  driver->stop();

  delete driver;
  return status;
}

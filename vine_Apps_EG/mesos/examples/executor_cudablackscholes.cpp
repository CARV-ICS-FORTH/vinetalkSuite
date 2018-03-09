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
#include <stdlib.h>
#include <string>
#include <chrono>
#include <mesos/executor.hpp>
#include "./cuda4mesos/gpuFunCaller.hpp"
#include <stout/duration.hpp>
#include <stout/os.hpp>

using namespace mesos;

using std::cout;
using std::endl;
using std::string;
class TestExecutor : public Executor
{
public:
  virtual ~TestExecutor() {}

  virtual void registered(ExecutorDriver* driver,
                          const ExecutorInfo& executorInfo,
                          const FrameworkInfo& frameworkInfo,
                          const SlaveInfo& slaveInfo)
  {
    cout << "Registered executor on " << slaveInfo.hostname() << endl;
  }

  virtual void reregistered(ExecutorDriver* driver,
                            const SlaveInfo& slaveInfo)
  {
    cout << "Re-registered executor on " << slaveInfo.hostname() << endl;
  }

  virtual void disconnected(ExecutorDriver* driver) {}


  virtual void launchTask(ExecutorDriver* driver, const TaskInfo& task)
  {
    cout << "Starting task " << task.task_id().value() << endl;
    float totalGpuTime=0;
    float* gpuTimesArray;
    int NUM_ITERATIONS=1000;
    TaskStatus status;
    status.mutable_task_id()->MergeFrom(task.task_id());
    status.set_state(TASK_RUNNING);

    driver->sendStatusUpdate(status);

    cout<<"Calling task in the GPU  "<<endl;
    gpuInterface gpuCall;
    std::chrono::milliseconds cuda_start_time = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch());

    //for(int index=0;index<10;index++)
    //{
       cout<<"Calling BlackScholesMain "<< endl;
       gpuTimesArray=gpuCall.blackScholesMain();
    //}
    std::chrono::milliseconds cuda_end_time = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch());
    std::chrono::milliseconds cuda_duration=cuda_end_time - cuda_start_time;
    for(int index=0;index<NUM_ITERATIONS;index++){
      totalGpuTime+=gpuTimesArray[index];
    }
    //Write executorAnalytics.txt
    std::ofstream file;
    file.open("/home1/public/gelagotis/src/outputFiles/cudaBStasks.txt");
    if(file.is_open()){
        for (int index = 0; index < NUM_ITERATIONS; index++) {
          file<<std::to_string(gpuTimesArray[index])<<endl;
        }
        file.close();
    }
    else cout << "Unable to open file\n";


    cout<<"***********************"<<endl;
    cout <<"Total GPU time is: "<<totalGpuTime<<endl;
    cout<<"CUDA Task lasted for: "<<cuda_duration.count()<<" milliseconds.\n";
    cout<<"***********************"<<endl;

    cout << "Finishing task " << task.task_id().value() << endl;

    status.mutable_task_id()->MergeFrom(task.task_id());
    status.set_state(TASK_FINISHED);

    driver->sendStatusUpdate(status);
    string testMessage=task.task_id().value() + " " + std::to_string(cuda_duration.count()) + " " + std::to_string(totalGpuTime);
    cout << testMessage << endl;
    driver->sendFrameworkMessage(testMessage);
  }

  virtual void killTask(ExecutorDriver* driver, const TaskID& taskId) {}
  virtual void frameworkMessage(ExecutorDriver* driver, const string& data) {
  cout<<"The following arrived from the framework"<<data<<endl;



  }
  virtual void shutdown(ExecutorDriver* driver) {}
  virtual void error(ExecutorDriver* driver, const string& message) {}

private:
 int argument;

};


int main(int argc, char** argv)
{
  cout <<"Executor Is launched"<<endl;
  if(argc == 2){
  cout<<"Passed argument is: "<<argv[1]<<endl;
  }
  TestExecutor executor;
  MesosExecutorDriver driver(&executor);
  return driver.run() == DRIVER_STOPPED ? 0 : 1;
}

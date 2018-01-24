#include <iostream>
#include "FPGAaccelThread.h"
using namespace std;

FPGAaccelThread::FPGAaccelThread(vine_pipe_s * v_pipe, AccelConfig & conf): accelThread(v_pipe, conf) {}
FPGAaccelThread::~FPGAaccelThread() {}

/*initializes the FPGA accelerator*/
bool FPGAaccelThread::acceleratorInit()
{
    cout << "FPGA initalization done." << endl;
    return true;
}
/*Releases the CPU accelerator*/
void FPGAaccelThread::acceleratorRelease()
{
    cout << "FPGA released." << endl;
}

REGISTER_ACCEL_THREAD(FPGAaccelThread)

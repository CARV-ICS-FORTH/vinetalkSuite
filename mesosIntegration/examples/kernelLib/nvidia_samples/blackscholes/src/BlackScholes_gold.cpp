/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <iostream>

#include <math.h>

/* Custom header files */
#include "blackScholesArgs.h"
#include "c_blackScholes.h"
#include "VineLibUtilsCPU.h"

using std::cout;
using std::endl;
using std::cerr;

///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
static double CND(double d) {
  const double A1 = 0.31938153;
  const double A2 = -0.356563782;
  const double A3 = 1.781477937;
  const double A4 = -1.821255978;
  const double A5 = 1.330274429;
  const double RSQRT2PI = 0.39894228040143267793994605993438;

  double K = 1.0 / (1.0 + 0.2316419 * fabs(d));

  double cnd = RSQRT2PI * exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

  if (d > 0) cnd = 1.0 - cnd;

  return cnd;
}

///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
static void BlackScholesBodyCPU(float &callResult, float &putResult,
                                float Sf,  // Stock price
                                float Xf,  // Option strike
                                float Tf,  // Option years
                                float Rf,  // Riskless rate
                                float Vf   // Volatility rate
                                ) {
  double S = Sf, X = Xf, T = Tf, R = Rf, V = Vf;

  double sqrtT = sqrt(T);
  double d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
  double d2 = d1 - V * sqrtT;
  double CNDD1 = CND(d1);
  double CNDD2 = CND(d2);

  // Calculate Call and Put simultaneously
  double expRT = exp(-R * T);
  callResult = (float)(S * CNDD1 - X * expRT * CNDD2);
  putResult = (float)(X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1));
}

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(float *h_CallResult, float *h_PutResult,
                                float *h_StockPrice, float *h_OptionStrike,
                                float *h_OptionYears, float Riskfree,
                                float Volatility, int optN) {
  for (int opt = 0; opt < optN; opt++)
    BlackScholesBodyCPU(h_CallResult[opt], h_PutResult[opt], h_StockPrice[opt],
                        h_OptionStrike[opt], h_OptionYears[opt], Riskfree,
                        Volatility);
}

/**
 * Contains the code that is being executed in the host CPU.
 * It takes as a parameter a vine_task descriptor and returns
 * the status of the task after its execution.
 */
vine_task_state_e hostCodeCPU(vine_task_msg_s *vine_task) {
  std::vector<void *> ioVector;  // Input and output data.
  blackScholesArgs *d_args;      // BlackScholes arguments.
  std::chrono::time_point<std::chrono::system_clock> start, end;
  d_args = (blackScholesArgs *)vine_data_deref(vine_task->args.vine_data);

  // Allocate memory in the device and transfer data.
  Host2CPU(vine_task, ioVector);
#ifdef TIMERS_ENABLED
  start = std::chrono::system_clock::now();
#endif

  cout << "BlackScholes execution in CPU." << endl;

  // The actual BlackScholes function call.
  BlackScholesCPU((float *)vine_data_deref(vine_task->io[3].vine_data),  // OUT
                  (float *)vine_data_deref(vine_task->io[4].vine_data),  // OUT
                  (float *)vine_data_deref(vine_task->io[0].vine_data),  // IN
                  (float *)vine_data_deref(vine_task->io[1].vine_data),  // IN
                  (float *)vine_data_deref(vine_task->io[2].vine_data),  // IN
                  d_args->Riskfree, d_args->Volatility, d_args->optN);
#ifdef TIMERS_ENABLED

  end = std::chrono::system_clock::now();

  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::chrono::duration<double> elapsed_seconds = end - start;

  cout << "BlackScholesCPU kernel execution time: " << elapsed_seconds.count()
       << " secs." << endl;
#endif

  // Memory transfer from local memory to shared segment.
  CPU2Host(vine_task, ioVector);

  // Free allocated memory.
  CPUMemFree(ioVector);

  return vine_task_stat(vine_task, 0);
}

/*
 * Mapping of function "BlackScholesCPU" with "hostCodeCPU" that executes a
 * blackScholes task.
 */
VINE_PROC_LIST_START()
VINE_PROCEDURE("BlackScholes", CPU, hostCodeCPU, sizeof(blackScholesArgs))
VINE_PROC_LIST_END()

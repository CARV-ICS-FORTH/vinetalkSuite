/**
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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <curand.h>

//#include "curand_kernel.h"
#include "helper_cuda.h"
#include "vine_talk.h"
#include "monteCarloArgs.h"
#include "VineLibUtilsCPU.h"

using std::cout;
using std::endl;
using std::cerr;

////////////////////////////////////////////////////////////////////////////////
// Common types
////////////////////////////////////////////////////////////////////////////////
#include "MonteCarlo_common.h"

////////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for Monte Carlo results validation
////////////////////////////////////////////////////////////////////////////////
#define A1 0.31938153
#define A2 -0.356563782
#define A3 1.781477937
#define A4 -1.821255978
#define A5 1.330274429
#define RSQRT2PI 0.39894228040143267793994605993438

// Polynomial approxiamtion of
// cumulative normal distribution function
double CND(double d) {
  double K = 1.0 / (1.0 + 0.2316419 * fabs(d));

  double cnd = RSQRT2PI * exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

  if (d > 0) cnd = 1.0 - cnd;

  return cnd;
}

// Black-Scholes formula for call value
extern "C" void BlackScholesCall(float &callValue, TOptionData optionData) {
  double S = optionData.S;
  double X = optionData.X;
  double T = optionData.T;
  double R = optionData.R;
  double V = optionData.V;

  double sqrtT = sqrt(T);
  double d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
  double d2 = d1 - V * sqrtT;
  double CNDD1 = CND(d1);
  double CNDD2 = CND(d2);
  double expRT = exp(-R * T);

  callValue = (float)(S * CNDD1 - X * expRT * CNDD2);
}

////////////////////////////////////////////////////////////////////////////////
// CPU Monte Carlo
////////////////////////////////////////////////////////////////////////////////
static double endCallValue(double S, double X, double r, double MuByT,
                           double VBySqrtT) {
  double callValue = S * exp(MuByT + VBySqrtT * r) - X;
  return (callValue > 0) ? callValue : 0;
}

extern "C" void MonteCarloCPU(TOptionValue *callValue, TOptionData *optionData,
                              float *h_Samples, int pathN, int optionN) {
  float *samples;
  curandGenerator_t gen;

  checkCudaErrors(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  unsigned long long seed = 1234ULL;
  checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, seed));

  if (h_Samples != NULL) {
    samples = h_Samples;
  } else {
    samples = (float *)malloc(pathN * sizeof(float));
    checkCudaErrors(curandGenerateNormal(gen, samples, pathN, 0.0, 1.0));
  }

  // for(int i=0; i<10; i++) printf("CPU sample = %f\n", samples[i]);

  for (int i = 0; i < optionN; ++i) {
    const double S = optionData[i].S;
    const double X = optionData[i].X;
    const double T = optionData[i].T;
    const double R = optionData[i].R;
    const double V = optionData[i].V;
    const double MuByT = (R - 0.5 * V * V) * T;
    const double VBySqrtT = V * sqrt(T);
    double sum = 0, sum2 = 0;

    for (int pos = 0; pos < pathN; pos++) {
      double sample = samples[pos];
      double callValue = endCallValue(S, X, sample, MuByT, VBySqrtT);
      sum += callValue;
      sum2 += callValue * callValue;
    }

    // Derive average from the total sum and discount by riskfree rate
    callValue[i].Expected = (float)(exp(-R * T) * sum / (double)pathN);
    // Standart deviation
    double stdDev = sqrt(((double)pathN * sum2 - sum * sum) /
                         ((double)pathN * (double)(pathN - 1)));
    // Confidence width; in 95% of all cases theoretical value lies within these
    // borders
    callValue[i].Confidence =
        (float)(exp(-R * T) * 1.96 * stdDev / sqrt((double)pathN));
  }

  if (h_Samples == NULL) free(samples);
  checkCudaErrors(curandDestroyGenerator(gen));
}

/**
 * Contains the code that is being executed in the host CPU.
 * It takes as a parameter a vine_task descriptor and returns
 * the status of the task after its execution.
 */
vine_task_state_e hostCodeCPU(vine_task_msg_s *vine_task) {
  std::vector<void *> ioVector;  // Input and output data.
  monteCarloArgs *d_args;        // BlackScholes arguments.

  d_args = (monteCarloArgs *)vine_data_deref(vine_task->args.vine_data);

  // Allocate memory in the device and transfer data.
  Host2CPU(vine_task, ioVector);

  cout << "MonteCarlo execution in CPU." << endl;

#ifdef TIMERS_ENABLED
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
#endif

  // The actual BlackScholes function call.
  MonteCarloCPU((TOptionValue *)vine_data_deref(vine_task->io[1].vine_data),
                (TOptionData *)vine_data_deref(vine_task->io[0].vine_data),
                NULL,  //(float *) vine_data_deref(vine_task->io[0]),
                d_args->pathN, d_args->optionCount);
#ifdef TIMERS_ENABLED

  end = std::chrono::system_clock::now();

  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::chrono::duration<double> elapsed_seconds = end - start;

  cout << "MonteCarlo CPU kernel execution time: " << elapsed_seconds.count()
       << " secs." << endl;
#endif

  // Memory transfer from local memory to shared segment.
  CPU2Host(vine_task, ioVector);

  // Free allocated memory.
  CPUMemFree(ioVector);

  return vine_task_stat(vine_task, 0);
}

/*
 * Mapping of function "MonteCarloGPU" with "hostCodeGPU" that executes a
 * blackScholes task.
 */
VINE_PROC_LIST_START()
VINE_PROCEDURE("MonteCarlo", CPU, hostCodeCPU, sizeof(monteCarloArgs))
VINE_PROC_LIST_END()

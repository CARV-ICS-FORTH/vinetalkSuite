#ifndef MONTECARLOARGS_H
#define MONTECARLOARGS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int gridSize;
    int optionCount;        // Option count for this plan
    int pathN;              // Pseudorandom samples count
} monteCarloArgs;

#ifdef __cplusplus
}
#endif

#endif

#ifndef OPENCL_DARKGRAY_H
#define OPENCL_DARKGRAY_H

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>


#ifdef RD_WG_SIZE_0_0
#define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE RD_WG_SIZE
#else
#define BLOCK_SIZE 16
#endif

bool opencl_darkGray(const int, const int, const unsigned char *, unsigned char *);


#endif

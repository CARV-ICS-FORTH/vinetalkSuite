cmake_minimum_required(VERSION 2.8)

# project name
project(VINE)

# we need CUDA
find_package(CUDA QUIET)

# disable builds in source tree
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

#set(scheduler WeightedRoundRobinScheduler CACHE STRING "Scheduler Class to be used.")
#set_property(CACHE scheduler PROPERTY STRINGS WeightedRoundRobinScheduler VineRoundRobinScheduler)
#set(CMAKE_CXX_FLAGS "-DSCHED_CLASS=${scheduler} ${CMAKE_CXX_FLAGS}")


# export compile commands in compile_commands.json, for use with IDEs
# and editors
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#set options
option(DATA_TRANSFER "Disable/Enable prints for data size" OFF)
option(ELASTICITY "Disable/Enable elasticity in accelerator groups" )
option(BREAKDOWNS_CONTROLLER "Disable/Enable controller breakdowns" OFF)
option(POWER_TIMER "Disable/Enable power clock" OFF)
option(REDUCTION_UNITS_FROM_EXEC_TIME "Select if reduction units (for Weighted round robin) are according to task execution time or statically defined ")
SET(GPU_OCCUPANCY "OFF" CACHE STRING "OFF/PER_KERNEL/SAMPLING")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/definesEnable.h.cmake ${CMAKE_CURRENT_SOURCE_DIR}/include/definesEnable.h)

string(COMPARE EQUAL ${GPU_OCCUPANCY} PER_KERNEL _res1)
if(_res1)
	#message("Using per kernel mode")
	add_definitions(-DSM_OCCUPANCY)
	set(SM_OCCUPANCY ON)
endif()
string(COMPARE EQUAL ${GPU_OCCUPANCY} SAMPLING _res1)
if(_res1)
	#message("Using sampling")
	add_definitions(-DSAMPLE_OCCUPANCY)
	set(SM_OCCUPANCY ON)
endif()


# the default CFLAGS and CUDA flags
set(CMAKE_CXX_FLAGS "-Wall -std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb3")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -finline-functions -Werror")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/occupancy)

cmake_policy(SET CMP0015 NEW)

# FIXME Not really automatic, but the user has a chance to select it.
set(VINE_TALK_BUILD_PATH "../vine_talk/build" CACHE STRING "Vinetalk build location")
include_directories(${VINE_TALK_BUILD_PATH}/include)
link_directories(${VINE_TALK_BUILD_PATH}/build)
link_directories("/usr/local/cuda-8.0/extras/CUPTI/lib64")

#List of 'common' source files
set(CONTROLLER_SOURCES
src/vine_controller.cpp
src/Config.cpp
src/VineLibMgr.cpp
src/Utilities.cpp
#Accel Threads
src/accelThread.cpp
src/CPUaccelThread.cpp
src/FPGAaccelThread.cpp
src/NanoCoreaccelThread.cpp
#Schedulers
src/Scheduler.cpp
src/VineRoundRobinScheduler.cpp
)

SET(CONTROLLER_LIBS)

option(ENABLE_SDACCEL "Enable support for Xilinx SDAaccel devices/library." OFF)
if(ENABLE_SDACCEL)
    set(XILINX_SDACCEL "" CACHE STRING "Xilinx SDAaccel installation directory")
    set(CONTROLLER_SOURCES ${CONTROLLER_SOURCES} src/SDAaccelThread.cpp src/xcl.cpp)
    link_directories(${XILINX_SDACCEL}/runtime/lib/x86_64)
    set(CONTROLLER_LIBS xilinxopencl lmx6.0)
#	 -lxilinxopencl -llmx6.0
else()
    unset(XILINX_SDACCEL CACHE)
endif(ENABLE_SDACCEL)

# if cuda is available build the cuda version as well
if(CUDA_FOUND)
    if(CMAKE_MAJOR_VERSION LESS 3)
	  set(CUDA_NVCC_FLAGS " -Wno-deprecated-gpu-targets -std=c++11 ${CUDA_NVCC_FLAGS}")
	  #set(CUDA_NVCC_FLAGS " -Wno-deprecated-gpu-targets -std=c++11 -lnvidia-ml ${CUDA_NVCC_FLAGS}")
    endif(CMAKE_MAJOR_VERSION LESS 3)
    add_custom_target(run_nvvp
       COMMAND nvvp ./vine_controller ../../vine-applications
       DEPENDS vine_controller
       WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    

    if(SM_OCCUPANCY)
      cuda_add_executable(vine_controller
        ${CONTROLLER_SOURCES}
        src/GPUaccelThread.cpp
        src/prepareGPU.cu
        src/occupancy.cu
    )
    else(SM_OCCUPANCY)
      cuda_add_executable(vine_controller
        ${CONTROLLER_SOURCES}
        src/GPUaccelThread.cpp
        src/prepareGPU.cu
      )
    endif(SM_OCCUPANCY)

    cuda_add_executable(ResetGpu src/ResetGpu.cpp)

else(CUDA_FOUND)
    add_executable(vine_controller
      ${CONTROLLER_SOURCES}
    )

endif(CUDA_FOUND)
target_link_libraries(vine_controller ${CONTROLLER_LIBS})

add_custom_target(run
  COMMAND cd .. && ./build/vine_controller example_config || cd -
  DEPENDS vine_controller
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )



if(SM_OCCUPANCY)
target_link_libraries(vine_controller
  -lvine
  -ldl
  -lrt
  -lpthread
  -lcuda
  -lcupti)
else(SM_OCCUPANCY)
target_link_libraries(vine_controller
  -lvine
  -ldl
  -lrt
  -lpthread)
endif(SM_OCCUPANCY)


################################################################################
#
# Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:
#
# This source code is subject to NVIDIA ownership rights under U.S. and
# international Copyright laws.
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
# OR PERFORMANCE OF THIS SOURCE CODE.
#
# U.S. Government End Users.  This source code is a "commercial item" as
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
# "commercial computer software" and "commercial computer software
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
# and is provided to the U.S. Government only as a commercial end item.
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
# source code with only those rights set forth herein.
#
################################################################################
#
# Makefile project only supported on Mac OS X and Linux Platforms)
#
################################################################################

# Location of the CUDA Toolkit

#Type make machine=hydra when you want to run in hdra

CUDA_VERSION=`which nvcc | cut -f 4 -d '/'`
#`pkg-config --list-all | grep cuda | grep -v 'cudart'| cut -f 1 -d ' '`

CUDA_PATH = /usr/local/$(CUDA_VERSION)

HOST_COMPILER = g++
CC = gcc

NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER) --default-stream per-thread

# internal flags
NVCCFLAGS = -std=c++11 -m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 \
-gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 \
-gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52

CCFLAGS     := -std=c++11
LDFLAGS     :=


# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))


################################################################################

SAMPLE_ENABLED := 1

# Gencode arguments
ifeq ($(TARGET_ARCH),armv7l)
SMS ?= 35 37 50 52
else
SMS ?= 35 37 50 52
endif

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

ALL_CCFLAGS += -po maxrregcount=16

ifeq ($(SAMPLE_ENABLED),0)
EXEC ?= @echo "[@]"
endif

################################################################################
CXX = $(HOST_COMPILER) 

VINETALK  = ../../../../vine_talk
VINECONTR = ../../../../vine_controller
JOBGEN = ../../../job_generator

INCLUDES  := -I../common/inc -I./include 
INCLUDES += -I$(VINETALK)/build/include -I$(VINECONTR)/include -I$(JOBGEN)/include
INCLUDES += -I$(CUDA_PATH)/include

LIBFLAGS = -shared -fPIC
NVCCLIB = --compiler-options '-fPIC' --shared

LCUDA = -L$(CUDA_PATH)/lib64 -lcuda -lcudart
#LFLVINE = -Wl,-rpath,$(VINETALK)/build -L$(VINETALK)/build -lvine -lrt
LFLVINE = -lvine -lrt
LDFLAGS += -lc -ldl -lm
################################################################################

# Target rules
all: clean libs build

build: BlackScholes 
	mkdir -p bin
	mv $^ bin

libs: c_BlackScholes.so cu_BlackScholes.so
	mkdir -p lib
	mv $^ lib
	
c_BlackScholes.so: src/BlackScholes_gold.cpp
	$(CXX) $(INCLUDES) $(CCFLAGS) $(LIBFLAGS) $< -o $@ $(LDFLAGS) -DLIBRARY_BUILD

cu_BlackScholes.so: src/BlackScholes.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< $(LFLVINE) -o $@ $(NVCCLIB) -DLIBRARY_BUILD
	
BlackScholes: src/BlackScholes.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< $(LFLVINE) -o $@

run_gpu: build
	$(EXEC) LD_LIBRARY_PATH=.:$(CUDA_PATH)/lib64 ./bin/BlackScholes 1

run_cpu: build
	$(EXEC) ./bin/BlackScholes 3


clean:
	rm -rf bin lib



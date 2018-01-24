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

#Type make machine=hydra when you want to run in hdra
CUDA_VERSION=`which nvcc | cut -f 4 -d '/'`
#`pkg-config --list-all | grep cuda | grep -v 'cudart'| cut -f 1 -d ' '`

CUDA_PATH = /usr/local/$(CUDA_VERSION)

# Location of the CUDA Toolkit
CUDALOCATION =-I"$(CUDA_PATH)/include"

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le))
    TARGET_SIZE := 64
else ifeq ($(TARGET_ARCH),armv7l)
    TARGET_SIZE := 32
else
    $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif
ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq (,$(filter $(HOST_ARCH)-$(TARGET_ARCH),aarch64-armv7l x86_64-armv7l x86_64-aarch64 x86_64-ppc64le))
        $(error ERROR - cross compiling from $(HOST_ARCH) to $(TARGET_ARCH) is not supported!)
    endif
endif

# host compiler
HOST_COMPILER = g++
CC = gcc

NVCC        := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER) --default-stream per-thread

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     := -g -std=c++11
LDFLAGS     :=

# build flags
ifeq ($(TARGET_OS),darwin)
    LDFLAGS += -rpath $(CUDA_PATH)/lib
    CCFLAGS += -arch $(HOST_ARCH)
else ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-armv7l-linux)
    LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
    CCFLAGS += -mfloat-abi=hard
else ifeq ($(TARGET_OS),android)
    LDFLAGS += -pie
    CCFLAGS += -fpie -fpic -fexceptions
endif

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
        ifneq ($(TARGET_FS),)
            GCCVERSIONLTEQ46 := $(shell expr `$(HOST_COMPILER) -dumpversion` \<= 4.6)
            ifeq ($(GCCVERSIONLTEQ46),1)
                CCFLAGS += --sysroot=$(TARGET_FS)
            endif
            LDFLAGS += --sysroot=$(TARGET_FS)
            LDFLAGS += -rpath-link=$(TARGET_FS)/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-gnueabihf
        endif
    endif
endif

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

VINETALK  = ../../../../vine_talk/build
VINECONTR = ../../../../vine_controller
JOBGEN = ../../../job_generator

#LDDFLAGS = -Wl,-rpath,$(VINETALK)/build -L$(VINETALK)/build -lvine -lrt -lpthread -lX11
LDDFLAGS = -lvine -lrt -lpthread -lX11

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(CCFLAGS)
#ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))
ALL_LDFLAGS += $(LDDFLAGS)

# Common includes and paths for CUDA
INCLUDES  := -I../common/inc -I./include
INCLUDES  += -I$(VINECONTR)/include -I$(VINETALK)/include -I$(JOBGEN)/include
LIBRARIES :=

NVCCLIB = --ptxas-options=-v --compiler-options '-fPIC' --shared
################################################################################

SAMPLE_ENABLED := 1

# Gencode arguments
ifeq ($(TARGET_ARCH),armv7l)
SMS ?= 20 30 32 35 37 50 52
else
SMS ?= 20 30 35 37 50 52
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

ifeq ($(SAMPLE_ENABLED),0)
EXEC ?= @echo "[@]"
endif

################################################################################

# Target rules
all: build libs

build: convolutionSeparable
	mkdir -p bin
	mv $^ bin

main.o: src/main.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

convolutionSeparable: main.o
	$(EXEC) $(NVCC) $(LDDFLAGS) -o $@ $+ $(LIBRARIES)

##################### .so file ########################
libs: lib lib/convolutionSeparable.so 

lib:
	mkdir -p lib
		
lib/convolutionSeparable.so: src/convolutionSeparable.cu src/convolutionSeparable_gold.cpp
	 $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(LDDFLAGS) -o $@ $^ $(NVCCLIB) -DLIBRARY_BUILD

##################### run #######################
run: build
	$(EXEC) ./bin/convolutionSeparable
	
run_gpu: build
	$(EXEC) ./bin/convolutionSeparable accelType=1
	
run_cpu: build
	$(EXEC) ./bin/convolutionSeparable accelType=3

clean:
	rm -f convolutionSeparable convolutionSeparable.o convolutionSeparable_gold.o main.o
	-rm -rf lib
	rm -rf *.o
	-rm -rf bin
	
clobber: clean

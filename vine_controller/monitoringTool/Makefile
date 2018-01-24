#Type make direct=y when you want to enable instant printing

# Gencode arguments
SMS ?= 50

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
CPPFLAGS = -Wall -Wstrict-prototypes -pipe -g

NVCCFLAGS  = -lcuda -lcudart -lnvidia-ml -std=c++11 -g
ifeq ($(ALL),y)
NVCCFLAGS += -DALLCPUs
else
NVCCFLAGS += 
endif

CUDA_VERSION=cuda-8.0
CUDA_PATH = /usr/local/$(CUDA_VERSION)
NVCC = $(CUDA_PATH)/bin/nvcc -ccbin g++
all: monitoringThread

count.o: count.c count.h
	g++ $(CPPFLAGS) -o $@ -c $< 

common.o: common.c rd_stats.h common.h
	g++ $(CPPFLAGS) -o $@ -c $< 

monitoringThread.o: monitoringThread.cu  
	$(NVCC) $(GENCODE_FLAGS) $(NVCCFLAGS) -o $@ -c $<

monitoringThread: monitoringThread.o common.o count.o
	$(NVCC) -L$(CUDA_PATH)/lib64 $(GENCODE_FLAGS) $(NVCCFLAGS) -o $@ $^

clean:
	rm -rf *.o
	rm -rf monitoringThread

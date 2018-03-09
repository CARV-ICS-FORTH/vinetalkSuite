#Type make machine=hydra when you want to run in hdra
#Type make cuda=7 if you use cuda version 7
#else type make cuda=7.5

ifeq ($(machine),hydra)
    CXX =  /opt/gcc-4.8.4/bin/g++
else
    CXX = g++
endif

CUDA_VERSION=`which nvcc | cut -f 4 -d '/'`
#`pkg-config --list-all | grep cuda | grep -v 'cudart'| cut -f 1 -d ' '`

CUDA = /usr/local/$(CUDA_VERSION)
VINETALK = ../../../vine_talk
CONTROLLER = ../../../vine_controller
JOBGEN = ../../job_generator

CXXFLAGS = -g -std=c++11 -I./include -I$(VINETALK)/build/include -I$(VINETALK)/src -I$(CONTROLLER)/include -I$(JOBGEN)/include
CUDAFLAGS = -I$(CUDA)/include
#LDDFLAGS = -Wl,-rpath,$(VINETALK)/build -L$(VINETALK)/build -lvine -lrt -lpthread -lX11
LDDFLAGS = -lvine -lrt -lpthread -lX11

NVCC = $(CUDA)/bin/nvcc -ccbin $(CXX) --default-stream per-thread

all: darkGray libs

darkGray: bin bin/vine_darkGray bin/compare

bin: 
	mkdir -p $@

lib:
	mkdir -p $@ 

libs: lib lib/c_darkGray.so lib/cu_darkGray.so

bin/vine_darkGray: src/vine_darkGray.o
	$(CXX) $< $(LDDFLAGS) -o $@

bin/compare: src/compare.o
	$(CXX) $< $(LDDFLAGS) -o $@


lib/c_darkGray.so: src/c_darkGray.cpp
	$(CXX) -fPIC -shared $< $(CXXFLAGS) $(LDDFLAGS) -o $@
	
lib/cu_darkGray.so: src/cu_darkGray.cu
	$(NVCC) --ptxas-options=-v --compiler-options '-fPIC' $(CXXFLAGS) $(LDDFLAGS) -o $@ --shared $< -DLIBRARY_BUILD
	
runGPU_1: 
	./bin/vine_darkGray input_images/image01.jpg out.jpg 1
runGPU_2: 
	./bin/vine_darkGray input_images/image03.jpg out.jpg 1
runGPU_3: 
	./bin/vine_darkGray input_images/image06.jpg out.jpg 1
runCPU:
	./bin/vine_darkGray input_images/image01.jpg out.jpg 3

runCompare:
	./bin/compare input_images/image01.jpg out.jpg

clean:
	-rm -rf bin
	-rm -rf lib
	-rm  src/vine_darkGray.o
	-rm -rf output_images
	-rm  src/compare.o
	-rm -rf *.jpg


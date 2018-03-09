CUDA_VERSION=`which nvcc | cut -f 4 -d '/'`
#`pkg-config --list-all | grep cuda | grep -v 'cudart'| cut -f 1 -d ' '`

CUDA = /usr/local/cuda
VINETALK = ../../../vine_talk
CONTROLLER = ../../../vine_controller
#JOBGEN = ../../job_generator

CXXFLAGS = -g -std=c++11 -I./include -I$(VINETALK)/build/include -I$(VINETALK)/src -I$(CONTROLLER)/include 
CUDAFLAGS = -I$(CUDA)/include
#LDDFLAGS = -Wl,-rpath,$(VINETALK)/build -L$(VINETALK)/build -lrt -lpthread -lX11
LDDFLAGS = -lrt -lpthread -lX11

NVCC = $(CUDA)/bin/nvcc -ccbin $(CXX) --default-stream per-thread

all: ./include/CImg.h darkGray libs

CImg:
	git clone https://github.com/dtschump/CImg.git

./include/CImg.h: CImg
	ln -s ../CImg/CImg.h $@

darkGray: bin bin/vine_darkGray bin/compare

bin:
	mkdir -p $@

lib:
	mkdir -p $@

libs: lib lib/c_darkGray.so lib/cu_darkGray.so

bin/vine_darkGray: src/vine_darkGray.o
	$(CXX) $< $(LDDFLAGS) $(VINETALK)/build/libvine_st.a -o $@

bin/compare: src/compare.o
	$(CXX) $^ $(LDDFLAGS) -o $@


lib/c_darkGray.so: src/c_darkGray.cpp
	$(CXX) -fPIC -shared $< $(CXXFLAGS) $(LDDFLAGS) -o $@ -DLIBRARY_BUILD

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

runCompare: compare
	./bin/compare input_images/image01.jpg out.jpg

clean:
	-rm -rf CImg
	-unlink include/CImg.h
	-rm -rf bin
	-rm -rf lib
	-rm  src/vine_darkGray.o
	-rm -rf output_images
	-rm  src/compare.o
	-rm -rf *.jpg

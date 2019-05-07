#C compiler
CC = gcc
CXX = g++
LINK = g++ -fPIC


# INTEL FPGA
OPENCL_DIR = /home/intelFPGA_pro/18.0/hld
OPENCL_INC = $(OPENCL_DIR)/host/include/
OPENCL_LIB = $(OPENCL_DIR)/linux64/lib/ -lOpenCL
#endif


VINETALK  = ../../../vine_talk
VINECONTR = ../../../vine_controller
JOBGEN = ../../job_generator

CXXFLAGS = -g -std=c++11 -Dcimg_display=0
CXXFLAGS += -I./include -I$(VINETALK)/build/include -I$(VINETALK)/src -I$(VINECONTR)/include -I$(JOBGEN)/include 
OPENCL_INCLUDE = -I$(OPENCL_INC)
OPENCL_LIBS = -L$(OPENCL_LIB)

LDDFLAGS = -lrt -lpthread #-lX11
OPENCLFLAGS = -lOpenCL

all: ./include/CImg.h darkGray libs

./include/CImg.h:
ifeq ($(wildcard ~/CImg),)
		git clone https://github.com/dtschump/CImg.git
		ln -s ../CImg/CImg.h $@
endif

darkGray: bin bin/vine_darkGray bin/compare

libs: lib lib/opencl_darkGray.so

bin:
	mkdir -p $@

lib:
	mkdir -p $@


bin/vine_darkGray: src/vine_darkGray.o
	$(CXX) $< $(LDDFLAGS) $(VINETALK)/build/libvine_st.a -o $@

bin/compare: src/compare.o
	$(CXX) $^ $(LDDFLAGS) -o $@


lib/c_darkGray.so: src/c_darkGray.cpp
	$(LINK) -shared $< $(CXXFLAGS) $(LDDFLAGS) -o $@ -DLIBRARY_BUILD

lib/opencl_darkGray.so: src/opencl_darkGray.cpp
	$(LINK) -shared $< $(CXXFLAGS) $(OPENCL_INCLUDE) $(OPENCL_LIBS) $(LDDFLAGS) $(OPENCLFLAGS) -o $@ -DLIBRARY_BUILD

run:
	./bin/vine_darkGray input_images/image01.jpg out.jpg 7

runCompare: bin/compare
	./bin/compare input_images/image01.jpg out.jpg

clean:
	#-rm -rf CImg
	#-unlink include/CImg.h
	-rm -rf bin
	-rm -rf lib
	-rm  src/vine_darkGray.o
	-rm -rf output_images
	-rm  src/compare.o
	-rm -rf *.jpg

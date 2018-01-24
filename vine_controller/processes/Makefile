COMMON=../../vine-applications/vine_applications/djinn_tonic/djinn-1.0/common

include $(COMMON)/Makefile.config
VINE_BASE = ../../
VINE_TALK = ${VINE_BASE}/vine_talk
VINE_CONTROLLER=$(VINE_BASE)/vine_controller
VINE_INC = -I$(VINE_TALK)/build/include -I$(VINETALK)/src

VINE_APPS = $(VINE_BASE)/vine-applications/vine_applications
SENNA_INC = -I$(VINE_APPS)/djinn_tonic/djinn-1.0/tonic-suite/nlp/src/

CUDA_VERSION=cuda-8.0
NVCCFLAGS  = -lcuda -lcudart 
NVCC = $(CUDA_PATH)/bin/nvcc -ccbin g++
CUDA_PATH = /usr/local/$(CUDA_VERSION)
SRC = ../../vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/src
SOURCES = $(wildcard $(SRC)/SENNA_*.cpp) $(COMMON)/src/tonic.cpp
OBJECTS = $(SOURCES:.cpp=.o)

VINE_LIB = -lvine

CXX = g++ -g

CXX_FLAGS  = -O3 \
	     -fpermissive \
	     -I$(CUDA_PATH)/include\
	     $(VINE_INC)\
	     -I$(COMMON)/include \
	     -I$(CAFFE)/include \
	     -I$(COMMON)/../tonic-suite/img/src


LINK_FLAGS =   -lrt \
	       -lpthread \
	       -lPocoUtil -lPocoFoundation -lPocoNet\
	       $(NVCCFLAGS)\
	       $(VINE_LIB)\
	       -lopencv_core \
	       -lopencv_highgui \
	       -lopencv_imgproc \
	       -lopencv_objdetect \
	       $(COMMON)/libtonic.a \
	       $(COMMON)/../tonic-suite/img/libflandmark_static.a \
	       -lboost_program_options \
	       -lboost_filesystem \
	       -lboost_system \
	       -lglog \
	       $(CAFFE)/lib/libcaffe.so

SOURCES = $(wildcard $(SRC)/SENNA_*.cpp) $(COMMON)/src/tonic.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Main target
accelProcess: accelProcess.o 
	$(CXX) accelProcess.o -L$(CUDA_PATH)/lib64 $(NVCCFLAGS) -o accelProcess $(OBJECTS) $(LINK_FLAGS)

# To obtain object files
accelProcess.o: accelProcess.cpp
	$(CXX) -c $(CXX_FLAGS) $< $(SENNA_INC) -o $@

# To obtain object files
$(OBJECTS): $(SOURCES) 
	$(CXX) -c $(CXX_FLAGS) $(EXTRA_FLAGS) $< $(VINE_INC) -o $@

# To remove generated files
clean:
	rm -f accelProcess 
	rm -f accelProcess.o 


#all:
#	g++ accelProcess.cpp -o accelProcess -lPocoUtil -lPocoFoundation -lPocoNet

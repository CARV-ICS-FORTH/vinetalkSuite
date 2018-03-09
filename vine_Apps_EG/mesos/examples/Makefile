#
#
#
#
#TODO: This Makefile works only for mesos-1.4.0 version
#
#
#
# 

CUDA_INSTALL_PATH := /usr/local/cuda
NVCC := $(CUDA_INSTALL_PATH)/bin/nvcc
CUDA_INCLUDES=-I. -I$(CUDA_INSTALL_PATH)/include, -I$(CUDA_INSTALL_PATH )/samples/common/inc,  -I$(CUDA_INSTALL_PATH )/targets/x86_64-linux/include
NVCCFLAGS+=$(CUDA_INCLUDES)
#LIB_CUDA:=  -L$(CUDA_INSTALL_PATH)/lib64/ -lcudart -lcurand
LIB_CUDA:=  -L$(CUDA_INSTALL_PATH)/lib64 -lcudart
MY_CUDA_DIR:= ~/src/cuda4mesos
CUDA_OBJS:=             $(MY_CUDA_DIR)/simpleCudaPrintf.cu.o \
			$(MY_CUDA_DIR)/BlackScholesobj.o \
                        $(MY_CUDA_DIR)/convolutionSeparableMain.o $(MY_CUDA_DIR)/convolutionSeparable.cu.o $(MY_CUDA_DIR)/convolutionSeparable_gold.o \
                        $(MY_CUDA_DIR)/histogramMain.o $(MY_CUDA_DIR)/histogram256.cu.o $(MY_CUDA_DIR)/histogram64.cu.o $(MY_CUDA_DIR)/histogram_gold.o \
			$(MY_CUDA_DIR)/MonteCarloMultiGPUobj.o
VAC_DIR:= ./kernelLib/objects
VAC_OBJS:=      $(VAC_DIR)/BlackScholes.o \
                $(VAC_DIR)/vine_convolutionSeparable.o \
                $(VAC_DIR)/vine_histogram.o \
                $(VAC_DIR)/vine_MonteCarloMultiGPU.o \
		$(VAC_DIR)/vine_darkGray.o 
#                $(VAC_DIR)/c_darkGray.o \
#                $(VAC_DIR)/cu_darkGray.o \
#                $(VAC_DIR)/vine_darkGray.o

MESOS_TOP_DIR = ../../mesos
top_srcdir:= $(MESOS_TOP_DIR)
#srcdir = $(top_srcdir)/src
srcdir= .

STOUT := 3rdparty/stout
MESOS_HEADERS := $(MESOS_TOP_DIR)/include
CXX= g++
CXXLD = $(CXX)
CPPFLAGS = -isystem /usr/include/subversion-1 -isystem /usr/include/apr-1 -isystem /usr/include/apr-1.0	 -isystem $(MESOS_HEADERS)
CXXFLAGS = -g1 -O0 -Wno-unused-local-typedefs -std=c++11
DEFS = -DPACKAGE_NAME=\"mesos\" -DPACKAGE_TARNAME=\"mesos\" -DPACKAGE_VERSION=\"1.4.0\" -DPACKAGE_STRING=\"mesos\ 1.4.0\" -DPACKAGE_BUGREPORT=\"\" -DPACKAGE_URL=\"\" -DPACKAGE=\"mesos\" -DVERSION=\"1.4.0\" -DSTDC_HEADERS=1 -DHAVE_SYS_TYPES_H=1 -DHAVE_SYS_STAT_H=1 -DHAVE_STDLIB_H=1 -DHAVE_STRING_H=1 -DHAVE_MEMORY_H=1 -DHAVE_STRINGS_H=1 -DHAVE_INTTYPES_H=1 -DHAVE_STDINT_H=1 -DHAVE_UNISTD_H=1 -DHAVE_DLFCN_H=1 -DLT_OBJDIR=\".libs/\" -DHAVE_CXX11=1 -DHAVE_PTHREAD_PRIO_INHERIT=1 -DHAVE_PTHREAD=1 -DHAVE_LIBZ=1 -DHAVE_FTS_H=1 -DHAVE_APR_POOLS_H=1 -DHAVE_LIBAPR_1=1 -DHAVE_LIBCURL=1 -DHAVE_LIBSASL2=1 -DHAVE_SVN_VERSION_H=1 -DHAVE_LIBSVN_SUBR_1=1 -DHAVE_SVN_DELTA_H=1 -DHAVE_LIBSVN_DELTA_1=1 -DHAVE_LIBZ=1

GOOGLETEST_VERSION = 1.8.0
PROTOBUF_VERSION = 3.3.0
PROTOBUF = 3rdparty/protobuf-$(PROTOBUF_VERSION)

GOOGLETEST = 3rdparty/googletest-release-$(GOOGLETEST_VERSION)
GMOCK = $(GOOGLETEST)/googlemock
GTEST = $(GOOGLETEST)/googletest


prefix = /usr/local
exec_prefix = ${prefix}
libdir = ${exec_prefix}/lib
# Protect many of the functions with stack guards
# (either -fstack-protector-strong or -fstack-protector depending on compiler support).
# Produce position independent code when appropriate.
am__append_3 = -fstack-protector -fPIC -fPIE
libexecdir = ${exec_prefix}/libexec
PACKAGE = mesos
pkglibexecdir = $(libexecdir)/$(PACKAGE)
datarootdir = ${prefix}/share
datadir = ${datarootdir}
pkgdatadir = $(datadir)/mesos
pkglibdir = $(libdir)/mesos
pkgmoduledir = $(pkglibdir)/modules
BOOST_VERSION = 1.53.0
BOOST = 3rdparty/boost-$(BOOST_VERSION)
am__append_4 = -isystem $(top_srcdir)/build/$(BOOST)
ELFIO_VERSION = 3.2
ELFIO = 3rdparty/elfio-$(ELFIO_VERSION)
am__append_5 = -I$(top_srcdir)/build/$(ELFIO)
GLOG_VERSION = 0.3.3
GLOG = 3rdparty/glog-$(GLOG_VERSION)
am__append_6 = -I$(top_srcdir)/build/$(GLOG)/src
LEVELDB_VERSION = 1.19
LEVELDB = 3rdparty/leveldb-$(LEVELDB_VERSION)
am__append_8 = -I$(top_srcdir)/build/$(LEVELDB)/include
LIBPROCESS = 3rdparty/libprocess
am__append_10 = -I$(top_srcdir)/build/$(LIBPROCESS)/include
NVML_VERSION = 352.79
NVML =  3rdparty/nvml-$(NVML_VERSION)
PICOJSON_VERSION = 1.3.0
PICOJSON = 3rdparty/picojson-$(PICOJSON_VERSION)
ZOOKEEPER_VERSION = 3.4.8
ZOOKEEPER = 3rdparty/zookeeper-$(ZOOKEEPER_VERSION)/src/c
am__append_11 = -I$(top_srcdir)/build/$(NVML)
am__append_12 = -I$(top_srcdir)/build/$(PICOJSON)
am__append_13 = -I$(top_srcdir)/build/$(PROTOBUF)/src
am__append_15 = -I$(top_srcdir)/$(STOUT)/include
am__append_16 = -I$(top_srcdir)/build/$(ZOOKEEPER)/include -I$(top_srcdir)/build/$(ZOOKEEPER)/generated

MESOS_CPPFLAGS = -Werror -DLIBDIR=\"$(libdir)\" \
	-I$(MESOS_HEADERS)  \
	-DPKGLIBEXECDIR=\"$(pkglibexecdir)\" \
	-DPKGDATADIR=\"$(pkgdatadir)\" \
	-DPKGMODULEDIR=\"$(pkgmoduledir)\" -I$(top_srcdir)/include \
	-I$(top_srcdir)/build/include -I$(top_srcdir)/build/include/mesos -DPICOJSON_USE_INT64 \
	-D__STDC_FORMAT_MACROS $(am__append_4) $(am__append_5) \
	$(am__append_6) $(am__append_8) $(am__append_10) \
	$(am__append_11) $(am__append_12) $(am__append_13) \
	$(am__append_15) $(am__append_16) -DHAS_AUTHENTICATION=1

test_executor_CPPFLAGS = $(MESOS_CPPFLAGS)

PTHREAD_CFLAGS = -pthread
# Enable common (and some language specific) warnings.
# # Warn when a comparison is made between signed and unsigned values.
# # Warn about use of format functions that can produce security issues.
AM_CXXFLAGS = $(PTHREAD_CFLAGS) -Wall -Wsign-compare -Wformat-security \
 	$(am__append_3)
DEFAULT_INCLUDES = -I$(top_srcdir)/build/src -I$(MESOS_TOP_DIR)/src
#DEFAULT_INCLUDES = -I$(srcdir)


#am__append_7 = -lglog
#am__append_9 = -lleveldb
#am__append_14 = -lprotobuf
#am__append_17 = -lzookeeper_mt
LDADD = $(am__append_7) $(am__append_9) $(am__append_14) \
	$(am__append_17)
top_builddir= $(MESOS_TOP_DIR)/build
test_executor_LDADD =  $(top_builddir)/src/.libs/libmesos.la $(LDADD)
LIBS = -lz -lsvn_delta-1 -lsvn_subr-1 -lsasl2 -lcurl -lapr-1 -lz  -lrt
SHELL = /bin/bash

LIBTOOL = $(SHELL) $(top_builddir)/libtool
AM_V_lt=
am__append_1 = -Wl,--as-needed
am__append_2 = -pie

AM_LDFLAGS = $(am__append_1) $(am__append_2)
LDFLAGS=
CXXLINK = $(LIBTOOL) $(AM_V_lt) --tag=CXX \
	--mode=link $(CXXLD) $(AM_CXXFLAGS) \
	$(CXXFLAGS) $(AM_LDFLAGS) $(LDFLAGS)
am__DEPENDENCIES_6 =
am__DEPENDENCIES_7 = $(am__DEPENDENCIES_6) $(am__DEPENDENCIES_6) \
	$(am__DEPENDENCIES_6) $(am__DEPENDENCIES_6)
test_executor_DEPENDENCIES =  $(top_builddir)/src/.libs/libmesos.so $(am__DEPENDENCIES_7)
AM_V_CXX = $(am__v_CXX_$(V))
AM_DEFAULT_VERBOSITY=1
am__v_CXX_ = $(am__v_CXX_$(AM_DEFAULT_VERBOSITY))
am__v_CXX_0 = @echo "  CXX     " $@;
am__v_CXX_1 =


all: executors frameworks

executors: mesos_vine_darkGray 
#executor-cudablackscholes executor-vineblackscholes executor-vinemontecarlo
frameworks: vaq-framework


mesos_vine_darkGray:
	@rm -f ./obj/mesos_vine_darkGray.o
	$(AM_V_CXX)$(CXX) $(DEFS)  $(DEFAULT_INCLUDES) $(MESOS_CPPFLAGS) $(CPPFLAGS) $(AM_CXXFLAGS)  $(AM_CXXFLAGS) $(CXXFLAGS)  mesos_vine_darkGray.cpp  -c -o obj/mesos_vine_darkGray.o   $(test_executor_DEPENDENCIES)
	$(CXXLINK) -o ./exec/mesos_vine_darkGray  ./obj/mesos_vine_darkGray.o  $(VAC_OBJS)  $(test_executor_LDADD) $(LIBS) $(LIB_CUDA)

executor-vineblackscholes:
	 @rm -f ./obj/executor_vineblackscholes.o
	$(AM_V_CXX)$(CXX) $(DEFS)  $(DEFAULT_INCLUDES) $(MESOS_CPPFLAGS) $(CPPFLAGS) $(AM_CXXFLAGS)  $(AM_CXXFLAGS) $(CXXFLAGS)  executor_vineblackscholes.cpp  -c -o obj/executor_vineblackscholes.o   $(test_executor_DEPENDENCIES)
	$(CXXLINK) -o ./exec/executor-vineblackscholes  ./obj/executor_vineblackscholes.o  $(VAC_OBJS)  $(test_executor_LDADD) $(LIBS) $(LIB_CUDA)

executor-vinemontecarlo:
	 @rm -f ./obj/executor_vineMonteCarlo.o
	$(AM_V_CXX)$(CXX) $(DEFS)  $(DEFAULT_INCLUDES) $(MESOS_CPPFLAGS) $(CPPFLAGS) $(AM_CXXFLAGS)  $(AM_CXXFLAGS) $(CXXFLAGS)  executor_vineMonteCarlo.cpp  -c -o obj/executor_vineMonteCarlo.o   $(test_executor_DEPENDENCIES)
	$(CXXLINK) -o ./exec/executor-vinemontecarlo  ./obj/executor_vineMonteCarlo.o 	$(CUDA_OBJS) $(VAC_OBJS)  $(test_executor_LDADD) $(LIBS) $(LIB_CUDA)


vaq-framework:
	@rm -f ./obj/vaq_framework.o
	$(AM_V_CXX)$(CXX) $(DEFS)  $(DEFAULT_INCLUDES) $(MESOS_CPPFLAGS) $(CPPFLAGS) $(AM_CXXFLAGS)  $(AM_CXXFLAGS) $(CXXFLAGS)  vaq_framework.cpp  -c -o obj/vaq_framework.o   $(test_executor_DEPENDENCIES)
	$(CXXLINK) -o ./exec/vaq-framework ./obj/vaq_framework.o $(test_executor_LDADD) $(LIBS)


executor-cudablackscholes:
	rm -f ./obj/executor_cudablackscholes.o
	$(AM_V_CXX)$(CXX) $(DEFS)  $(DEFAULT_INCLUDES) $(MESOS_CPPFLAGS) $(CPPFLAGS) $(AM_CXXFLAGS)  $(AM_CXXFLAGS) $(CXXFLAGS)  executor_cudablackscholes.cpp  -c -o obj/executor_cudablackscholes.o   $(test_executor_DEPENDENCIES)
	$(CXXLINK) -o ./exec/executor-cudablackscholes ./obj/executor_cudablackscholes.o $(MY_CUDA_DIR)/BlackScholesobj.o  $(test_executor_LDADD) $(LIBS) $(LIB_CUDA)


executor-cudamontecarlo:
	rm -f ./obj/executor_cudamontecarlo.o
	rm -f ./exec/executor-cudamontecarlo
	rm -f ./exec/.libs/executor-cudamontecarlo
	$(AM_V_CXX)$(CXX) $(DEFS)  $(DEFAULT_INCLUDES) $(MESOS_CPPFLAGS) $(CPPFLAGS) $(AM_CXXFLAGS)  $(AM_CXXFLAGS) $(CXXFLAGS)  executor_cudamontecarlo.cpp  -c -o obj/executor_cudamontecarlo.o   $(test_executor_DEPENDENCIES)
	#$(CXXLINK) -o ./exec/executor-cudamontecarlo ./obj/executor_cudamontecarlo.o  $(MY_CUDA_DIR)/MonteCarloMultiGPUobj.o  $(test_executor_LDADD) $(LIBS) $(LIB_CUDA)
	$(AM_V_CXX)$(CXX) $(DEFS)  $(DEFAULT_INCLUDES) $(MESOS_CPPFLAGS) $(CPPFLAGS) $(AM_CXXFLAGS)  $(AM_CXXFLAGS) $(CXXFLAGS)  -o obj/executor_cudamontecarlo.o -c executor_cudamontecarlo.cpp    $(test_executor_DEPENDENCIES)
	$(CXXLINK) -o ./exec/executor-cudamontecarlo ./obj/executor_cudamontecarlo.o  $(MY_CUDA_DIR)/MonteCarloMultiGPUobj.o  $(test_executor_LDADD) $(LIBS) $(LIB_CUDA)

clean:
	rm -f ./obj/*.o
	rm -f ./exec/.libs/*
	rm -f ./exec/*

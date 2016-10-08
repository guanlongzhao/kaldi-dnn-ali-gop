# makefiles/kaldi.mk.cygwin contains Cygwin-specific rules

ifndef FSTROOT
$(error FSTROOT not defined.)
endif

DOUBLE_PRECISION = 0
CXXFLAGS = -msse -msse2 -Wall -I.. -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) \
    -DHAVE_CLAPACK -I ../../tools/CLAPACK/ \
    -Wno-sign-compare -Wno-unused-local-typedefs -Winit-self \
    -I ../../tools/CLAPACK/ \
    -I $(FSTROOT)/include \
    $(EXTRA_CXXFLAGS) \
    -g # -O0 -DKALDI_PARANOID

ifeq ($(KALDI_FLAVOR), dynamic)
CXXFLAGS += -fPIC
endif

LDFLAGS = -g --enable-auto-import
LDLIBS = $(EXTRA_LDLIBS) $(FSTROOT)/lib/libfst.a -ldl -L/usr/lib/lapack \
         --enable-auto-import -lcyglapack-0 -lcygblas-0 -lm -lpthread
CXX = g++
CC = g++
RANLIB = ranlib
AR = ar


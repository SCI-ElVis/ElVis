COMPATH	= /usr/bin
CXX	= $(COMPATH)/g++
CC	= $(COMPATH)/gcc
MACHTYPE = $(shell uname -m)
ifeq ($(MACHTYPE),i386)
FC	= /sw/bin/g95
FFLAGS = -DNO_DLAMCH
CPPFLAGS := $(CPPFLAGS) -D__APPLE_INTEL__
else
FC 	= /sw/bin/g77
endif

MPILIB	= -L/usr/local/lib
MPIINC	= -I/usr/local/include

ifdef PARALLEL
# Using OpenMPI implementation for C/C++ (www.open-mpi.org)
CXX	= /usr/local/bin/mpic++
CC	= /usr/local/bin/mpicc
FC	= /usr/local/bin/mpif77
endif
OPTXX	= -O2 $(MCPU) -ffast-math 
MOPTXX	= -O3 $(MCPU) -ffast-math -funroll-loops -fstrict-aliasing 
OPT	= -O2 $(MCPU) -ffast-math 
MOPT	= -O3 $(MCPU) -ffast-math -funroll-loops -fstrict-aliasing 
OPTF	= -O2 $(MCPU) -ffast-math 
MOPTF	= -O3 $(MCPU) -ffast-math -funroll-loops  
DEBUG	= -g -gstabs+
DEBUGF	= -g $(FPPFLAGS)
LD	= $(CXX) -bind_at_load
DEBUGLD	= $(DEBUG)
MOPTLD	= $(MOPTXX)
OPTLD	= $(OPTXX)
ifdef FFTPRG
FFTLIB	= -lfftw
endif
ifdef UMFPACKSLV
UMFPACKLIB = -lumfpack -lamd
endif
LPG   	= -L/usr/local/lib -lpgftnrtl -lpgc   
ifeq ($(MACHTYPE),i386)
LIBS	= -lvec $(FFTLIB) $(UMFPACKLIB) -framework Accelerate -L/sw/lib -lg95
else
LIBS	= -lvec $(FFTLIB) $(UMFPACKLIB) -framework Accelerate -lSystem -L/sw/lib -lg2c 
endif

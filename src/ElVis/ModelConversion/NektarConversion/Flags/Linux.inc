COMPATH	= /usr/bin
CXX	= $(COMPATH)/g++
CC	= $(COMPATH)/gcc 
FC	= $(COMPATH)/gfortran
YACC= $(COMPATH)/bison -y

MPILIB	= -L/usr/local/bin/
MPIINC	= -I/usr/local/include/

ifdef PARALLEL
CXX	= /usr/local/bin/mpiCC
CC	= /usr/local/bin/mpicc
FC	= /usr/local/bin/mpif77
endif
OPTXX	= -O2 $(MCPU) -ffast-math 
MOPTXX	= -O3 $(MCPU) -ffast-math -funroll-loops -fstrict-aliasing 
OPT	= -O2 $(MCPU) -ffast-math 
MOPT	= -O3 $(MCPU) -ffast-math -funroll-loops -fstrict-aliasing 
OPTF	= -O2 $(MCPU) -ffast-math 
MOPTF	= -O3 $(MCPU) -ffast-math -funroll-loops  
DEBUG	= -g 
DEBUGF	= -g 
LD	= $(CXX)
DEBUGLD	= $(DEBUG)
MOPTLD	= $(MOPTXX)
OPTLD	= $(OPTXX)
ifdef FFTPRG
FFTLIB	= -lfftw
endif
LLAPACK = -L/usr/lib64 -llapack 
LBLAS   = -L/usr/lib64 -lblas
ifdef UMFPACKSLV
UMFPACKLIB = -lumfpack -lamd
endif
LPG   	= -L/usr/local/lib -lpgftnrtl -lpgc   
LIBS	= -lvec $(FFTLIB) $(UMFPACKLIB) $(LLAPACK) $(LBLAS) -lg2c

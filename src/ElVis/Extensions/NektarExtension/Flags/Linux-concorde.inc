# In order to use this file, the modules intel_cc/8.0 and intel_fc/8.0 must be loaded.
# Use option -V in qsub to export environment variables, so the executable can find
# the shared libraries when running on the nodes
CXX	= icpc
CC	= icc 
FC	= ifort

# Below is for concorde
MPIBIN	= /usr/local/src/mpich-1.2.5.2-1/bin/
MPILIB	= -L/usr/local/src/mpich-1.2.5.2-1/lib/
MPIINC	= -I/usr/local/src/mpich-1.2.5.2-1/include/

ifdef PARALLEL
CXX	= $(MPIBIN)/mpiCC
CC	= $(MPIBIN)/mpicc
FC	= $(MPIBIN)/mpif77
endif
OPTXX	= -O2 $(MCPU)  
MOPTXX	= -O3 $(MCPU) -funroll-loops -fstrict-aliasing 
OPT	= -O2 $(MCPU)  
MOPT	= -O3 $(MCPU) -funroll-loops -fstrict-aliasing 
OPTF	= -O2 $(MCPU)  
MOPTF	= -O3 $(MCPU) -funroll-loops  
DEBUG	= -g 
DEBUGF	= -g 
LD	= $(CXX)
DEBUGLD	= $(DEBUG)
MOPTLD	= $(MOPTXX)
OPTLD	= $(OPTXX)
ifdef FFTPRG
FFTLIB	= -lfftw
endif
LLAPACK = -L/usr/local/lib -llapack 
LBLAS   = -L/usr/local/lib -lblas
LPG   	= -L/usr/local/lib -lpgftnrtl -lpgc 
ifdef UMFPACKSLV
UMFPACKLIB = -lumfpack -lamd
endif
LG2C    = -L/slave-images/default-image/lib/ssa/gcc-lib/i386-redhat-linux-gnu/3.5-tree-ssa -lg2c  
LIFCORE = -L/usr/local/Cluster-Apps/intel_fc_80/lib -lifcore 
LIBS	= -lvec $(FFTLIB) $(UMFPACKLIB) $(LLAPACK) $(LBLAS) $(LG2C) $(LIFCORE)

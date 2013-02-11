CXX	= icpc
CC	= icc 
FC	= ifort

ifdef PARALLEL
CXX	= mpiicpc
CC	= mpiicc
FC	= mpiifort
endif
OPTXX	= -O2 $(MCPU)  
MOPTXX	= -O3 $(MCPU)  -funroll-loops -fstrict-aliasing 
OPT	= -O2 $(MCPU)  
MOPT	= -O3 $(MCPU)  -funroll-loops -fstrict-aliasing 
OPTF	= -O2 $(MCPU)  
MOPTF	= -O3 $(MCPU)  -funroll-loops  
DEBUG	= -g 
DEBUGF	= -g 
LD	= $(CXX)
DEBUGLD	= $(DEBUG)
MOPTLD	= $(MOPTXX)
OPTLD	= $(OPTXX)

ifdef FFTPRG
FFTLIB	= -lfftw
endif

LLAPACK = -L $(MKL_HOME)/lib/64 -lmkl_lapack 
LSCALAPACK = -L $(MKL_HOME)/lib/64 -lmkl_scalapack
LBLACS = -L $(MKL_HOME)/lib/64 -lmkl_blacs_intelmpi20_lp64
LBLAS   = -lmkl  -lsvml
LIBS	= -L../../Metis/metis-4.0 -L ../../gs -L../../Veclib -lvec $(FFTLIB) $(LSCALAPACK) $(LBLACS) $(LLAPACK) $(LBLAS) -lguide -liomp5 -lpthread -lsvml -L $(IFC_HOME)/lib -lifcore 

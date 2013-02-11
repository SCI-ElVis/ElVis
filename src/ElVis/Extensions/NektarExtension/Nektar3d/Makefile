ARCH = $(shell uname)
MACH    := $(shell uname -m)
MACHNAME = $(shell uname -n)
PROG    = nektar
VPATH   = ../src
HLIBDIR  = ../../Hlib/$(ARCH)
HDRDIR  = ../../include
TARFILE = $(PROG).tar

ifeq ($(ARCH),Linux)
  ifeq ($(MACH),ia64)
        ARCH =  Linux-ia64
  endif
endif

defines = METIS 
CPPFLAGS = -I$(HDRDIR)
ifdef PARALLEL
MPIPRG  = yes
endif

ALL     = 	analyser    bwoptim     cfl  convective     comm   	\
		dgalerkin   drive       forces              io          \
		mlevel 	    prepost     pressure            rotational	\
		sections    stokes      subcycle     	    womersley   \
		velinterp   

include ../../Flags/$(ARCH).inc

ifeq ($(ARCH),cray-pvp)			# Cray PVP
include ../../Flags/CRAYPVP.inc
endif

ifeq ($(ARCH),cray-t3d)			# Cray T3D
include ../../Flags/T3D.inc
endif

ifeq ($(ARCH),unicosmk)			# Cray T3E
include ../../Flags/T3E.inc
endif

ifdef PARALLEL
ifeq ($(ARCH), AIX)
CXX = $(MPICXX)
endif
override CPPFLAGS :=  -DPARALLEL  $(CPPFLAGS) $(MPIINC)
override LIBS     := $(LIBS) -lmetis -lgs #$(MPILIB) -lmpich
ifeq ($(ARCH),ap3000)
override LIBS     := $(LIBS) -L/opt/FSUNaprun/lib -lmpl -lemi -lthread 
endif
ifeq ($(ARCH),OSF1)
override LIBS     := $(LIBS) -lelan -lelan3
endif
endif

ifdef WOMERR
override ADDONS := WOMERR
PROG = nektar.wom
endif

ifdef VELINTERP
override ADDONS := VELINTERP
PROG = nektar.vel
endif

ifdef    defines
DEF     = $(foreach def, $(defines) $(ADDONS), -D$(def))
override CPPFLAGS :=  $(DEF) $(CPPFLAGS)
endif

export


help:
	@echo ''
	@echo ' Nektar3d Makefiles:'
	@echo ''
	@echo ' Run make with one of the following options:'	
	@echo '	a) dbx.... dbx version'
	@echo '	b) opt.... optimised version'
	@echo '	c) mopt... maximally (?) optimised version'
	@echo ''
	@echo '	need PARALLEL=1 in command line for parallel version'


dbx: 
	$(MAKE) LTYPE=g -f MakeNek
opt:
	$(MAKE) LTYPE=opt -f MakeNek
mopt:
	$(MAKE) LTYPE=mopt -f MakeNek

tar: Makefile
	@echo 'creating a tape archive in' $(TARFILE)
	@tar cvf $(TARFILE) Makefile MakeNek src/*.C
clean:
	$(MAKE) -f MakeNek clean



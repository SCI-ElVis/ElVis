ARCH	 = $(shell uname)
MACHTYPE = $(shell uname -m)
MACHNAME = $(shell uname -n)
ifeq ($(ARCH),OSF1)
override ARCH := $(MACHTYPE)
else
ifeq ($(MACHTYPE),alpha)
override ARCH := $(ARCH)-$(MACHTYPE)
endif
endif
HYBRID	 = ../
VPATH	 = ../src
LIB	 = libhybrid
INCDIR	 = ../../include
TARFILE	 = $(LIB).tar

# Compilers and options

defines  =  HOSURF ZERONULLDIV SVVSYMM SPARSEMAT #LE_VERT_OLD #VELINTERP #Lanczos #DUMPLEMODES  

CPPFLAGS :=  -I$(INCDIR)

include ../../Flags/${ARCH}.inc

ifeq ($(ARCH),cray-t3d)			# Cray T3D
include ../../Flags/T3D.inc
endif

ifeq ($(ARCH),unicosmk)			# Cray T3E
include ../../Flags/T3E.inc
endif

ifeq ($(ARCH),alpha)			# DEC Alpha AXP
include ../../Flags/DECAXP.inc
endif

ifeq ($(ARCH),Linux-alpha)	        # Linux on alpha processor
include ../../Flags/GCC.inc
endif

ifeq ($(ARCH),SUPER-UX)			# NEC machine
include ../../Flags/Nec.inc
endif

ifeq ($(PARALLEL),1)
override defines := $(defines) PARALLEL
endif

ifeq ($(LEGENDRE),1)
override defines := $(defines) LEGENDRE 
endif

ifeq ($(LANC),1)
override defines := $(defines) Lanczos
endif

ifeq ($(UMFPACKSLV),1)
override defines := $(defines) UMFPACKSLV
endif

ifdef    defines
DEF     = $(foreach def, $(defines) , -D$(def))
override CPPFLAGS := $(CPPFLAGS) $(DEF)
endif


# These are the module names:

ALL   = Basis		Boundary 	Coords 		\
	Curvi		Dbutils		Deriv 		\
	EigenMatrix 	Element  	Element_List	\
	Felisa		FelisaF 	Fieldfiles 	\
	Geofac		Gradient 	HOSurf          Hex  		\
	InnerProd       Integrate   	Interp		\
	Interp_point	Matrix 		manager		\
	MatrixR		Matrix_Stokes	Memory 		\
	Misc  		Mrhs  		Nekvec 		Norms		\
	Nums 		Overlap 	OverlapFull 	\
	Parallel	Particle	polylib		Prism		\
	Pyr		Quad	       	Refine		\
	SMatrix 	Solve		SolveR		Solve_Stokes	\
	Solve_cg	Tet		Transforms	\
	tree		Tri		Utilities	

export

help:
	@echo ''
	@echo ' Nektar Makefiles:'
	@echo ''
	@echo ' Run make with one of the following options:'	
	@echo '	a) dbx....dbx version'
	@echo '	b) opt....optimised version'
	@echo '	c) mopt...maximally (?) optimized version'
	@echo ''
	@echo '$(message)'

dbx: 
	$(MAKE) LTYPE=g -f MakeHybrid
	@echo '$(message)'

opt:
	$(MAKE) LTYPE=opt -f MakeHybrid
	@echo '$(message)'

mopt:
	$(MAKE) LTYPE=mopt -f MakeHybrid
	@echo '$(message)'

tar: Makefile
	@echo 'creating a tape archive in' $(TARFILE)
	@tar cvf $(TARFILE) Makefile MakeHybrid src/*.[cCfFy]

clean:
	-rm -f *.o *~

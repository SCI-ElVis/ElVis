ARCH	 = $(shell uname)
MACHNAME	 = $(shell uname -n)
VPATH	 = ./
LIB	 = libblkmat.a
LIBDIR	 = ../${ARCH}
INCDIR	 = ../../include
TARFILE	 = blkmat.tar

# Compilers and options

CPPFLAGS = -I./ -I$(INCDIR) -I./include  -g -gstabs+

include ../../Flags/${ARCH}.inc

ifeq ($(ARCH),cray-pvp)			# Cray PVP
include ../../Flags/CRAYPVP.inc
endif

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


# These are the module names:

ALL   = BlkMat

# ---------------------------------------------------------------- #
# The following expand to give the names of all sections (ALL),    #
# all library member dependencies (LMOD), all associated           #
# object files (OBJS), and all source code files (SRC).            #
# ---------------------------------------------------------------- #

LMOD  = $(foreach routine, $(ALL), $(LIB)($(routine).o))
OBJS  = $(foreach routine, $(ALL), $(routine).o)

# ----------------------------------------------------------------------------
# Make library, default action.
#
$(LIB): $(LMOD)
	ranlib $(LIB)

# ----------------------------------------------------------------------------
# Install library & associated header files.
#
install:
	cp -f BlkMat.h     $(INCDIR)	;			
	cp -f BlkVec.h     $(INCDIR)	;			
	cp -f BlkSubMat.h  $(INCDIR)	;			
	cp -f libblkmat.a  $(LIBDIR)	;			

help:
	@echo ''
	@echo ' Run make :'	
	@echo ''

tar: Makefile
	@echo 'creating a tape archive in' $(TARFILE)
	@tar cvf $(TARFILE) Makefile *.cpp *.h 
	gzip -9 -f  $(TARFILE)
clean:
	-rm -f *.o *~



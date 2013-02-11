

SET(Headers
    
) 

SET(Sources
  math/xfill.c
  math/xneg.c
  math/xvneg.c
  math/xvadd.c
  math/xvsub.c
  math/xsmul.c
  math/xvmul.c
  math/xvrecp.c
  math/xvdiv.c
  math/xzero.c
  math/xsadd.c
  math/xsdiv.c
  math/xvabs.c
  math/xvamax.c
  math/xvsin.c
  math/xvcos.c
  math/xvsqrt.c
  math/xvlg10.c
  math/xvexp.c
  triads/xsvtvm.c
  triads/xsvtvp.c
  triads/xsvvmt.c
  triads/xsvvpt.c
  triads/xsvvtm.c
  triads/xsvvtp.c
  triads/xvvtvm.c
  triads/xvvtvp.c
  triads/xvvpvt.c
  triads/xvvmvt.c
  triads/xsvtsp.c
  triads/xvvvtm.c
  relational/xseq.c
  relational/xsle.c
  relational/xslt.c
  relational/xsne.c
  reduction/xsum.c
  reduction/ixmax.c
  reduction/ixmin.c
  reduction/icount.c
  reduction/ifirst.c
  misc/xvpoly.c
  conversion/xvfloa.c
  conversion/vdble.c
  conversion/vsngl.c
  math/xvrand.c
  conversion/xvreal.c
  conversion/xvimag.c
  conversion/xvcmplx.c
  conversion/xbrev.c
  misc/xscatr.c
  misc/xgathr.c
  misc/xramp.c
  misc/xcndst.c
  misc/xmask.c
  misc/polint.c
  math/xfft.c
  misc/mxm.c
  misc/mxv.c
  misc/mxva.c
  misc/spline.c
  misc/veclib.c
  misc/dclock.c
  misc/mach.c
  memory/xvector.c
  memory/xmatrix.c
  memory/icopy.c
)

SET(FortranSources
  misc/zbesj.f
  misc/fftdf.f

)

SET(ProjectName "veclib")
SET(FortranProjectName "veclibf")

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/ElVis/Extensions/NektarExtension/include)
ADD_ELVIS_SHARED_LIBRARY(${ProjectName} ${Sources} ${FortranSources} ${Headers})
#ADD_ELVIS_STATIC_LIBRARY(${FortranProjectName} ${FortranSources})
#set_target_properties(${FortranProjectName} PROPERTIES LINKER_LANGUAGE fortran)

TARGET_LINK_LIBRARIES(${ProjectName} 
)

SET_TARGET_PROPERTIES(${ProjectName} PROPERTIES VERSION ${ELVIS_VERSION})


 

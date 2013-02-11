


SET(Headers
    
) 

SET(Sources
  src/analyser.C
  src/bwoptim.C
  src/cfl.C
  src/convective.C
  src/comm.C
  src/dgalerkin.C
  src/drive.C
  src/forces.C
  src/io.C
  src/mlevel.C
  src/prepost.C
  src/pressure.C
  src/rotational.C
  src/sections.C
  src/stokes.C
  src/subcycle.C
  src/womersley.C
  src/velinterp.C
)

SET(FortranSources
  
)

SET(ProjectName "nektar3d")

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Modules/NektarConversion/include)
ADD_ELVIS_SHARED_LIBRARY(${ProjectName} ${Sources} ${FortranSources} ${Headers})
	
TARGET_LINK_LIBRARIES(${ProjectName} 
)

SET_TARGET_PROPERTIES(${ProjectName} PROPERTIES VERSION ${ELVIS_VERSION})


 

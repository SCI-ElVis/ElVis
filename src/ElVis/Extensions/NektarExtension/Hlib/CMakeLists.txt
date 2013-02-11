


SET(Headers
    
) 

SET(Sources
  src/Basis.C
  src/Boundary.C
  src/Coords.C
  src/Curvi.C
  src/Dbutils.C
  src/Deriv.C
  src/EigenMatrix.C
  src/Element.C
  src/Element_List.C
  src/Felisa.C
  src/Fieldfiles.C
  src/Geofac.C
  src/Gradient.C
  src/HOSurf.C
  src/Hex.C
  src/InnerProd.C
  src/Integrate.C
  src/Interp.C
  src/Interp_point.C
  src/manager.c
  src/Matrix.C
  src/MatrixR.C
  src/Matrix_Stokes.C
  src/Memory.C
  src/Misc.C
  src/Mrhs.C
  src/Nekvec.C
  src/Norms.C
  src/Nums.C
  src/Overlap.C
  src/OverlapFull.C
  src/Parallel.C
  src/Particle.C
  src/polylib.C
  src/Prism.C
  src/Pyr.C
  src/Quad.C
  src/Refine.C
  src/SMatrix.C
  src/Solve.C
  src/SolveR.C
  src/Solve_Stokes.C
  src/Solve_cg.C
  src/Tet.C
  src/Transforms.C
  src/tree.c
  src/Tri.C
  src/Utilities.C
)

SET(FortranSources
  src/FelisaF.F
)

SET(ProjectName "hybrid")

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/ElVis/Extensions/NektarExtension/include)

ADD_ELVIS_SHARED_LIBRARY(${ProjectName} ${Sources} ${FortranSources} ${Headers})
	
TARGET_LINK_LIBRARIES(${ProjectName} 
    veclib
    ${BLAS_blas_LIBRARY}
    ${LAPACK_Accelerate_LIBRARY}
)

SET_TARGET_PROPERTIES(${ProjectName} PROPERTIES VERSION ${ELVIS_VERSION})


 

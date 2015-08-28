
IF( NOT QT4_FOUND )
    FIND_PACKAGE(Qt4)
ENDIF() 

IF( QT4_FOUND )
    include(${QT_USE_FILE})
ENDIF()

 
SET(ProjectName ElVisCLI)

# Default location
IF( MSVC )
  SET(GLUT_ROOT_PATH ${CMAKE_SOURCE_DIR}/Externals/glut)
  SET(GLUT_INCLUDE_DIR ${GLUT_ROOT_PATH}/include)
  IF( CMAKE_CL_64 )
    SET(GLUT_glut_LIBRARY ${GLUT_ROOT_PATH}/release/glut64.lib)
  ELSE()
    SET(GLUT_glut_LIBRARY ${GLUT_ROOT_PATH}/release/glut32.lib)
  ENDIF()
ELSE()
  FIND_PACKAGE(GLUT REQUIRED)
ENDIF()


INCLUDE_DIRECTORIES(
    ${OPENGL_INCLUDE_DIR} 
    ${QT_QTCORE_INCLUDE_DIR}
    ${QT_QTOPENGL_INCLUDE_DIR}
    ${QT_QTGUI_INCLUDE_DIR}
    ${GLUT_INCLUDE_DIR}
    ${NEKTAR++_INCLUDE_DIRS}
    ${PNG_INCLUDE}
    ${ZLIB_INCLUDE}
)

add_definitions(-DQT_NO_KEYWORDS)

SET(HEADERS
  NewApproachCanister.h
  ViewSettingsRendering.h
  VolumeRendering.h
) 

SET(SOURCES
  main.cpp
  NewApproachCanister.cpp
  ViewSettingsRendering.cpp
  VolumeRendering.cpp
)


ADD_ELVIS_EXECUTABLE(${ProjectName} ${HEADERS} ${SOURCES} )
SET_PROPERTY(TARGET ${ProjectName} PROPERTY FOLDER "Core")

TARGET_LINK_LIBRARIES(${ProjectName} 
#    ${OPENGL_LIBRARIES} #CMake says this is not set on Linux, maybe on Windows or Apple?
    ElVisCore
    ${QT_LIBRARIES}
    ${QT_QTOPENGL_LIBRARY}
    ${GLUT_glut_LIBRARY}
    ${PNG_LIBRARY}
)


IF( MSVC )
  INSTALL(TARGETS ${ProjectName} EXPORT ElVisLibraries RUNTIME DESTINATION ${ELVIS_BIN_DIR}
                                LIBRARY DESTINATION ${ELVIS_BIN_DIR}
                                ARCHIVE DESTINATION ${ELVIS_LIB_DIR} )
ENDIF()
SET(BaselineDir ${CMAKE_SOURCE_DIR}/../Testing/${ELVIS_FLOAT}/${CMAKE_HOST_SYSTEM_NAME})

#All regression test dependencies
SET( REG_DEPEND ${ProjectName} )

#==============================================================================
# This is the 'regcheck' targets that run all the regression tests
ADD_CUSTOM_TARGET( regcheck
                   COMMAND ${CMAKE_MAKE_PROGRAM} install
                   COMMAND ${CMAKE_CTEST_COMMAND} --timeout 3600 --output-on-failure -L ElVisRegCheck $(CTESTARGS)
                   DEPENDS ${REG_DEPEND}
                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )

ADD_CUSTOM_TARGET( regcheck_memcheck
                   COMMAND ${CMAKE_MAKE_PROGRAM} install
                   COMMAND ${CMAKE_CTEST_COMMAND} --timeout 86400 --output-on-failure -L ElVisRegMemCheck $(CTESTARGS)
                   DEPENDS ${REG_DEPEND}
                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )

ADD_CUSTOM_TARGET( regcheck_stackcheck
                   COMMAND ${CMAKE_MAKE_PROGRAM} install
                   COMMAND ${CMAKE_CTEST_COMMAND} --timeout 86400 --output-on-failure -L ElVisRegStackCheck $(CTESTARGS)
                   DEPENDS ${REG_DEPEND}
                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )

ADD_COVERAGE( regcheck_coverage regcheck )


#==============================================================================
# Creates a regression test
MACRO( ADD_ELVIS_REGTEST REGNAME )

  # Add the test to ctest
  ADD_TEST( NAME ${REGNAME}
            COMMAND ${ProjectName} ${ARGN}
            WORKING_DIRECTORY ${ELVIS_BIN_DIR} )
  SET_TESTS_PROPERTIES( ${REGNAME} PROPERTIES LABELS ElVisRegCheck )

  # Create a target so the test can be run indivindually
  ADD_CUSTOM_TARGET( ${REGNAME}
                     COMMAND ${ProjectName} ${ARGN}
                     DEPENDS ${REG_DEPEND} 
                     WORKING_DIRECTORY ${ELVIS_BIN_DIR} )

  # The memory checking command
  SET( REG_MEMCHECK   ${VALGRIND_COMMAND}   $<TARGET_FILE:${ProjectName}> ${ARGN} )
  SET( REG_STACKCHECK ${STACKCHECK_COMMAND} $<TARGET_FILE:${ProjectName}> ${ARGN} )
  SET( REG_CUDAMEMCHECK cuda-memcheck --leak-check full   $<TARGET_FILE:${ProjectName}> ${ARGN} )
  
  # Add the ctest memcheck
  ADD_TEST( NAME ${REGNAME}_memcheck
            COMMAND ${REG_MEMCHECK}
            WORKING_DIRECTORY ${ELVIS_BIN_DIR} )
  SET_TESTS_PROPERTIES( ${REGNAME}_memcheck PROPERTIES LABELS ElVisRegMemCheck )

  # Add the indivindual memcheck target
  ADD_CUSTOM_TARGET( ${REGNAME}_memcheck
                     COMMAND ${REG_MEMCHECK}
                     DEPENDS ${REG_DEPEND} 
                     WORKING_DIRECTORY ${ELVIS_BIN_DIR} )

  # Add the ctest stackcheck
  ADD_TEST( NAME ${REGNAME}_stackcheck
            COMMAND ${REG_STACKCHECK}
            WORKING_DIRECTORY ${ELVIS_BIN_DIR} )
  SET_TESTS_PROPERTIES( ${REGNAME}_stackcheck PROPERTIES LABELS ElVisRegStackCheck )

  # Add the indivindual stackcheck target
  ADD_CUSTOM_TARGET( ${REGNAME}_stackcheck
                     COMMAND ${REG_STACKCHECK}
                     DEPENDS ${REG_DEPEND} 
                     WORKING_DIRECTORY ${ELVIS_BIN_DIR} )

  # Add the ctest memcheck
  ADD_TEST( NAME ${REGNAME}_cudamemcheck
            COMMAND ${REG_CUDAMEMCHECK}
            WORKING_DIRECTORY ${ELVIS_BIN_DIR} )
  SET_TESTS_PROPERTIES( ${REGNAME}_cudamemcheck PROPERTIES LABELS ElVisRegMemCheck )

  # Add the indivindual memcheck target
  ADD_CUSTOM_TARGET( ${REGNAME}_cudamemcheck
                     COMMAND ${REG_CUDAMEMCHECK}
                     DEPENDS ${REG_DEPEND} 
                     WORKING_DIRECTORY ${ELVIS_BIN_DIR} )

ENDMACRO()



IF( ELVIS_ENABLE_JACOBI_EXTENSION )
    #GET_PROPERTY(JacobiExtensionPath TARGET JacobiExtension PROPERTY RELEASE_LOCATION)
    SET( JacobiExtensionPath $<TARGET_FILE:JacobiExtension> )
    
    ADD_ELVIS_REGTEST( BulletCutSurfaceP2 
        --TestName CutSurfaceBullet
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/bulletH1P2.dat"
        --Module ${JacobiExtensionPath}
        --Width 1000
        --Height 1000
        --OutFile BulletCutSurfaceP2
        --ColorMap "${CMAKE_SOURCE_DIR}/../ColorMaps/diverging.xml"
        --ColorMapMin -0.12
        --ColorMapMax 0.0
        --CompareFile "${BaselineDir}/BulletCutSurfaceP2Baseline"
    ) 

    ADD_ELVIS_REGTEST( BulletCutSurfaceP2WithFaces 
        --TestName Generic
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/bulletH1P2.dat"
        --Module ${JacobiExtensionPath}
        --Width 1000
        --Height 600
        --OutFile BulletCutSurfaceP2WithFaces
        --CompareFile "${BaselineDir}/BulletCutSurfaceP2WithFacesBaseline"
        --ColorMap "${CMAKE_SOURCE_DIR}/../ColorMaps/Rainbow.xml"
        --ColorMapMin -0.1
        --ColorMapMax 0.0
        --RenderFaces 14000
        --Eye 1.8 0.46 3.7 --Up 0 1 0 --At 0 0 2.7
        --CutPlaneNormal 1.0 0.0 0.0
        --CutPlanePoint 0.0 -1.0 0.0
    )

    ADD_ELVIS_REGTEST( BulletCutSurfaceP3 
        --TestName CutSurfaceBullet
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/bulletH1P3.dat"
        --Module ${JacobiExtensionPath}
        --Width 1000
        --Height 1000
        --OutFile BulletCutSurfaceP3
        --ColorMap "${CMAKE_SOURCE_DIR}/../ColorMaps/diverging.xml"
        --ColorMapMin -0.12
        --ColorMapMax 0.0
        --CompareFile "${BaselineDir}/BulletCutSurfaceP3Baseline"
    )

    ADD_TEST( VolumeRenderBulletFull 
        --TestName VolumeRenderBullet
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/bulletH1P2.dat"
        --Module ${JacobiExtensionPath}
        --RenderIntegrationType 0
        --Width 1152  --Height 672 --IntegrationType 7
        --Breakpoints -.04 -.045 -.03 .01 .015 .02 -.001 0.0 .001  --Colors 0.0 0.0 0.0 0.0   1.0 0.0 0.0 100    0.0 0.0 0.0 0.0   0.0 0.0 0.0 0.0   0.0 1.0 0.0 20.0    0.0 0.0 0.0 0.0       0.0 0.0 0.0 0.0  0.0 0.0 1.0 5.0   0.0 0.0 0.0 0.0
        --Eye 3 .1 1.5 --Up 0 1 0 --At 0 0 1.5
        --OutFile VolumeRenderBulletFull
        --CompareFile "${BaselineDir}/VolumeRenderBulletFullBaseline"
    )

    ADD_ELVIS_REGTEST(VolumeRenderBulletRiemann 
        --TestName VolumeRenderBullet
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/bulletH1P2.dat"
        --Module ${JacobiExtensionPath}
        --RenderIntegrationType 0
        --Width 1152  --Height 672 --IntegrationType 0
        --Breakpoints -.04 -.045 -.03 .01 .015 .02 -.001 0.0 .001  --Colors 0.0 0.0 0.0 0.0   1.0 0.0 0.0 100    0.0 0.0 0.0 0.0   0.0 0.0 0.0 0.0   0.0 1.0 0.0 20.0    0.0 0.0 0.0 0.0       0.0 0.0 0.0 0.0  0.0 0.0 1.0 5.0   0.0 0.0 0.0 0.0
        --Eye 3 .1 1.5 --Up 0 1 0 --At 0 0 1.5
        --OutFile VolumeRenderBulletRiemann
        --CompareFile "${BaselineDir}/VolumeRenderBulletRiemannBaseline"
    )

    ADD_ELVIS_REGTEST(VolumeRenderBulletTrapezoidal 
        --TestName VolumeRenderBullet
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/bulletH1P2.dat"
        --Module ${JacobiExtensionPath}
        --RenderIntegrationType 0
        --Width 1152  --Height 672 --IntegrationType 0
        --Breakpoints -.04 -.045 -.03 .01 .015 .02 -.001 0.0 .001  --Colors 0.0 0.0 0.0 0.0   1.0 0.0 0.0 100    0.0 0.0 0.0 0.0   0.0 0.0 0.0 0.0   0.0 1.0 0.0 20.0    0.0 0.0 0.0 0.0       0.0 0.0 0.0 0.0  0.0 0.0 1.0 5.0   0.0 0.0 0.0 0.0
        --Eye 3 .1 1.5 --Up 0 1 0 --At 0 0 1.5
        --OutFile VolumeRenderBulletTrapezoidal
        --CompareFile "${BaselineDir}/VolumeRenderBulletTrapezoidalBaseline"
    )

    ADD_ELVIS_REGTEST(IsosurfaceBullet1
        --TestName IsosurfaceBullet
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/bulletH1P2.dat"
        --Module ${JacobiExtensionPath}
        --Width 1152  --Height 672
        --Eye 3 .1 1.5 --Up 0 1 0 --At 0 0 1.5
        --OutFile IsosurfaceBullet1
        --Isovalues 0.0
        --ColorMap "${CMAKE_SOURCE_DIR}/../ColorMaps/diverging.xml"
        --ColorMapMin -0.12
        --ColorMapMax 0.0
        --CompareFile "${BaselineDir}/IsosurfaceBullet1Baseline"
    )

    ADD_ELVIS_REGTEST(IsosurfacePaperFigure5B
        --TestName ViewSettings
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/paddles1H1P3.dat"
        --Width 1024 --Height 1024
        --OutFile IsosurfacePaperFigure5B
        --Module ${JacobiExtensionPath}
        --Settings "${CMAKE_SOURCE_DIR}/../ViewingParameters/IsosurfacePaper_5b.xml"
        --CompareFile "${BaselineDir}/IsosurfacePaperFigure5BBaseline"
    )

    ADD_ELVIS_REGTEST(IsosurfacePaperFigure6B
        --TestName ViewSettings
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/bulletH1P3.dat"
        --Width 1024 --Height 1024
        --OutFile IsosurfacePaperFigure6B
        --Module ${JacobiExtensionPath}
        --Settings "${CMAKE_SOURCE_DIR}/../ViewingParameters/IsosurfacePaper_6b.xml"
        --CompareFile "${BaselineDir}/IsosurfacePaperFigure6BBaseline"
    )

    ADD_ELVIS_REGTEST(IsosurfacePaperFigure9A
        --TestName ViewSettings
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/bulletH1P3.dat"
        --Width 1024 --Height 1024
        --OutFile IsosurfacePaperFigure9A
        --Module ${JacobiExtensionPath}
        --Settings "${CMAKE_SOURCE_DIR}/../ViewingParameters/IsosurfacePaper_9a.xml"
        --CompareFile "${BaselineDir}/IsosurfacePaperFigure9ABaseline"
    )

    ADD_ELVIS_REGTEST(IsosurfacePaperFigure9B
        --TestName ViewSettings
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/bulletH1P3.dat"
        --Width 1024 --Height 1024
        --OutFile IsosurfacePaperFigure9B
        --Module ${JacobiExtensionPath}
        --Settings "${CMAKE_SOURCE_DIR}/../ViewingParameters/IsosurfacePaper_9b.xml"
        --CompareFile "${BaselineDir}/IsosurfacePaperFigure9BBaseline"
    )
ENDIF()

IF( ELVIS_ENABLE_ProjectX_EXTENSION )
    #GET_PROPERTY(PXExtensionPath TARGET ProjectXExtension PROPERTY RELEASE_LOCATION)
    SET( PXExtensionPath $<TARGET_FILE:ProjectXExtension> )
    
#    ADD_ELVIS_REGTEST(OneraWingSurface 
#        --TestName Generic
#        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/Onera_wing_M0p84_AA3p06_Euler_P2_100000DOF_P2_a06_Primal_3.pxa"
#        --Module ${PXExtensionPath}
#        --Width 785
#        --Height 535
#        --Eye 0.2 0.7 1.9 --Up 0 1 0 --At 0.8 0.0 0.4
#        --BoundarySurfaces 1 2 3 4 5
#        --ColorMap "${CMAKE_SOURCE_DIR}/../ColorMaps/Rainbow.xml"
#        --ColorMapMin 0.0
#        --ColorMapMax 1.0
#        --OutFile OneraWingSurface
#        --CompareFile "${BaselineDir}/OneraWingSurface"
#    )

    FOREACH( P 0 1 2 )
        ADD_ELVIS_REGTEST(naca_Euler2d_Q2_P${P}
            --TestName Generic
            --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/naca_Euler2d_TEST_P${P}_Primal_0.pxa"
            --Module ${PXExtensionPath}
            --Width 785
            --Height 535
            --Eye 0.5 0.0 2.0 --At 0.5 0.0 0.0 --Up 0.0 1.0 0.0
            --Field 0
            --ColorMap "${CMAKE_SOURCE_DIR}/../ColorMaps/Rainbow.xml"
            --ColorMapMin 0.98
            --ColorMapMax 1.04
            --CutPlaneNormal 0.0 0.0 1.0
            --CutPlanePoint 0.0 0.0 0.0
            #--EnableTrace 1 --TraceX 100 --TraceY 100 #Used to debug with ELVIS_PRINTF
            --OutFile naca_Euler2d_Q2_P${P}
            --CompareFile "${BaselineDir}/naca_Euler2d_Q2_P${P}_Baseline"
        )
    ENDFOREACH()

    ADD_ELVIS_REGTEST(naca_Euler2d_CurvedAdapt_Q3_P2
        --TestName Generic
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/naca_Euler2d_CurvedAdapt_TEST_P2_a02_Primal_1.pxa"
        --Module ${PXExtensionPath}
        --Width 785
        --Height 535
        --Eye 0.5 0.0 2.0 --At 0.5 0.0 0.0 --Up 0.0 1.0 0.0
        --Field 0
        --ColorMap "${CMAKE_SOURCE_DIR}/../ColorMaps/Rainbow.xml"
        --ColorMapMin 0.98
        --ColorMapMax 1.04
        --CutPlaneNormal 0.0 0.0 1.0
        --CutPlanePoint 0.0 0.0 0.0
        #--EnableTrace 1 --TraceX 100 --TraceY 100 #Used to debug with ELVIS_PRINTF
        --OutFile naca_Euler2d_CurvedAdapt_Q3_P2
        --CompareFile "${BaselineDir}/naca_Euler2d_CurvedAdapt_Q3_P2_Baseline"
    )
ENDIF()

IF( ELVIS_ENABLE_NEKTAR++_EXTENSION )
    #GET_PROPERTY(ExtensionPath TARGET NektarPlusPlusExtension PROPERTY RELEASE_LOCATION)
    SET( ExtensionPath $<TARGET_FILE:NektarPlusPlusExtension> )
    
    ADD_ELVIS_REGTEST( NektarPlusPlusColorMapSphere 
        --TestName Generic
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/Hex_01_01_01_03Modes_x2_Offset2.xml"
        --Module ${ExtensionPath}
        --Width 1024    --Height 1024    --OutFile NektarPlusPlusColorMapSphere
        --ColorMap "${CMAKE_SOURCE_DIR}/../ColorMaps/Rainbow.xml"
        --ColorMapMin 0.0
        --ColorMapMax 11.0
        --Eye 0.0 0.0 1.0 --Up 0 1 0 --At 0.0 0.0 0.0
        --CutPlaneNormal 0.0 0.0 1.0
        --CutPlanePoint 0.0 0.0 -2.0
        --CompareFile "${BaselineDir}/NektarPlusPlusColorMapSphereBaseline"
    )

    ADD_ELVIS_REGTEST( NektarPlusPlusVolumeRenderSphere_Type5 
        --TestName Generic
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/Hex_01_01_01_03Modes_x2_Offset3.xml"    --Width 1024    --Height 1024    --OutFile NektarPlusPlusVolumeRenderSphere_Type1
        --VolumeRenderingModuleEnabled 1
        --IntegrationType 0
        --Module ${ExtensionPath}
        --Eye 0.0 0.0 0.0 --Up 0 1 0 --At 0.0 0.0 -2.0
        --Breakpoints 4.0 4.1 4.2 5.0 5.1 5.2  6.0 6.1 6.2 
        --Colors 1.0 1.0 1.0 0.0   1.0 1.0 0.0 20.0   1.0 1.0 1.0 0.0   1.0 1.0 1.0 0.0    0.0 1.0 0.0 10.0    1.0 1.0 1.0 0.0   1.0 1.0 1.0 0.0  1.0 0.0 0.0 40.0   1.0 1.0 1.0 0.0
        --CompareFile "${BaselineDir}/NektarPlusPlusVolumeRenderSphere_Type1Baseline"
    )

    ADD_ELVIS_REGTEST(NektarPlusPlusVolumeRenderSphere_Type6 
        --TestName Generic
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/Hex_01_01_01_03Modes_x2_Offset3.xml"    --Width 1024    --Height 1024
        --Module ${ExtensionPath}
        --VolumeRenderingModuleEnabled 1
        --IntegrationType 1
        --Eye 0.0 0.0 0.0 --Up 0 1 0 --At 0.0 0.0 -2.0
        --Breakpoints 4.0 4.1 4.2 5.0 5.1 5.2  6.0 6.1 6.2 
        --Colors 1.0 1.0 1.0 0.0   1.0 1.0 0.0 20.0   1.0 1.0 1.0 0.0   1.0 1.0 1.0 0.0    0.0 1.0 0.0 10.0    1.0 1.0 1.0 0.0   1.0 1.0 1.0 0.0  1.0 0.0 0.0 40.0   1.0 1.0 1.0 0.0
        --OutFile NektarPlusPlusVolumeRenderSphere_Type1_h.1
        --h .1
        --CompareFile "${BaselineDir}/NektarPlusPlusVolumeRenderSphere_Type1_h.1Baseline"
    )
        
    ADD_ELVIS_REGTEST( NektarPlusPlusVolumeRenderSphere_Type1_h01 
        --TestName Generic
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/Hex_01_01_01_03Modes_x2_Offset3.xml"    --Width 1024    --Height 1024
        --Module ${ExtensionPath}
        --VolumeRenderingModuleEnabled 1
        --IntegrationType 1
        --Eye 0.0 0.0 0.0 --Up 0 1 0 --At 0.0 0.0 -2.0
        --Breakpoints 4.0 4.1 4.2 5.0 5.1 5.2  6.0 6.1 6.2 
        --Colors 1.0 1.0 1.0 0.0   1.0 1.0 0.0 20.0   1.0 1.0 1.0 0.0   1.0 1.0 1.0 0.0    0.0 1.0 0.0 10.0    1.0 1.0 1.0 0.0   1.0 1.0 1.0 0.0  1.0 0.0 0.0 40.0   1.0 1.0 1.0 0.0
        --OutFile NektarPlusPlusVolumeRenderSphere_Type1_h.01
        --h .01
        --CompareFile "${BaselineDir}/NektarPlusPlusVolumeRenderSphere_Type1_h.01Baseline"
    )
ENDIF()


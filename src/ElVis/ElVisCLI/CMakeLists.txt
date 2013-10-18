
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

add_definitions(${PNG_DEFINITIONS})
add_definitions(-DQT_NO_KEYWORDS)

SET(HEADERS
    Nektar++Models.h
    NewApproachCanister.h
    VolumeRendering.h
) 

SET(SOURCES
  main.cpp
  Nektar++Models.cpp
  NewApproachCanister.cpp
    VolumeRendering.cpp
)


ADD_ELVIS_EXECUTABLE(${ProjectName} ${HEADERS} ${SOURCES} )
SET_PROPERTY(TARGET ${ProjectName} PROPERTY FOLDER "Core")

TARGET_LINK_LIBRARIES(${ProjectName} 
    ${OPENGL_LIBRARIES}
    ElVisCore
    ${QT_QTCORE_LIBRARY}
    ${QT_QTOPENGL_LIBRARY}
    ${QT_QTGUI_LIBRARY}
    ${QT_MAIN_LIBRARY}
    ${GLUT_glut_LIBRARY}
    ${PNG_LIBRARY}
)


IF( MSVC )
  INSTALL(TARGETS ${ProjectName} EXPORT ElVisLibraries RUNTIME DESTINATION ${ELVIS_BIN_DIR}
                                LIBRARY DESTINATION ${ELVIS_BIN_DIR}
                                ARCHIVE DESTINATION ${ELVIS_LIB_DIR} )
ENDIF()
SET(BaselineDir ${CMAKE_SOURCE_DIR}/../Testing/${ELVIS_FLOAT}/${CMAKE_HOST_SYSTEM_NAME})

IF( ELVIS_ENABLE_JACOBI_EXTENSION )
    GET_PROPERTY(JacobiExtensionPath TARGET JacobiExtension PROPERTY RELEASE_LOCATION)
    ADD_TEST(NAME BulletCutSurfaceP2 WORKING_DIRECTORY ${ELVIS_BIN_DIR} COMMAND ${ProjectName} 
        --TestName CutSurfaceBullet
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/bulletH1P2.dat"
        --Module ${JacobiExtensionPath}
        --Width 1000
        --Height 1000
        --OutFile BulletCutSurfaceP2
        --CompareFile "${BaselineDir}/BulletCutSurfaceP2Baseline"
    )

    ADD_TEST(NAME BulletCutSurfaceP3 WORKING_DIRECTORY ${ELVIS_BIN_DIR} COMMAND ${ProjectName} --TestName CutSurfaceBullet
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/bulletH1P3.dat"
        --Module ${JacobiExtensionPath}
        --Width 1000
        --Height 1000
        --OutFile BulletCutSurfaceP3
        --CompareFile "${BaselineDir}/BulletCutSurfaceP3Baseline"
    )

    ADD_TEST(NAME VolumeRenderBulletFull WORKING_DIRECTORY ${ELVIS_BIN_DIR} COMMAND ${ProjectName} --TestName VolumeRenderBullet
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/bulletH1P2.dat"
        --Module ${JacobiExtensionPath}
        --RenderIntegrationType 0
        --Width 1152  --Height 672 --IntegrationType 7
        --Breakpoints -.04 -.045 -.03 .01 .015 .02 -.001 0.0 .001  --Colors 0.0 0.0 0.0 0.0   1.0 0.0 0.0 100    0.0 0.0 0.0 0.0   0.0 0.0 0.0 0.0   0.0 1.0 0.0 20.0    0.0 0.0 0.0 0.0       0.0 0.0 0.0 0.0  0.0 0.0 1.0 5.0   0.0 0.0 0.0 0.0
        --Eye 3 .1 1.5 --Up 0 1 0 --At 0 0 1.5
        --OutFile VolumeRenderBulletFull
        --CompareFile "${BaselineDir}/VolumeRenderBulletFullBaseline"
    )

    ADD_TEST(NAME VolumeRenderBulletRiemann WORKING_DIRECTORY ${ELVIS_BIN_DIR} COMMAND ${ProjectName} --TestName VolumeRenderBullet
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/bulletH1P2.dat"
        --Module ${JacobiExtensionPath}
        --RenderIntegrationType 0
        --Width 1152  --Height 672 --IntegrationType 0
        --Breakpoints -.04 -.045 -.03 .01 .015 .02 -.001 0.0 .001  --Colors 0.0 0.0 0.0 0.0   1.0 0.0 0.0 100    0.0 0.0 0.0 0.0   0.0 0.0 0.0 0.0   0.0 1.0 0.0 20.0    0.0 0.0 0.0 0.0       0.0 0.0 0.0 0.0  0.0 0.0 1.0 5.0   0.0 0.0 0.0 0.0
        --Eye 3 .1 1.5 --Up 0 1 0 --At 0 0 1.5
        --OutFile VolumeRenderBulletRiemann
        --CompareFile "${BaselineDir}/VolumeRenderBulletRiemannBaseline"
    )

    ADD_TEST(NAME VolumeRenderBulletTrapezoidal WORKING_DIRECTORY ${ELVIS_BIN_DIR} COMMAND ${ProjectName} --TestName VolumeRenderBullet
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/bulletH1P2.dat"
        --Module ${JacobiExtensionPath}
        --RenderIntegrationType 0
        --Width 1152  --Height 672 --IntegrationType 0
        --Breakpoints -.04 -.045 -.03 .01 .015 .02 -.001 0.0 .001  --Colors 0.0 0.0 0.0 0.0   1.0 0.0 0.0 100    0.0 0.0 0.0 0.0   0.0 0.0 0.0 0.0   0.0 1.0 0.0 20.0    0.0 0.0 0.0 0.0       0.0 0.0 0.0 0.0  0.0 0.0 1.0 5.0   0.0 0.0 0.0 0.0
        --Eye 3 .1 1.5 --Up 0 1 0 --At 0 0 1.5
        --OutFile VolumeRenderBulletTrapezoidal
        --CompareFile "${BaselineDir}/VolumeRenderBulletTrapezoidalBaseline"
    )

    ADD_TEST(NAME IsosurfaceBullet1 WORKING_DIRECTORY ${ELVIS_BIN_DIR} COMMAND ${ProjectName} --TestName IsosurfaceBullet
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/bulletH1P2.dat"
        --Module ${JacobiExtensionPath}
        --Width 1152  --Height 672
        --Eye 3 .1 1.5 --Up 0 1 0 --At 0 0 1.5
        --OutFile IsosurfaceBullet1
        --Isovalues 0.0
        --CompareFile "${BaselineDir}/IsosurfaceBullet1Baseline"
    )
ENDIF()

IF( ELVIS_ENABLE_ProjectX_EXTENSION )
    GET_PROPERTY(PXExtensionPath TARGET ProjectXExtension PROPERTY RELEASE_LOCATION)
    ADD_TEST(NAME OneraWingSurface WORKING_DIRECTORY ${ELVIS_BIN_DIR} COMMAND ${ProjectName} --TestName Generic
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/Onera_wing_M0p84_AA3p06_Euler_P2_100000DOF_P2_a06_Primal_3.pxa"
        --Module ${PXExtensionPath}
        --Width 785
        --Height 535
        --Eye 0.2 0.7 1.9 --Up 0 1 0 --At 0.8 0.0 0.4
        --BoundarySurfaces 1 2 3 4 5
        --ColorMap "${CMAKE_SOURCE_DIR}/../ColorMaps/Rainbow.xml"
        --ColorMapMin 0.0
        --ColorMapMax 1.0
        --OutFile OneraWingSurface
        --CompareFile "${BaselineDir}/OneraWingSurface"
    )
ENDIF()

IF( ELVIS_ENABLE_NEKTAR++_EXTENSION )
    GET_PROPERTY(ExtensionPath TARGET NektarPlusPlusExtension PROPERTY RELEASE_LOCATION)
    
    ADD_TEST(NAME NektarPlusPlusColorMapSphere WORKING_DIRECTORY ${ELVIS_BIN_DIR} COMMAND ${ProjectName} --TestName Generic
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

    ADD_TEST(NAME NektarPlusPlusVolumeRenderSphere_Type5 WORKING_DIRECTORY ${ELVIS_BIN_DIR} COMMAND ${ProjectName} --TestName Generic
        --ModelPath "${CMAKE_SOURCE_DIR}/../volumes/Hex_01_01_01_03Modes_x2_Offset3.xml"    --Width 1024    --Height 1024    --OutFile NektarPlusPlusVolumeRenderSphere_Type1
        --VolumeRenderingModuleEnabled 1
        --IntegrationType 0
        --Module ${ExtensionPath}
        --Eye 0.0 0.0 0.0 --Up 0 1 0 --At 0.0 0.0 -2.0
        --Breakpoints 4.0 4.1 4.2 5.0 5.1 5.2  6.0 6.1 6.2 
        --Colors 1.0 1.0 1.0 0.0   1.0 1.0 0.0 20.0   1.0 1.0 1.0 0.0   1.0 1.0 1.0 0.0    0.0 1.0 0.0 10.0    1.0 1.0 1.0 0.0   1.0 1.0 1.0 0.0  1.0 0.0 0.0 40.0   1.0 1.0 1.0 0.0
        --CompareFile "${BaselineDir}/NektarPlusPlusVolumeRenderSphere_Type1Baseline"
    )

    ADD_TEST(NAME NektarPlusPlusVolumeRenderSphere_Type6 WORKING_DIRECTORY ${ELVIS_BIN_DIR} COMMAND ${ProjectName} --TestName Generic
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
        
    ADD_TEST(NAME NektarPlusPlusVolumeRenderSphere_Type1_h01 WORKING_DIRECTORY ${ELVIS_BIN_DIR} COMMAND ${ProjectName} --TestName Generic
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


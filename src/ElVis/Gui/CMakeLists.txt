SET(Project "ElVis")

SET(QT_UI_HEADERS 
)

SET(QT_QOBJECT_HEADERS
    Breakpoints.h
    CameraPropertyManager.h
    ColorMapEditorWidget.h
    ColorMapRect.h
    ColorMapWidget.h
    ColorPropertyManager.h
    ContourDockWidget.h
    DebugSettingsDockWidget.h
    ElementFaceRenderingDockWidget.h
    ElVisUI.h
    IsosurfaceDockWidget.h
    LightingDockWidget.h
    ModelInspectorWidget.h
    ObjectInspectorUI.h
    PlanePropertyManager.h
    PointPropertyManager.h
    Property.hpp
    RenderSettingsDockWidget.h
    SampleOntoNrrdDockWidget.h
    SceneItemsDockWidget.h
    SceneViewWidget.h
    VectorPropertyManager.h
    ViewSettingsUI.h
    VolumeRenderingSettingsWidget.h
)

#QT_WRAP_UI(${Project} QT_PROCESSED_UI_HEADERS QT_PROCESSED_UI_SOURCE ${QT_UI_HEADERS})
QT_WRAP_CPP(${Project} QT_PROCESSED_MOC_SOURCE ${QT_QOBJECT_HEADERS})

SET(Sources
    ApplicationState.cpp    
    Breakpoints.cpp
    ColorMapEditorWidget.cpp
    ColorMapRect.cpp
    ColorMapWidget.cpp
    ContourDockWidget.cpp
    DebugSettingsDockWidget.cpp
    ElementFaceRenderingDockWidget.cpp
    ElVisUI.cpp
    IsosurfaceDockWidget.cpp
    LightingDockWidget.cpp
    main.cpp
    ModelInspectorWidget.cpp
    ObjectInspectorUI.cpp
    RenderSettingsDockWidget.cpp
    SampleOntoNrrdDockWidget.cpp
    SceneItemsDockWidget.cpp
    SceneViewWidget.cpp
    ViewSettingsUI.cpp
    VolumeRenderingSettingsWidget.cpp
)

SET(Headers
    ApplicationState.h
    ElVis.qss
)

# Allows interoperability between boost::signals and qt signals.
add_definitions(-DQT_NO_KEYWORDS)

ADD_ELVIS_EXECUTABLE(${Project} 
    ${Sources} 
    ${Headers}
    ${QT_QOBJECT_HEADERS}
    ${QT_PROCESSED_UI_HEADERS}
    #${QT_PROCESSED_UI_SOURCE}
    ${QT_PROCESSED_MOC_SOURCE}
)

SET_PROPERTY(TARGET ${Project} PROPERTY FOLDER "Core")

INCLUDE_DIRECTORIES(
    ${CMAKE_SOURCE_DIR}/Externals/tinyxml
)

TARGET_LINK_LIBRARIES(${Project}
    ${QT_LIBRARIES}
    ${QT_QTOPENGL_LIBRARY}
    ${BOOST_LIBRARY}
    ElVisCore
    QtPropertyBrowser
    glew
    ${PNG_LIBRARY}
    ${ZLIB_LIBRARY}
    ${TinyXML_Library}
)

INSTALL(FILES ElVis.qss
    DESTINATION ${ELVIS_BIN_DIR}
)

if(WIN32)
    add_custom_command(
        TARGET ${Project}
        COMMAND ${CMAKE_COMMAND} -E copy ${CORE_BUILD_PATH}/Debug/ElVisCore.dll ${CMAKE_CURRENT_BINARY_DIR}/Debug
        COMMAND ${CMAKE_COMMAND} -E copy ${GLEW_BUILD_PATH}/Debug/glew.dll ${CMAKE_CURRENT_BINARY_DIR}/Debug
        COMMAND ${CMAKE_COMMAND} -E copy ${optix_DLL} ${CMAKE_CURRENT_BINARY_DIR}/Debug
        COMMAND ${CMAKE_COMMAND} -E copy ${OptiX_INSTALL_DIR}/bin${bit_dest}/cudart64_42_9.dll ${CMAKE_CURRENT_BINARY_DIR}/Debug
    )
endif()
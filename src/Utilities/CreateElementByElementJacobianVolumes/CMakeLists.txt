

 
SET(ProjectName CreateElementByElementJacobianVolumes)



INCLUDE_DIRECTORIES(${NEKTAR++_INCLUDE_DIRS} 
                    ${OPENGL_INCLUDE_DIR} 
                    ${GLEW_INCLUDE_DIR}
                    ${MATH_INCLUDE_DIR}
                    #${OptiX_INCLUDE}/../SDK/cuda
                    )
                    
SET(HEADERS
    
) 

SET(SOURCES
    main.cpp
)

LINK_DIRECTORIES(${Boost_LIBRARY_DIRS} ${NEKTAR++_LIBRARY_DIRS})

ADD_ELVIS_EXECUTABLE(${ProjectName} ${HEADERS} ${SOURCES} )

# Adding a Nektar library messes with the postfix.  Fix it up here.
#SET_TARGET_PROPERTIES(${ProjectName} PROPERTIES DEBUG_POSTFIX "")
#SET_TARGET_PROPERTIES(${ProjectName} PROPERTIES MINSIZEREL_POSTFIX "")
#SET_TARGET_PROPERTIES(${ProjectName} PROPERTIES RELWITHDEBINFO_POSTFIX "")

TARGET_LINK_LIBRARIES(${ProjectName} 
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${Boost_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ElVisNative
    JacobiExtension
)

INSTALL(TARGETS ${ProjectName} RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX} 
                                LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
                                ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX} )



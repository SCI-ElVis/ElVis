 
SET(ProjectName GenerateStaticTables)

                    
SET(HEADERS
    
) 

SET(SOURCES
    main.cpp
)

ADD_ELVIS_EXECUTABLE(${ProjectName} ${HEADERS} ${SOURCES} )



SET_PROPERTY(TARGET ${ProjectName} PROPERTY FOLDER "Utilities")

INSTALL(TARGETS ${ProjectName} RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX} 
                                LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
                                ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX} )



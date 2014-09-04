#===============================================================================
# Macro for CallGrind
#===============================================================================
IF( NOT WIN32 )
  FIND_PROGRAM( KCACHEGRIND kcachegrind )
  IF( NOT KCACHEGRIND )
    MESSAGE( WARNING "Please install kcachegrind for graphical profiling." )
    SET( USE_KCACHEGRIND OFF )
  ELSE()
    SET( USE_KCACHEGRIND ON CACHE BOOL "Use kcachegrind with callgrind."  )
  ENDIF()

  MACRO( ADD_CALLGRIND_TEST UNIT_TEST )

    SET( CALLGRIND_MESSAGE "Callgrind file: $<TARGET_FILE_DIR:${UNIT_TEST}_build>/callgrind.out" )

    IF( KCACHEGRIND AND USE_KCACHEGRIND )
      #Add the callgrind target with kcachegrind
      ADD_CUSTOM_TARGET( ${UNIT_TEST}_callgrind 
                         COMMAND ${CALLGRIND_COMMAND} --callgrind-out-file=$<TARGET_FILE_DIR:${UNIT_TEST}_build>/callgrind.out $<TARGET_FILE:${UNIT_TEST}_build> 
                         COMMAND ${KCACHEGRIND} $<TARGET_FILE_DIR:${UNIT_TEST}_build>/callgrind.out 
                         COMMAND ${CMAKE_COMMAND} -E echo
                         COMMAND ${CMAKE_COMMAND} -E echo ${CALLGRIND_MESSAGE}
                         COMMAND ${CMAKE_COMMAND} -E echo
                         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
    ELSE()
      #Add the callgrind target
      ADD_CUSTOM_TARGET( ${UNIT_TEST}_callgrind 
                         COMMAND ${CALLGRIND_COMMAND} --callgrind-out-file=$<TARGET_FILE_DIR:${UNIT_TEST}_build>/callgrind.out $<TARGET_FILE:${UNIT_TEST}_build>
                         COMMAND ${CMAKE_COMMAND} -E echo
                         COMMAND ${CMAKE_COMMAND} -E echo ${CALLGRIND_MESSAGE} 
                         COMMAND ${CMAKE_COMMAND} -E echo
                         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
    ENDIF()

  ENDMACRO()
ENDIF()
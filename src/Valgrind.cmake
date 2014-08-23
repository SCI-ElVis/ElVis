#==================================================
# Setup valgrind for memory checking
#==================================================
IF( NOT WIN32 )
    FIND_PROGRAM( VALGRIND_EXEC valgrind  )
    
    SET( VALGRIND_FLAGS --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=50 --error-exitcode=1 --gen-suppressions=all )
    SET( VALGRIND_FLAGS ${VALGRIND_FLAGS} --suppressions=${CMAKE_SOURCE_DIR}/Externals/valgrind/ElVis.supp )
    SET( VALGRIND_FLAGS ${VALGRIND_FLAGS} --suppressions=${CMAKE_SOURCE_DIR}/Externals/valgrind/Qt.supp )
    IF( APPLE )
      SET( VALGRIND_FLAGS ${VALGRIND_FLAGS} --suppressions=${CMAKE_SOURCE_DIR}/Externals/valgrind/OSX.supp )
    ENDIF()
    
    SET( VALGRIND_EXTRA_FLAGS CACHE STRING "Additonal valgrind flags: Usefull ones are --track-origins=yes" )
    SET( VALGRIND_COMMAND ${VALGRIND_EXEC} ${VALGRIND_FLAGS} ${VALGRIND_EXTRA_FLAGS} )
    
    IF( BUILD_TYPE AND BUILD_TYPE MATCHES "DEBUG" )
      SET( STACKCHECK_COMMAND ${VALGRIND_EXEC} --tool=exp-sgcheck --gen-suppressions=all )
      SET( STACKCHECK_COMMAND ${STACKCHECK_COMMAND} --suppressions=${CMAKE_SOURCE_DIR}/Externals/valgrind/sgcheck.supp )
    ELSE()
      SET( STACKCHECK_COMMAND ${CMAKE_COMMAND} -E echo "Please set CMAKE_BUILD_TYPE to \\'debug\\' for stack checking of" )
    ENDIF()
    
    SET( CALLGRIND_COMMAND ${VALGRIND_EXEC} --tool=callgrind )
ENDIF()

IF( CMAKE_BUILD_TYPE )
  STRING( TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE ) 
ENDIF()

IF( BUILD_TYPE AND BUILD_TYPE MATCHES "COVERAGE" )

  FIND_PROGRAM( LCOV lcov )
  IF( NOT LCOV )
    MESSAGE( FATAL_ERROR "Please install lcov and add it to your path." )
  ENDIF()

  FIND_PROGRAM( GENHTML genhtml )
  IF( NOT GENHTML )
    MESSAGE( FATAL_ERROR "Could not find genhtml. Please install lcov and add it to your path." )
  ENDIF()

  FIND_PROGRAM( GCOV ${GCOV_COMMAND} )
  IF( NOT GCOV )
    MESSAGE( FATAL_ERROR "Could not find gcov. Please install gcov and add it to your path." )
  ENDIF()
  
  IF( CYGWIN )
    SET( OPEN cygstart )
  ELSEIF( APPLE )
    SET( OPEN open )
  ELSE()
    SET( OPEN xdg-open )
  ENDIF()

  SET( COVERAGE_INFO coverage.info )
  SET( HTMLDIR CoverageHTML )
  SET( LCOV_FLAGS --capture -q --gcov-tool ${GCOV} --base-directory ${CMAKE_SOURCE_DIR} --directory . --output-file ${COVERAGE_INFO} )
  SET( GENHTML_FLAGS ${COVERAGE_INFO} -q --legend --frames --show-details --demangle-cpp --output-directory ${HTMLDIR} )
  SET( BRANCH_COVERAGE )
  
  EXECUTE_PROCESS( COMMAND ${LCOV} --version OUTPUT_VARIABLE LCOV_VERSION )
  #MESSAGE( STATUS "Found ${LCOV} ${LCOV_VERSION}" )
  STRING( REGEX MATCH "[0-9]+[.][0-9]+"  LCOV_VERSION ${LCOV_VERSION} )
  STRING( REGEX MATCH "^[0-9]+"  LCOV_MAJOR_VERSION ${LCOV_VERSION} )
  STRING( REGEX MATCH "[0-9]+$" LCOV_MINOR_VERSION ${LCOV_VERSION} )

  IF( (${LCOV_MAJOR_VERSION} GREATER 1) OR 
      (${LCOV_MAJOR_VERSION} EQUAL 1 AND ${LCOV_MINOR_VERSION} GREATER 9) )
    SET( BRANCH_COVERAGE --rc lcov_branch_coverage=1 )
    SET( LCOV_FLAGS ${LCOV_FLAGS} --no-external ${BRANCH_COVERAGE} )
    SET( GENHTML_FLAGS ${GENHTML_FLAGS} ${BRANCH_COVERAGE} )
  ENDIF()

  #This processes the coverage files only for the actual source, which is appropraite for the website
  ADD_CUSTOM_TARGET( coverage_info
                     COMMAND ${CMAKE_COMMAND} -E echo "Generating tracefile ${COVERAGE_INFO}..."
                     COMMAND ${LCOV} ${LCOV_FLAGS}
                     COMMAND ${LCOV} --remove ${COVERAGE_INFO} "*_?test.*" -q --output-file ${COVERAGE_INFO} ${BRANCH_COVERAGE}
                     COMMAND ${LCOV} --summary ${COVERAGE_INFO} ${BRANCH_COVERAGE}
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )
  
  MACRO( ADD_COVERAGE COVERAGE_NAME TARGET_NAME )  
    #This generates the coverage information for only the actual source
    ADD_CUSTOM_TARGET( ${COVERAGE_NAME}
                       COMMAND ${CMAKE_COMMAND} -E echo "Generating tracefile ${COVERAGE_INFO}..."
                       COMMAND ${LCOV} ${LCOV_FLAGS}
                       COMMAND ${LCOV} --remove ${COVERAGE_INFO} "*_?test.*" -q --output-file ${COVERAGE_INFO} ${BRANCH_COVERAGE}
                       COMMAND ${CMAKE_COMMAND} -E echo "Generating html documents in ${HTMLDIR}..."
                       COMMAND ${GENHTML} ${GENHTML_FLAGS} 
                       COMMAND ${LCOV} --summary ${COVERAGE_INFO} ${BRANCH_COVERAGE}
                       DEPENDS ${TARGET_NAME}
                       WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )
  ENDMACRO()
  
  #Create the generic coverage target that simply generates coverage information without qtest files
  ADD_COVERAGE( coverage "" )
  
  MACRO( ADD_COVERAGE_TEST COVERAGE_NAME TARGET_NAME )  
    #This provides coverage information that includes btest files
    ADD_CUSTOM_TARGET( ${COVERAGE_NAME}
                       COMMAND ${CMAKE_COMMAND} -E echo "Generating tracefile ${COVERAGE_INFO}..."
                       COMMAND ${LCOV} ${LCOV_FLAGS}
                       COMMAND ${CMAKE_COMMAND} -E echo "Generating html documents in ${HTMLDIR}..."
                       COMMAND ${GENHTML}  ${GENHTML_FLAGS} 
                       COMMAND ${LCOV} --summary ${COVERAGE_INFO} ${BRANCH_COVERAGE}
                       DEPENDS ${TARGET_NAME}
                       WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )
  ENDMACRO()
 
  #Create the generic coverage_btest target that simply generates coverage information with btest files
  ADD_COVERAGE_TEST( coverage_btest "" )
  
  ADD_CUSTOM_TARGET( coverage_show
                     COMMAND ${OPEN} ${HTMLDIR}/index.html
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )

  ADD_CUSTOM_TARGET( coverage_clean
                     COMMAND ${CMAKE_COMMAND} -E remove_directory ${HTMLDIR}
                     COMMAND ${CMAKE_COMMAND} -E remove ${COVERAGE_INFO}
                     COMMAND find . -name "*.gcda" | xargs rm -f
                     COMMAND ${CMAKE_COMMAND} -E echo "-- Removed all coverage files"
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )

  ADD_CUSTOM_TARGET( coverage_cleaner
                     COMMAND find . -name "*.gcno" | xargs rm -f
                     COMMAND ${CMAKE_MAKE_PROGRAM} clean
                     COMMAND ${CMAKE_COMMAND} -E echo "-- Removed all binary and .gcno files"
                     DEPENDS coverage_clean
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )
ELSE()
  SET( NO_COVERAGE_MESSAGE "Please set the CMAKE_BUILD_TYPE to \\'coverage\\' or \\'coverage_release\\' to generate coverage information." )
  
  MACRO( ADD_COVERAGE COVERAGE_NAME TARGET_NAME )
    ADD_CUSTOM_TARGET( ${COVERAGE_NAME} COMMAND ${CMAKE_COMMAND} -E echo ${NO_COVERAGE_MESSAGE} )
  ENDMACRO()
  MACRO( ADD_COVERAGE_TEST COVERAGE_NAME TARGET_NAME )
    ADD_COVERAGE( ${COVERAGE_NAME} "" )
  ENDMACRO()

  ADD_COVERAGE( coverage "" )
  ADD_COVERAGE_TEST( coverage_btest "" )
  ADD_CUSTOM_TARGET( coverage_info    COMMAND ${CMAKE_COMMAND} -E echo ${NO_COVERAGE_MESSAGE} )
  ADD_CUSTOM_TARGET( coverage_show    COMMAND ${CMAKE_COMMAND} -E echo ${NO_COVERAGE_MESSAGE} )
  ADD_CUSTOM_TARGET( coverage_clean   COMMAND ${CMAKE_COMMAND} -E echo ${NO_COVERAGE_MESSAGE} )
  ADD_CUSTOM_TARGET( coverage_cleaner COMMAND ${CMAKE_COMMAND} -E echo ${NO_COVERAGE_MESSAGE} )
ENDIF()

ADD_COVERAGE( check_coverage check )

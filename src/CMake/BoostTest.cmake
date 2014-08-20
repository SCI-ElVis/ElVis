#First need to clear all objects from the CACHED list. Othewise we get duplicates from multiple executions of cmake
UNSET( ALL_BOOST_UNIT_OBJECTS CACHE )

# Produce a warning message if any unit tests are skipped
IF( UNIT_SKIP )
  MESSAGE( "" )  
  MESSAGE( "=================================" )
  MESSAGE( "       !!!!  WARNING !!!!" )
  MESSAGE( "The following unit tests are not compiled:" )
  FOREACH( UNIT ${UNIT_SKIP} )
    MESSAGE( "  ${UNIT}" )
  ENDFOREACH()
  MESSAGE( "---------------------------------" )
  MESSAGE( "Could someone please fix or remove them?" )
  MESSAGE( "The list can be modified in CMakeLists.txt in ")
  MESSAGE( "the ElVis root directory" )
  MESSAGE( "=================================" )
  MESSAGE( "" )  
ENDIF()

#Create a main object so it only needs to be compield once
ADD_LIBRARY( BOOST_TEST_MAIN_OBJECT EXCLUDE_FROM_ALL OBJECT ${CMAKE_SOURCE_DIR}/CMake/main_boost_btest.cpp )
SET( BOOST_TEST_MAIN $<TARGET_OBJECTS:BOOST_TEST_MAIN_OBJECT> )

#Flags for executing boost-test
SET( BOOST_TEST_FLAGS --report_level=short )

#------------------------------------------------------------------------------
MACRO( ADD_ELVIS_BOOST_TEST UNIT_TEST UNIT_TEST_SRC )
  
  #Create an object library and forward the call. This allows unit tests to share objects
  ADD_LIBRARY( ${UNIT_TEST}_OBJECT EXCLUDE_FROM_ALL OBJECT ${${UNIT_TEST_SRC}} )

  SET( OBJECT $<TARGET_OBJECTS:${UNIT_TEST}_OBJECT> )

  #Create the unit test targest
  ADD_ELVIS_BOOST_TEST_OBJECTS( ${UNIT_TEST} OBJECT ${ARGN} )
  
ENDMACRO()

#------------------------------------------------------------------------------
MACRO( ADD_ELVIS_BOOST_TEST_OBJECTS UNIT_TEST BOOST_UNIT_TEST_OBJECTS )
# This gnerates 4 targest for each unit test
# ${UNIT_TEST}_build      : compiles the unit test
# ${UNIT_TEST}            : compiles and executes the unit test
# ${UNIT_TEST}_memcheck   : compiles and executes the unit test with valgrind
# ${UNIT_TEST}_stackcheck : compiles and executes the unit test with valgrind
# ${UNIT_TEST}_coverage   : compiles and executes the unit test and generates coverage inforamtion
                 
  #Create the build target
  ADD_EXECUTABLE( ${UNIT_TEST}_build EXCLUDE_FROM_ALL ${BOOST_TEST_MAIN} ${${BOOST_UNIT_TEST_OBJECTS}} )
  TARGET_LINK_LIBRARIES( ${UNIT_TEST}_build ${ARGN} ${BOOST_LIBRARY} )
  SET_TARGET_PROPERTIES( ${UNIT_TEST}_build PROPERTIES OUTPUT_NAME "${UNIT_TEST}")

  #Add a target for executing the test
  ADD_CUSTOM_TARGET( ${UNIT_TEST} COMMAND $<TARGET_FILE:${UNIT_TEST}_build> ${BOOST_TEST_FLAGS} $(UNITARGS)
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )

  #Add memory checking target
  ADD_MEMCHECK( ${UNIT_TEST} )
  
  #Add the coverage target
  ADD_COVERAGE_TEST( ${UNIT_TEST}_coverage ${UNIT_TEST} )
  
  #Add the callgrind target
  ADD_CALLGRIND_TEST( ${UNIT_TEST} )
   
ENDMACRO()


#------------------------------------------------------------------------------
# Call GenerateBoostTests with the ElVis library dependencies of the boost unit test folder
MACRO( GenerateBoostTests )

  FILE_GLOB( UNIT_SRC "*_btest.cpp" )

  FOREACH( UNIT_SRC_FILE ${UNIT_SRC} )

    GET_FILENAME_COMPONENT( UNIT_TEST ${UNIT_SRC_FILE} NAME_WE )

    #Check to see if this test should be skipped
    LIST( FIND UNIT_SKIP ${UNIT_TEST} SKIP_TEST )

    IF( ${SKIP_TEST} EQUAL -1 )
      ADD_ELVIS_BOOST_TEST( ${UNIT_TEST} UNIT_SRC_FILE ${ARGN} )
      
      #Make all check_build depend on all unit tests
      ADD_DEPENDENCIES( check_build ${UNIT_TEST}_build  )
  
      #Add only individual files to ctest
      ADD_TEST( NAME ${UNIT_TEST} COMMAND $<TARGET_FILE:${UNIT_TEST}_build> 
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
      SET_TESTS_PROPERTIES( ${UNIT_TEST} PROPERTIES LABELS ElVisCheck )
      
      IF( NOT WIN32 )
        #Add intividiaul memcheck tests to ctest
        ADD_TEST( NAME ${UNIT_TEST}_memcheck COMMAND ${VALGRIND_COMMAND} $<TARGET_FILE:${UNIT_TEST}_build>
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
        SET_TESTS_PROPERTIES( ${UNIT_TEST}_memcheck PROPERTIES LABELS ElVisMemCheck )
          
        #Add intividiaul stackcheck tests to ctest
        ADD_TEST( NAME ${UNIT_TEST}_stackcheck COMMAND ${STACKCHECK_COMMAND} $<TARGET_FILE:${UNIT_TEST}_build>
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
        SET_TESTS_PROPERTIES( ${UNIT_TEST}_stackcheck PROPERTIES LABELS ElVisStackCheck )
        
        #A list of all the unit test objects in this folder
        SET( BOOST_UNIT_TEST_OBJECTS ${BOOST_UNIT_TEST_OBJECTS} $<TARGET_OBJECTS:${UNIT_TEST}_OBJECT> )
      ENDIF()
    ENDIF()
  ENDFOREACH()
 
   # Generate a target for the given folder that executes all the tests in the folder
  GET_FILENAME_COMPONENT( FOLDER_TEST ${CMAKE_CURRENT_BINARY_DIR} NAME_WE )
  ADD_ELVIS_BOOST_TEST_OBJECTS( ${FOLDER_TEST} BOOST_UNIT_TEST_OBJECTS ${ARGN} )
  ADD_DEPENDENCIES( check_build ${FOLDER_TEST}_build )

  #Automatically build up a list of all unit test objects for the 'unit' target
  SET( ALL_BOOST_UNIT_OBJECTS ${ALL_BOOST_UNIT_OBJECTS} ${BOOST_UNIT_TEST_OBJECTS} CACHE INTERNAL "All the unit tets objects" FORCE )
 
ENDMACRO()

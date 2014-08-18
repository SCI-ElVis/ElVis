
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

INCLUDE(XCodeFileGlob.cmake)
INCLUDE(Valgrind.cmake)

#MACRO(ADD_ELVIS_QTEST testname testsrc)
#    SET(test_${testname}_SRCS ${testsrc})
# 
#    QT4_AUTOMOC(${test_${testname}_SRCS})
#    ADD_EXECUTABLE(test_${testname} ${test_${testname}_SRCS} ${test_additional})
#    TARGET_LINK_LIBRARIES(test_${testname} ${QT_QTCORE_LIBRARY}
#            ${QT_QTTEST_LIBRARY} ${QT_QTGUI_LIBRARY}
#            ${GSOAP_LIBRARIES} ${QT_QTLOCATION_LIBRARY})
# 
#    ADD_TEST(test_${testname} test_${testname})
#ENDMACRO()

#QT libraries used to QTest
SET(QTEST_LIBS 
    ${QT_QTCORE_LIBRARY} 
    ${QT_QTTEST_LIBRARY} 
    ${QT_QTGUI_LIBRARY}
    ${GSOAP_LIBRARIES} 
    ${QT_QTLOCATION_LIBRARY}
   )

#------------------------------------------------------------------------------
MACRO( ADD_ELVIS_QTEST UNIT_TEST UNIT_TEST_SRC )
# This gnerates 4 targest for each unit test
# ${UNIT_TEST}_build      : compiles the unit test
# ${UNIT_TEST}            : compiles and executes the unit test
# ${UNIT_TEST}_memcheck   : compiles and executes the unit test with valgrind
# ${UNIT_TEST}_stackcheck : compiles and executes the unit test with valgrind
# ${UNIT_TEST}_coverage   : compiles and executes the unit test and generates coverage inforamtion
                 
  #Automoc the source files
  #QT4_AUTOMOC(${${UNIT_TEST_SRC}})

  #Create the build target
  ADD_EXECUTABLE( ${UNIT_TEST}_build EXCLUDE_FROM_ALL ${${UNIT_TEST_SRC}} )
  TARGET_LINK_LIBRARIES( ${UNIT_TEST}_build ${ARGN} ${QTEST_LIBS} )
  SET_TARGET_PROPERTIES( ${UNIT_TEST}_build PROPERTIES OUTPUT_NAME "${UNIT_TEST}" AUTOMOC TRUE )
  INCLUDE_DIRECTORIES( $<TARGET_FILE_DIR:${UNIT_TEST}_build> )

  #Add a target for executing the test
  ADD_CUSTOM_TARGET( ${UNIT_TEST} COMMAND $<TARGET_FILE:${UNIT_TEST}_build> ${BOOST_TEST_FLAGS} $(UNITARGS)
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )

  #Add the memory checking target
  ADD_CUSTOM_TARGET( ${UNIT_TEST}_memcheck COMMAND ${VALGRIND_COMMAND} $<TARGET_FILE:${UNIT_TEST}_build> $(UNITARGS)
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )

  #Add the stack checking target
  ADD_CUSTOM_TARGET( ${UNIT_TEST}_stackcheck COMMAND ${STACKCHECK_COMMAND} $<TARGET_FILE:${UNIT_TEST}_build> $(UNITARGS)
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )

  #Add the coverage target
  ADD_COVERAGE_TEST( ${UNIT_TEST}_coverage ${UNIT_TEST} )
  
  #Add the callgrind target
  if( NOT MSVC )
    ADD_CALLGRIND_TEST( ${UNIT_TEST} )
  endif()
  
ENDMACRO()


#------------------------------------------------------------------------------
# Call GenerateQTests with the ElVis library dependencies of the Qt unit test folder
MACRO( GenerateQTests )

  FILE_GLOB( UNIT_SRC "*_qtest.cpp" )

  FOREACH( UNIT_SRC_FILE ${UNIT_SRC} )

    GET_FILENAME_COMPONENT( UNIT_TEST ${UNIT_SRC_FILE} NAME_WE )

    #Check to see if this test should be skipped
    LIST( FIND UNIT_SKIP ${UNIT_TEST} SKIP_TEST )

    IF( ${SKIP_TEST} EQUAL -1 )
      ADD_ELVIS_QTEST( ${UNIT_TEST} UNIT_SRC_FILE ${ARGN} )
      
      #Make all check_build depend on all unit tests
      ADD_DEPENDENCIES( check_build ${UNIT_TEST}_build  )
  
      #Add only individual files to ctest
      ADD_TEST( NAME ${UNIT_TEST} COMMAND $<TARGET_FILE:${UNIT_TEST}_build> 
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
      SET_TESTS_PROPERTIES( ${UNIT_TEST} PROPERTIES LABELS ElVisCheck )
      
      #Add intividiaul memcheck tests to ctest
      ADD_TEST( NAME ${UNIT_TEST}_memcheck COMMAND ${VALGRIND_COMMAND} $<TARGET_FILE:${UNIT_TEST}_build>
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
      SET_TESTS_PROPERTIES( ${UNIT_TEST}_memcheck PROPERTIES LABELS ElVisMemCheck )
      
      #Add intividiaul stackcheck tests to ctest
      ADD_TEST( NAME ${UNIT_TEST}_stackcheck COMMAND ${STACKCHECK_COMMAND} $<TARGET_FILE:${UNIT_TEST}_build>
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
      SET_TESTS_PROPERTIES( ${UNIT_TEST}_stackcheck PROPERTIES LABELS ElVisStackCheck )
 
   ENDIF()
  ENDFOREACH()
 
ENDMACRO()

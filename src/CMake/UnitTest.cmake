ADD_CUSTOM_TARGET( check_build )

ADD_CUSTOM_TARGET( check 
                   COMMAND ${CMAKE_CTEST_COMMAND} $(CTESTARGS) --output-on-failure --no-compress-output -L ElVisCheck
                   DEPENDS check_build
                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )

IF( NOT WIN34 )
  ADD_CUSTOM_TARGET( memcheck 
                     COMMAND ${CMAKE_CTEST_COMMAND} $(CTESTARGS) --output-on-failure --no-compress-output -L ElVisMemCheck
                     DEPENDS check_build
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )

  ADD_CUSTOM_TARGET( stackcheck 
                     COMMAND ${CMAKE_CTEST_COMMAND} $(CTESTARGS) --output-on-failure --no-compress-output -L ElVisStackCheck
                     DEPENDS check_build
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )

  ADD_CUSTOM_TARGET( MemoryCheck 
                     COMMAND ${CMAKE_CTEST_COMMAND} $(CTESTARGS) --output-on-failure --no-compress-output -L \"ElVisMemCheck|ElVisStackCheck\"
                     DEPENDS check_build
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )
ENDIF()

INCLUDE(${CMAKE_SOURCE_DIR}/CMake/XCodeFileGlob.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/CMake/Valgrind.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/CMake/Coverage.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/CMake/CallGrind.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/CMake/QTest.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/CMake/BoostTest.cmake)

#===============================================================================
# Do a file glob that removes XCode temporary files
#===============================================================================

#Call with for example FILE_GLOB_RECURSE( SRCS "*.h" "*.cpp" )
FUNCTION( FILE_GLOB_IMPL GLOB_TYPE RESULT )

  SET( ALL_SRCS )
  SET( i 1 ) #Skip the RESULTS argument

  WHILE( i LESS ${ARGC} )
    SET( PATTERN ${ARGV${i}} )
    FILE( ${GLOB_TYPE} SRCS ${PATTERN} )
    FILE( GLOB_RECURSE XCODE_TMP "._${PATTERN}" )

    IF (${XCODE_TMP})
      LIST( REMOVE_ITEM SRCS ${XCODE_TMP} )
    ENDIF ( )

    LIST( APPEND ALL_SRCS ${SRCS} )
    MATH( EXPR i "${i} + 1" )
  ENDWHILE()

  SET( ${RESULT} ${ALL_SRCS} PARENT_SCOPE )

ENDFUNCTION()



FUNCTION( FILE_GLOB RESULT )

  FILE_GLOB_IMPL( GLOB RESULT_TMP ${ARGN} )

  SET( ${RESULT} ${RESULT_TMP} PARENT_SCOPE )

ENDFUNCTION()

FUNCTION( FILE_GLOB_RECURSE RESULT )

  FILE_GLOB_IMPL( GLOB_RECURSE RESULT_TMP ${ARGN} )

  SET( ${RESULT} ${RESULT_TMP} PARENT_SCOPE )

ENDFUNCTION()

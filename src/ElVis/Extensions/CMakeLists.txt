
SET(ELVIS_ENABLE_JACOBI_EXTENSION true CACHE BOOL "Enable the Jacobi Extension.")
SET(ELVIS_ENABLE_NEKTAR++_EXTENSION false CACHE BOOL "Enable the Nektar++ Extension.")
SET(ELVIS_ENABLE_NEKTAR_EXTENSION false CACHE BOOL "Enable the Nektar Extension.")
SET(ELVIS_ENABLE_ProjectX_EXTENSION false CACHE BOOL "Enable the ProjectX Extension.")

IF( ELVIS_ENABLE_JACOBI_EXTENSION )
	SUBDIRS(JacobiExtension)
ENDIF()

IF( ELVIS_ENABLE_NEKTAR++_EXTENSION )
	SUBDIRS(NektarPlusPlusExtension)
ENDIF()

IF( ELVIS_ENABLE_NEKTAR_EXTENSION )
	SUBDIRS(NektarExtension)
ENDIF()

IF( ELVIS_ENABLE_ProjectX_EXTENSION )
  SUBDIRS(ProjectXExtension)
ENDIF()

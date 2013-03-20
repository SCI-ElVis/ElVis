
SUBDIRS(
    CodeGenerator
    Stats
    GenerateStaticTables
    #GenerateTestingJacobiVolumes
)

#IF( ELVIS_ENABLE_NEKTAR++_EXTENSION )
#	SUBDIRS(
		#ConvertNektar++HexMeshToSampleMesh
    #GenerateNektar++HexMesh
#)
#ENDIF()

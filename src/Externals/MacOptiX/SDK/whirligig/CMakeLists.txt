
if(GLUT_FOUND AND OPENGL_FOUND)
  include_directories(${GLUT_INCLUDE_DIR})
  add_definitions(-DGLUT_FOUND -DGLUT_NO_LIB_PRAGMA)

  # See top level CMakeLists.txt file for documentation of OPTIX_add_sample_executable.
  OPTIX_add_sample_executable( whirligig
    whirligig.cpp
    ringsOfSpheres.h
    ringsOfSpheres.cpp
    sphere_list.cu

    # These files are common among multiple samples
    adaptive_pinhole_camera.cu
    commonStructs.h
    glass.cu
    helpers.h
    phong.cu
    phong.h
    pinhole_camera.cu
    random.h
    sunsky.h
    sunsky.cu
    )

else()
  # GLUT or OpenGL not found
  message("Disabling whirligig, which requires glut and opengl.")
endif()




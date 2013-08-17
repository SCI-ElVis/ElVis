
/*
 * Copyright (c) 2008 - 2011 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

//-----------------------------------------------------------------------------
//
//  displacement.cpp: Renders an Obj model with displacement mapping.
//  
//-----------------------------------------------------------------------------

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <sutil.h>
#include <GLUTDisplay.h>
#include <PlyLoader.h>
#include <ObjLoader.h>
#include "commonStructs.h"
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include "random.h"
#include "MeshScene.h"

using namespace optix;


//------------------------------------------------------------------------------
//
// MeshViewer class 
//
//------------------------------------------------------------------------------
class MeshViewer : public MeshScene
{
public:
  //
  // Helper types
  //
  enum ShadeMode
  {
    SM_PHONG=0,
    SM_AO,
    SM_NORMAL,
    SM_ONE_BOUNCE_DIFFUSE
  };

  enum CameraMode
  {
    CM_PINHOLE=0,
    CM_ORTHO
  };

  //
  // MeshViewer specific  
  //
  MeshViewer();

  // Setters for controlling application behavior
  void setShadeMode( ShadeMode mode )              { m_shade_mode = mode;               }
  void setCameraMode( CameraMode mode )            { m_camera_mode = mode;              }
  void setAORadius( float ao_radius )              { m_ao_radius = ao_radius;           }
  void setAOSampleMultiplier( int ao_sample_mult ) { m_ao_sample_mult = ao_sample_mult; }
  void setLightScale( float light_scale )          { m_light_scale = light_scale;       }

  //
  // From SampleScene
  //
  virtual void   initScene( InitialCameraData& camera_data );
  virtual void   doResize( unsigned int width, unsigned int height );
  virtual void   trace( const RayGenCameraData& camera_data );
  virtual void   cleanUp();
  virtual bool   keyPressed(unsigned char key, int x, int y);
  virtual Buffer getOutputBuffer();

private:
  void initContext();
  void initLights();
  void initMaterial();
  void initGeometry();
  void initCamera( InitialCameraData& cam_data );
  void preprocess();

  void resetAccumulation();
  void genRndSeeds( unsigned int width, unsigned int height );

  CameraMode    m_camera_mode;

  ShadeMode     m_shade_mode;
  float         m_ao_radius;
  int           m_ao_sample_mult;
  float         m_light_scale;

  Material      m_material;
  Aabb          m_aabb;
  Buffer        m_rnd_seeds;
  Buffer        m_accum_buffer;

  float         m_scene_epsilon;
  int           m_frame;
};


//------------------------------------------------------------------------------
//
// MeshViewer implementation
//
//------------------------------------------------------------------------------


MeshViewer::MeshViewer():
  MeshScene          ( false, false, false ),
  m_camera_mode       ( CM_PINHOLE ),
  m_shade_mode        ( SM_AO ),
  m_ao_radius         ( 1.0f ),
  m_ao_sample_mult    ( 1 ),
  m_light_scale       ( 1.0f ),
  m_scene_epsilon     ( 1e-4f ),
  m_frame             ( 0 )
{
}


void MeshViewer::initScene( InitialCameraData& camera_data )
{
  initContext();
  initLights();
  initMaterial();
  initGeometry();
  initCamera( camera_data );
  preprocess();
}


void MeshViewer::initContext()
{
  m_context->setRayTypeCount( 2 );
  m_context->setEntryPointCount( 1 );

  if (m_shade_mode == SM_PHONG || m_shade_mode == SM_ONE_BOUNCE_DIFFUSE) {
    // need more stack for recursive rays.
    m_context->setStackSize( 1200 );
  } else {
    m_context->setStackSize( 640 );
  }

  m_context[ "radiance_ray_type"   ]->setUint( 0u );
  m_context[ "shadow_ray_type"     ]->setUint( 1u );
  m_context[ "max_depth"           ]->setInt( 5 );
  m_context[ "ambient_light_color" ]->setFloat( 0.2f, 0.2f, 0.2f );
  m_context[ "output_buffer"       ]->set( createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, WIDTH, HEIGHT) );
  
  // Ray generation program setup
  const std::string camera_name = m_camera_mode == CM_PINHOLE ? "pinhole_camera" : "orthographic_camera"; 
  const std::string camera_file = m_shade_mode  == SM_AO                 ? "accum_camera.cu" :
                                  m_shade_mode  == SM_ONE_BOUNCE_DIFFUSE ? "accum_camera.cu"  :
                                  m_camera_mode == CM_PINHOLE            ? "pinhole_camera.cu"  :
                                                                          "orthographic_camera.cu";

  if( m_shade_mode == SM_AO || m_shade_mode == SM_ONE_BOUNCE_DIFFUSE ) {
    // The raygen program needs accum_buffer
    m_accum_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT4,
                                            WIDTH, HEIGHT );
    m_context["accum_buffer"]->set( m_accum_buffer );
    resetAccumulation();

  }

  const std::string camera_ptx  = ptxpath( "displacement", camera_file );
  Program ray_gen_program = m_context->createProgramFromPTXFile( camera_ptx, camera_name );
  m_context->setRayGenerationProgram( 0, ray_gen_program );


  // Exception program
  const std::string except_ptx  = ptxpath( "displacement", camera_file );
  m_context->setExceptionProgram( 0, m_context->createProgramFromPTXFile( except_ptx, "exception" ) );
  m_context[ "bad_color" ]->setFloat( 0.0f, 1.0f, 0.0f );


  // Miss program 
  const std::string miss_ptx = ptxpath( "displacement", "constantbg.cu" );
  m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( miss_ptx, "miss" ) );
  m_context[ "bg_color" ]->setFloat(  0.34f, 0.55f, 0.85f );
}


void MeshViewer::initLights()
{
  // Lights buffer
  BasicLight lights[] = {
    { make_float3( -60.0f,  30.0f, -120.0f ), make_float3( 0.2f, 0.2f, 0.25f )*m_light_scale, 0 },
    { make_float3( -60.0f,   0.0f,  120.0f ), make_float3( 0.1f, 0.1f, 0.10f )*m_light_scale, 0 },
    { make_float3(  60.0f,  60.0f,   60.0f ), make_float3( 0.7f, 0.7f, 0.65f )*m_light_scale, 1 }
  };

  Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
  light_buffer->setFormat(RT_FORMAT_USER);
  light_buffer->setElementSize(sizeof( BasicLight ) );
  light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
  memcpy(light_buffer->map(), lights, sizeof(lights));
  light_buffer->unmap();

  m_context[ "lights" ]->set( light_buffer );
}


void MeshViewer::initMaterial()
{
  switch( m_shade_mode ) {
    case SM_PHONG: {
      // Use the default obj_material created by ObjLoader
      break;
    }

    case SM_NORMAL: {
      const std::string ptx_path = ptxpath("displacement", "normal_shader.cu");
      m_material = m_context->createMaterial();
      m_material->setClosestHitProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "closest_hit_radiance" ) );
      break;
    }

    case SM_AO: {
      const std::string ptx_path = ptxpath("displacement", "ambocc.cu");
      m_material = m_context->createMaterial();
      m_material->setClosestHitProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "closest_hit_radiance" ) );
      m_material->setAnyHitProgram    ( 1, m_context->createProgramFromPTXFile( ptx_path, "any_hit_occlusion" ) );    
      genRndSeeds( WIDTH, HEIGHT );
      break;
    } 
    
    case SM_ONE_BOUNCE_DIFFUSE: {
      const std::string ptx_path = ptxpath("displacement", "one_bounce_diffuse.cu");
      m_material = m_context->createMaterial();
      m_material->setClosestHitProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "closest_hit_radiance" ) );
      m_material->setAnyHitProgram    ( 1, m_context->createProgramFromPTXFile( ptx_path, "any_hit_shadow" ) );
      genRndSeeds( WIDTH, HEIGHT );
      break;
    }
  }
}


void MeshViewer::initGeometry()
{
  m_geometry_group = m_context->createGeometryGroup();

  if( ObjLoader::isMyFile( m_filename.c_str() ) ) {
    // Load OBJ model 
    ObjLoader* loader = 0;
    if( m_shade_mode == SM_NORMAL || m_shade_mode == SM_AO ) {
      loader = new ObjLoader( m_filename.c_str(), m_context, m_geometry_group, m_material );
    } else if ( m_shade_mode == SM_ONE_BOUNCE_DIFFUSE ) {
      loader = new ObjLoader( m_filename.c_str(), m_context, m_geometry_group, m_material, true );
    } else {
      loader = new ObjLoader( m_filename.c_str(), m_context, m_geometry_group );
    }

    const std::string geom_ptx = ptxpath( "displacement", "geometry_programs.cu" );
    loader->setIntersectProgram(m_context->createProgramFromPTXFile( geom_ptx, "mesh_intersect" ) );
    loader->setBboxProgram(m_context->createProgramFromPTXFile( geom_ptx, "mesh_bounds" ) );

    loader->load();
    m_aabb = loader->getSceneBBox();
    delete loader;

  } else if( PlyLoader::isMyFile( m_filename ) ) {
    // Load PLY model 
    PlyLoader loader( m_filename, m_context, m_geometry_group, m_material );
    loader.load();

    m_aabb = loader.getSceneBBox();

  } else {
    std::cerr << "Unrecognized model file extension '" << m_filename << "'" << std::endl;
    exit( 0 );
  }

  // Override acceleration structure builder. The default used by the ObjLoader is Sbvh.
  if( !m_accel_builder.empty() ) {
    Acceleration accel = m_geometry_group->getAcceleration();
    accel->setBuilder( m_accel_builder );
  }

  Acceleration accel = m_geometry_group->getAcceleration();
  accel->setBuilder("Bvh");

  // Override traverer if one is given.
  if( !m_accel_traverser.empty() ) {
    Acceleration accel = m_geometry_group->getAcceleration();
    accel->setTraverser( m_accel_traverser );
  }

  if( m_accel_builder == "TriangleKdTree" || m_accel_traverser == "KdTree") {
    Acceleration accel = m_geometry_group->getAcceleration();
    accel->setProperty( "vertex_buffer_name", "vertex_buffer" );
    accel->setProperty( "index_buffer_name", "vindex_buffer" );
  }
  
  // Load acceleration structure from a file if that was enabled on the
  // command line, and if we can find a cache file. Note that the type of
  // acceleration used will be overridden by what is found in the file.
  loadAccelCache();

  
  m_context[ "top_object" ]->set( m_geometry_group );
  m_context[ "top_shadower" ]->set( m_geometry_group );

}


void MeshViewer::initCamera( InitialCameraData& camera_data )
{
  // Set up camera
  float3 eye     = m_aabb.m_min;
  float3 size    = m_aabb.m_max - m_aabb.m_min;
  eye           += size * make_float3(0.25f, 4.f, 1.f);

  camera_data = InitialCameraData( eye,                             // eye
                                   m_aabb.center(),                  // lookat
                                   make_float3( 0.0f, 0.0f, 1.0f ), // up
                                   30.0f );                         // vfov

  // Declare camera variables.  The values do not matter, they will be overwritten in trace.
  m_context[ "eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context[ "U"  ]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context[ "V"  ]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context[ "W"  ]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

}


void MeshViewer::preprocess()
{
  // Settings which rely on previous initialization
  m_scene_epsilon = 1.e-4f * m_aabb.maxExtent();
  m_context[ "scene_epsilon"      ]->setFloat( m_scene_epsilon );
  m_context[ "occlusion_distance" ]->setFloat( m_aabb.maxExtent() * 0.3f * m_ao_radius );

  // Prepare to run 
  m_context->validate();
  double start, end_compile, end_AS_build;
  sutilCurrentTime(&start);
  m_context->compile();
  sutilCurrentTime(&end_compile);
  std::cerr << "Time to compile kernel: "<<end_compile-start<<" s.\n";
  m_context->launch(0,0);
  sutilCurrentTime(&end_AS_build);
  std::cerr << "Time to build AS      : "<<end_AS_build-end_compile<<" s.\n";
}


bool MeshViewer::keyPressed(unsigned char key, int x, int y)
{
   switch (key)
   {
     case 'e':
       m_scene_epsilon *= .1f;
       std::cerr << "scene_epsilon: " << m_scene_epsilon << std::endl;
       m_context[ "scene_epsilon" ]->setFloat( m_scene_epsilon );
       return true;
     case 'E':
       m_scene_epsilon *= 10.0f;
       std::cerr << "scene_epsilon: " << m_scene_epsilon << std::endl;
       m_context[ "scene_epsilon" ]->setFloat( m_scene_epsilon );
       return true;
   }
   return false;
}

          
void MeshViewer::doResize( unsigned int width, unsigned int height )
{
  // output_buffer resizing handled in base class
  if( m_shade_mode == SM_AO || m_shade_mode == SM_ONE_BOUNCE_DIFFUSE ) {
    m_accum_buffer->setSize( width, height );
    m_rnd_seeds->setSize( width, height );
    genRndSeeds( width, height );
    resetAccumulation();
  }
}


void MeshViewer::trace( const RayGenCameraData& camera_data )
{
  m_context["eye"]->setFloat( camera_data.eye );
  m_context["U"]->setFloat( camera_data.U );
  m_context["V"]->setFloat( camera_data.V );
  m_context["W"]->setFloat( camera_data.W );

  Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );

  if( (m_shade_mode == SM_AO || m_shade_mode == SM_ONE_BOUNCE_DIFFUSE) && !m_camera_changed ) {
    // Use more AO samples if the camera is not moving, for increased !/$.
    m_context["sqrt_occlusion_samples"]->setInt( 2 * m_ao_sample_mult );
    m_context["sqrt_diffuse_samples"]->setInt( 2 );
  }

  m_context->launch( 0, static_cast<unsigned int>(buffer_width), static_cast<unsigned int>(buffer_height) );

  if( m_shade_mode == SM_AO || m_shade_mode == SM_ONE_BOUNCE_DIFFUSE ) {

    // Update frame number for accumulation.
    ++m_frame;
    if( m_camera_changed ) {
      m_camera_changed = false;
      resetAccumulation();
    }

    // The frame number is used as part of the random seed.
    m_context["frame"]->setInt( m_frame );
  }
}


void MeshViewer::cleanUp()
{
  // Store the acceleration cache if required.
  saveAccelCache();
  SampleScene::cleanUp();
}


Buffer MeshViewer::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}


void MeshViewer::resetAccumulation()
{
  m_frame = 0;
  m_context[ "frame"                  ]->setInt( m_frame );
  m_context[ "sqrt_occlusion_samples" ]->setInt( 1 * m_ao_sample_mult );
  m_context[ "sqrt_diffuse_samples"   ]->setInt( 1 );
}


void MeshViewer::genRndSeeds( unsigned int width, unsigned int height )
{
  // Init random number buffer if necessary.
  if( m_rnd_seeds.get() == 0 ) {
    m_rnd_seeds = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_UNSIGNED_INT,
                                         WIDTH, HEIGHT);
    m_context["rnd_seeds"]->setBuffer(m_rnd_seeds);
  }

  unsigned int* seeds = static_cast<unsigned int*>( m_rnd_seeds->map() );
  fillRandBuffer(seeds, width*height);
  m_rnd_seeds->unmap();
}


//-----------------------------------------------------------------------------
//
// Main driver
//
//-----------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                               Print this usage message\n"
    << "  -o  | --obj <obj_file>                     Specify .OBJ model to be rendered\n"
    << "  -c  | --cache                              Turn on acceleration structure caching\n"
    << "  -ph | --phong-shade                        Use standard phong shader\n"    
    << "  -n  | --normal-shade                       Use normal shader\n"
    << "  -i  | --diffuse-shade                      Use one bounce diffuse shader\n"
    << "  -O  | --ortho                              Use orthographic camera (cannot use AO mode with ortho)\n"
    << "  -r  | --ao-radius <scale>                  Scale ambient occlusion radius\n"
    << "  -m  | --ao-sample-mult <n>                 Multiplier for the number of AO samples\n"
    << "  -l  | --light-scale <scale>                Scale lights by constant factor\n"
    << "  -t  | --trav <name>                        Set acceleration structure traverser\n"
    << "        --build <name>                       Set acceleration structure builder\n"
    << std::endl;
  GLUTDisplay::printUsage();

  std::cerr
    << "App keystrokes:\n"
    << "  e Decrease scene epsilon size (used for shadow ray offset)\n"
    << "  E Increase scene epsilon size (used for shadow ray offset)\n"
    << std::endl;

  if ( doExit ) exit(1);
}


int main( int argc, char** argv ) 
{
  GLUTDisplay::init( argc, argv );
  
  // need to use progressive drawing to match our default display
  // mode of ambient occlusion
  GLUTDisplay::contDraw_E draw_mode = GLUTDisplay::CDProgressive; 
  MeshViewer scene;
  scene.setMesh( (std::string( sutilSamplesDir() ) + "/swimmingShark/Fish_OBJ_PPM/Kumanomi.obj").c_str() );

  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "-c" || arg == "--cache" ) {
      scene.setAccelCaching( true );
    } else if( arg == "-n" || arg == "--normal-shade" ) {
      scene.setShadeMode( MeshViewer::SM_NORMAL );
    } else if( arg == "-ph" || arg == "--phong-shade" ) {
      scene.setShadeMode( MeshViewer::SM_PHONG );
      draw_mode = GLUTDisplay::CDNone;
    } else if( arg == "-i" || arg == "--diffuse-shade" ) {
      scene.setShadeMode( MeshViewer::SM_ONE_BOUNCE_DIFFUSE );
      draw_mode = GLUTDisplay::CDProgressive;
    } else if( arg == "-O" || arg == "--ortho" ) {
      scene.setCameraMode( MeshViewer::CM_ORTHO );
    } else if( arg == "-h" || arg == "--help" ) {
      printUsageAndExit( argv[0] ); 
    } else if( arg == "-o" || arg == "--obj" ) {
      if ( i == argc-1 ) {
        printUsageAndExit( argv[0] );
      }
      scene.setMesh( argv[++i] );
    } else if( arg == "-t" || arg == "--trav" ) {
      if ( i == argc-1 ) {
        printUsageAndExit( argv[0] );
      }
      scene.setTraverser( argv[++i] );
    } else if( arg == "--build" ) {
      if ( i == argc-1 ) {
        printUsageAndExit( argv[0] );
      }
      scene.setBuilder( argv[++i] );
    } else if( arg == "--kd" ) {     // Keep this arg for a while for backward compatibility
      scene.setBuilder( "TriangleKdTree" );
      scene.setTraverser( "KdTree" );
    } else if( arg == "--lbvh" ) {   // Keep this arg for a while for backward compatibility
      scene.setBuilder( "Lbvh" );
    } else if( arg == "--bvh" ) {    // Keep this arg for a while for backward compatibility
      scene.setBuilder( "Bvh" );
    } else if( arg == "-r" || arg == "--ao-radius" ) {
      if ( i == argc-1 ) {
        printUsageAndExit( argv[0] );
      }
      scene.setAORadius( static_cast<float>( atof( argv[++i] ) ) );
    } else if( arg == "-m" || arg == "--ao-sample-mult" ) {
      if ( i == argc-1 ) {
        printUsageAndExit( argv[0] );
      }
      scene.setAOSampleMultiplier( atoi( argv[++i] ) );
    } else if( arg == "-l" || arg == "--light-scale" ) {
      if ( i == argc-1 ) {
        printUsageAndExit( argv[0] );
      }
      scene.setLightScale( static_cast<float>( atof( argv[++i] ) ) );
    } else {
      std::cerr << "Unknown option: '" << arg << "'" << std::endl;
      printUsageAndExit( argv[0] );
    }
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

  try {
    GLUTDisplay::run( "Displacement", &scene, draw_mode );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }

  return 0;
}

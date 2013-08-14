
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
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
//  transparency.cpp - Scene demonstrating transparent objects 
//
//-----------------------------------------------------------------------------

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <GLUTDisplay.h>
#include "commonStructs.h"
#include <ImageLoader.h>

#include <iostream>
#include <cstdlib>
#include <cstring>

using namespace optix;

//-----------------------------------------------------------------------------
// 
// TransparencyScene 
//
//-----------------------------------------------------------------------------

class TransparencyScene : public SampleScene
{
public:
  // From SampleScene
  virtual void   initScene( InitialCameraData& camera_data );
  virtual void   trace( const RayGenCameraData& camera_data );
  virtual Buffer getOutputBuffer();

  void createGeometry();

  static bool m_useGLBuffer;
private:
  static unsigned int WIDTH;
  static unsigned int HEIGHT;
};

unsigned int TransparencyScene::WIDTH  = 1024u;
unsigned int TransparencyScene::HEIGHT = 768u;

// make PBOs default since it's a lot of faster
bool         TransparencyScene::m_useGLBuffer = true;


void TransparencyScene::initScene( InitialCameraData& camera_data )
{
  try {
    // Setup state
    m_context->setRayTypeCount( 2 );
    m_context->setEntryPointCount( 1 );
    m_context->setStackSize( 1520 );

    m_context["max_depth"]->setInt( 6 );
    m_context["radiance_ray_type"]->setUint( 0u );
    m_context["shadow_ray_type"]->setUint( 1u );
    m_context["scene_epsilon"]->setFloat( 1.e-3f );

    Variable output_buffer = m_context["output_buffer"];

    output_buffer->set(createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, WIDTH, HEIGHT ) );

    // Set up camera
    camera_data = InitialCameraData( make_float3( 8.3f, 4.0f, -4.8f ), // eye
                                     make_float3( 0.5f, 0.3f,  1.0f ), // lookat
                                     make_float3( 0.0f, 1.0f,  0.0f ), // up
                                     60.0f );                          // vfov

    // Declare camera variables.  The values do not matter, they will be overwritten in trace.
    m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

    // Ray gen program
    std::string ptx_path( ptxpath( "transparency", "pinhole_camera.cu" ) );
    Program ray_gen_program = m_context->createProgramFromPTXFile( ptx_path, "pinhole_camera" );
    m_context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception
    Program exception_program = m_context->createProgramFromPTXFile( ptx_path, "exception" );
    m_context->setExceptionProgram( 0, exception_program );
    m_context["bad_color"]->setFloat( 0.0f, 1.0f, 0.0f );

    // Miss program
    m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptxpath( "transparency", "constantbg.cu" ), "miss" ) );
    m_context["bg_color"]->setFloat( make_float3(108.0f/255.0f, 166.0f/255.0f, 205.0f/255.0f) * 0.5f );

    // Setup lights
    m_context["ambient_light_color"]->setFloat(0,0,0);
    BasicLight lights[] = { 
      { { -7.0f, 15.0f, -7.0f }, { .8f, .8f, .8f }, 1 }
    };

    Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
    light_buffer->setFormat(RT_FORMAT_USER);
    light_buffer->setElementSize(sizeof(BasicLight));
    light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
    memcpy(light_buffer->map(), lights, sizeof(lights));
    light_buffer->unmap();

    m_context["lights"]->set(light_buffer);

    // Create scene geom
    createGeometry();

    // Finalize
    m_context->validate();
    m_context->compile();

  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }
}


Buffer TransparencyScene::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}


void TransparencyScene::trace( const RayGenCameraData& camera_data )
{
  m_context["eye"]->setFloat( camera_data.eye );
  m_context["U"]->setFloat( camera_data.U );
  m_context["V"]->setFloat( camera_data.V );
  m_context["W"]->setFloat( camera_data.W );

  Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );

  m_context->launch( 0, 
                    static_cast<unsigned int>(buffer_width),
                    static_cast<unsigned int>(buffer_height)
                    );
}

void TransparencyScene::createGeometry()
{
  // Material programs
  Program transparent_ch = m_context->createProgramFromPTXFile( ptxpath( "transparency", "transparent.cu" ), "closest_hit_radiance" );
  Program transparent_ah = m_context->createProgramFromPTXFile( ptxpath( "transparency", "transparent.cu" ), "any_hit_shadow" );

  // Box programs
  Program box_bounds    = m_context->createProgramFromPTXFile( ptxpath( "transparency", "box.cu" ), "box_bounds" );
  Program box_intersect = m_context->createProgramFromPTXFile( ptxpath( "transparency", "box.cu" ), "box_intersect" );

  // Boxes
  const int SQRT_NUM_BOXES = 3;
  const float grid_base = -static_cast<float>( SQRT_NUM_BOXES ) * 1.5f + 1.0f;
  std::vector<Geometry> boxes;
  std::vector<Material> box_matls;
  for( int i = 0; i < SQRT_NUM_BOXES; ++i ) {
    for( int j = 0; j < SQRT_NUM_BOXES; ++j ) {

      // Geometry
      const float minx = grid_base + i*3;
      const float minz = grid_base + j*3; 

      Geometry box = m_context->createGeometry();
      boxes.push_back( box );
      
      box->setPrimitiveCount( 1u );
      box->setBoundingBoxProgram( box_bounds );
      box->setIntersectionProgram( box_intersect );
      box["boxmin"]->setFloat( minx,         1.5f, minz         );
      box["boxmax"]->setFloat( minx + 1.0f,  2.5f, minz + 1.0f  );

      // Material
      Material box_matl = m_context->createMaterial();
      box_matls.push_back( box_matl );
      box_matl->setClosestHitProgram( 0, transparent_ch );
      box_matl->setAnyHitProgram( 1, transparent_ah );

      box_matl["Kd"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
      box_matl["Ks"]->setFloat( make_float3( 0.2f, 0.2f, 0.2f ) );
      box_matl["Ka"]->setFloat( make_float3( 0.05f, 0.05f, 0.05f ) );
      box_matl["phong_exp"]->setFloat(32);
      box_matl["refraction_index"]->setFloat( 1.2f );

      float color_scale = static_cast<float>( i*SQRT_NUM_BOXES+j )/ static_cast<float>( SQRT_NUM_BOXES*SQRT_NUM_BOXES );
      float3 Kd = make_float3( color_scale, 1.0f - color_scale, 1.0f );
      box_matl["transmissive_map"]->setTextureSampler( loadTexture( m_context, "", Kd ) );
    }
  }

  // Sphere
  Geometry sphere = m_context->createGeometry();
  sphere->setPrimitiveCount( 1u );
  sphere->setBoundingBoxProgram( m_context->createProgramFromPTXFile( ptxpath( "transparency", "sphere_texcoord.cu" ), "bounds" ) );
  sphere->setIntersectionProgram( m_context->createProgramFromPTXFile( ptxpath( "transparency", "sphere_texcoord.cu" ), "intersect" ) );
  sphere["sphere"]->setFloat( 0.0f, 3.5f, 0.0f, 1.0f );
  sphere["matrix_row_0"]->setFloat( 1.0f, 0.0f, 0.0f );
  sphere["matrix_row_1"]->setFloat( 0.0f, 1.0f, 0.0f );
  sphere["matrix_row_2"]->setFloat( 0.0f, 0.0f, 1.0f );

  Material sphere_matl = m_context->createMaterial();
  sphere_matl->setClosestHitProgram( 0, transparent_ch );
  sphere_matl->setAnyHitProgram( 1, transparent_ah );

  sphere_matl["Kd"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  sphere_matl["Ks"]->setFloat( make_float3( 0.3f, 0.3f, 0.3f ) );
  sphere_matl["Ka"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  std::string texture_path = std::string( sutilSamplesDir() ) + "/transparency/";
  sphere_matl["transmissive_map"]->setTextureSampler( loadTexture( m_context, texture_path + "sphere_texture.ppm",
                                                                   make_float3( 0.0f, 0.0f, 0.0f ) ) );
  sphere_matl["phong_exp"]->setFloat(64);
  sphere_matl["refraction_index"]->setFloat( 1.0f );

  // Floor
  Geometry parallelogram = m_context->createGeometry();
  parallelogram->setPrimitiveCount( 1u );

  std::string ptx_path( ptxpath( "transparency", "parallelogram.cu" ) );
  parallelogram->setBoundingBoxProgram( m_context->createProgramFromPTXFile( ptx_path, "bounds" ) );
  parallelogram->setIntersectionProgram( m_context->createProgramFromPTXFile( ptx_path, "intersect" ) );
  float3 anchor = make_float3( -20.0f, 0.01f, 20.0f);
  float3 v1 = make_float3( 40, 0, 0);
  float3 v2 = make_float3( 0, 0, -40);
  float3 normal = cross( v1, v2 );
  normal = normalize( normal );
  float d = dot( normal, anchor );
  v1 *= 1.0f/dot( v1, v1 );
  v2 *= 1.0f/dot( v2, v2 );
  float4 plane = make_float4( normal, d );
  parallelogram["plane"]->setFloat( plane );
  parallelogram["v1"]->setFloat( v1 );
  parallelogram["v2"]->setFloat( v2 );
  parallelogram["anchor"]->setFloat( anchor );

  // Floor material
  Material floor_matl = m_context->createMaterial();
  floor_matl->setClosestHitProgram( 0, transparent_ch );
  floor_matl->setAnyHitProgram( 1, transparent_ah );

  floor_matl["Kd"]->setFloat( make_float3( 0.7f, 0.7f, 0.7f ) );
  floor_matl["Ks"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  floor_matl["Ka"]->setFloat( make_float3( 0.05f, 0.05f, 0.05f ) );
  floor_matl["transmissive_map"]->setTextureSampler( loadTexture( m_context, "", make_float3( 0.0f, 0.0f, 0.0f ) ) );
  floor_matl["phong_exp"]->setFloat(32);
  floor_matl["refraction_index"]->setFloat( 1.0f );

  // Place geometry into hierarchy
  std::vector<GeometryInstance> gis;
  for( unsigned int i = 0; i < boxes.size(); ++i ) {
    GeometryInstance gi = m_context->createGeometryInstance(); 
    gi->setGeometry( boxes[i] );
    gi->setMaterialCount( 1 );
    gi->setMaterial( 0, box_matls[i] );
    gis.push_back( gi );
  }
  gis.push_back( m_context->createGeometryInstance( parallelogram, &floor_matl, &floor_matl+1 ) );
  gis.push_back( m_context->createGeometryInstance( sphere, &sphere_matl, &sphere_matl+1 ) );

  GeometryGroup geometrygroup = m_context->createGeometryGroup();
  geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()) );
  for ( unsigned int i = 0; i < gis.size(); ++i ) { 
    geometrygroup->setChild( i, gis[i] );
  }
  geometrygroup->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );

  m_context["top_object"]->set( geometrygroup );
  m_context["top_shadower"]->set( geometrygroup );
}

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                               Print this usage message\n"
    << "  -P  | --pbo                                Use OpenGL PBO for output buffer\n"
    << "  -n  | --nopbo                              Use OptiX internal output buffer (Default)\n"
    << std::endl;
  GLUTDisplay::printUsage();

  if ( doExit ) exit(1);
}

int main( int argc, char** argv )
{
  GLUTDisplay::init( argc, argv );

  bool use_vbo_buffer = true;
  for(int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if(arg == "-P" || arg == "--pbo") {
      use_vbo_buffer = true;
    } else if( arg == "-n" || arg == "--nopbo" ) {
      use_vbo_buffer = false;
    } else if( arg == "-h" || arg == "--help" ) {
      printUsageAndExit(argv[0]);
    } else {
      std::cerr << "Unknown option '" << arg << "'\n";
      printUsageAndExit(argv[0]);
    }
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit(argv[0], false);

  TransparencyScene scene;
  scene.setUseVBOBuffer( use_vbo_buffer );
  GLUTDisplay::run( "TransparencyScene", &scene );
  return 0;
}

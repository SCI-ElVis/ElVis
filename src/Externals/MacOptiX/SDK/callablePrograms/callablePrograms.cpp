
/*
 * Copyright (c) 2012 NVIDIA Corporation.  All rights reserved.
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
//  callablePrograms.cpp - Show all sorts of function variable usage
//
//-----------------------------------------------------------------------------

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <GLUTDisplay.h>
#include "commonStructs.h"
#include <iostream>
#include <cstdlib>
#include <cstring>

using namespace optix;

//-----------------------------------------------------------------------------
// 
// Manta Scene
//
//-----------------------------------------------------------------------------

class callablePrograms : public SampleScene
{
public:
  // From SampleScene
  virtual void   initScene( InitialCameraData& camera_data );
  virtual void   trace( const RayGenCameraData& camera_data );
  virtual Buffer getOutputBuffer();
  virtual bool   keyPressed(unsigned char key, int x, int y);

  void createGeometry();

  static bool m_useGLBuffer;
private:

  static unsigned int WIDTH;
  static unsigned int HEIGHT;

  std::vector<Program> m_colorPrograms;
  Program m_missProgram;
  Program m_rayGenProgram;
};

unsigned int callablePrograms::WIDTH  = 1024u;
unsigned int callablePrograms::HEIGHT = 1024u;


bool callablePrograms::keyPressed(unsigned char key, int x, int y)
{
  Program randProg = m_colorPrograms[rand() % m_colorPrograms.size()];

  switch(key) {
  case 'r':
    m_rayGenProgram["modColor"]->set(randProg);
    break;
  case 'm':
    m_missProgram["modColor"]->set(randProg);
    break;
  }

  return false;
}

void callablePrograms::initScene( InitialCameraData& camera_data )
{
  try {
    // Setup state
    m_context->setRayTypeCount( 1 );
    m_context->setEntryPointCount( 1 );
    m_context->setStackSize(1200);

    m_context["max_depth"]->setInt( 5 );
    m_context["radiance_ray_type"]->setUint( 0u );
    m_context["scene_epsilon"]->setFloat( 1.e-4f );

    Variable output_buffer = m_context["output_buffer"];

    output_buffer->set(createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, WIDTH, HEIGHT ) );

    // Set up camera
    camera_data = InitialCameraData( make_float3( 3.0f, 2.0f, -3.0f ), // eye
                                     make_float3( 0.0f, 0.3f,  0.0f ), // lookat
                                     make_float3( 0.0f, 1.0f,  0.0f ), // up
                                     60.0f );                          // vfov

    // Declare camera variables.  The values do not matter, they will be overwritten in trace.
    m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

    // Ray gen program
    std::string ptx_path( ptxpath( TARGET_NAME, "camera.cu" ) );
    m_rayGenProgram = m_context->createProgramFromPTXFile( ptx_path, "my_pinhole_camera" );
    m_context->setRayGenerationProgram( 0, m_rayGenProgram );
    m_context["draw_color"]->setFloat( 0.462f, 0.725f, 0.0f );


    // Exception program
    Program exception_program = m_context->createProgramFromPTXFile( ptx_path, "exception" );
    m_context->setExceptionProgram( 0, exception_program );
    m_context["bad_color"]->setFloat( 1.0f, 1.0f, 0.0f );

    // Callable programs
    std::string color_mods_path( ptxpath( TARGET_NAME, "color_mods.cu" ) );
    Program scale_color = m_context->createProgramFromPTXFile( color_mods_path, "scale_color" );
    Program checker_color = m_context->createProgramFromPTXFile( color_mods_path, "checker_color" );
    Program return_same_color = m_context->createProgramFromPTXFile( color_mods_path, "return_same_color" );
    Program wavey_color = m_context->createProgramFromPTXFile( color_mods_path, "wavey_color" );

    checker_color["scale"]->setFloat(0.75f);
    //m_context["modColor"]->set(scale_color);
    m_rayGenProgram["modColor"]->set(checker_color);
    //m_context["modColor"]->set(return_same_color);

    m_colorPrograms.push_back(scale_color);
    m_colorPrograms.push_back(checker_color);
    m_colorPrograms.push_back(return_same_color);
    m_colorPrograms.push_back(wavey_color);

    // Miss program
    m_missProgram = m_context->createProgramFromPTXFile( ptxpath( TARGET_NAME, "camera.cu" ), "miss" );
    m_context->setMissProgram( 0, m_missProgram );
    wavey_color["scale"]->setFloat(0.5f);
    m_missProgram["modColor"]->set(wavey_color);
    //m_context["bg_color"]->setFloat( make_float3(108.0f/255.0f, 166.0f/255.0f, 205.0f/255.0f) * 0.5f );

    // Setup lights
    m_context["ambient_light_color"]->setFloat(0,0,0);
    BasicLight lights[] = { 
      { { 0.0f, 8.0f, -5.0f }, { .6f, .1f, .1f }, 1 },
      { { 5.0f, 8.0f,  0.0f }, { .1f, .6f, .1f }, 1 },
      { { 5.0f, 2.0f, -5.0f }, { .2f, .2f, .2f }, 1 }
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


Buffer callablePrograms::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}


void callablePrograms::trace( const RayGenCameraData& camera_data )
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

void callablePrograms::createGeometry()
{
  // Sphere
  Geometry sphere = m_context->createGeometry();
  sphere->setPrimitiveCount( 1u );
  sphere->setBoundingBoxProgram( m_context->createProgramFromPTXFile( ptxpath( TARGET_NAME, "sphere.cu" ), "bounds" ) );
  sphere->setIntersectionProgram( m_context->createProgramFromPTXFile( ptxpath( TARGET_NAME, "sphere.cu" ), "robust_intersect" ) );
  sphere["sphere"]->setFloat( 0.0f, 1.2f, 0.0f, 1.0f );

  Program dot_product = m_context->createProgramFromPTXFile( ptxpath( TARGET_NAME, "materials.cu" ), "dot_product" );

  Material sphere_matl = m_context->createMaterial();
  sphere_matl->setClosestHitProgram( 0, dot_product );

  // Floor
  Geometry parallelogram = m_context->createGeometry();
  parallelogram->setPrimitiveCount( 1u );

  std::string ptx_path( ptxpath( TARGET_NAME, "parallelogram-programmable-normal.cu" ) );
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

  Program sine_adjust = m_context->createProgramFromPTXFile( ptxpath( TARGET_NAME, "perturb-normal.cu" ), "sine_adjust" );
  parallelogram["x_frequency"]->setFloat( 10.f );
  parallelogram["z_frequency"]->setFloat( 10.f );
  parallelogram["amplitude"]->setFloat( 0.2f );
  parallelogram["perturb_normal"]->set( sine_adjust );

  Material floor_matl = m_context->createMaterial();
  floor_matl->setClosestHitProgram( 0, dot_product );

  // Place geometry into hierarchy
  std::vector<GeometryInstance> gis;
  gis.push_back( m_context->createGeometryInstance( sphere,        &sphere_matl, &sphere_matl+1 ) );
  gis.push_back( m_context->createGeometryInstance( parallelogram, &floor_matl, &floor_matl+1 ) );

  GeometryGroup geometrygroup = m_context->createGeometryGroup(gis.begin(), gis.end());
  geometrygroup->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );
  
  m_context["top_object"]->set( geometrygroup );
}

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                               Print this usage message\n"
    << std::endl;
  GLUTDisplay::printUsage();

  std::cerr
    << "App keystrokes:\n"
    << "  r Assign a random RT_CALLABLE_PROGRAM to the ray generation program's modColor program variable\n"
    << "  m Assign a random RT_CALLABLE_PROGRAM to the miss program's modColor program variable\n"
    << std::endl;

  if ( doExit ) exit(1);
}

int main( int argc, char** argv )
{
  GLUTDisplay::init( argc, argv );

  for(int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (false) {
    } else if( arg == "-h" || arg == "--help" ) {
      printUsageAndExit( argv[0] );
    } else {
      std::cerr << "Unknown option '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

  callablePrograms scene;
  GLUTDisplay::run( "callablePrograms", &scene );
  return 0;
}

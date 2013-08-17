
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
//  shade_trees.cpp - Example of callable program usage.
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


static float3 make_contrast_color( int tag )
{
  static const unsigned char s_Colors[16][3] =
  {
    {  34, 139,  34}, // ForestGreen
    { 210, 180, 140}, // Tan
    { 250, 128, 114}, // Salmon
    { 173, 255,  47}, // GreenYellow
    { 255,   0, 255}, // Magenta
    { 255,   0,   0}, // Red
    {   0, 250, 154}, // MediumSpringGreen
    { 255, 165,   0}, // Orange
    { 240, 230, 140}, // Khaki
    { 255, 215,   0}, // Gold
    { 178,  34,  34}, // Firebrick
    { 154, 205,  50}, // YellowGreen
    {  64, 224, 208}, // Turquoise
    {   0,   0, 255}, // Blue
    { 100, 149, 237}, // CornflowerBlue
    { 153, 153, 255}, // (bright blue)
  };
  int i = tag & 0x0f;
  float3 color = make_float3( s_Colors[i][0], s_Colors[i][1], s_Colors[i][2] );
  color *= 1.f/255.f;
  i = (tag >> 4) & 0x3;
  color *= 1.f - float(i) * 0.23f;
  return color;
}        


class ShadeTreeScene : public SampleScene
{
public:

  explicit ShadeTreeScene( size_t num_lights );

  // From SampleScene
  virtual void   initScene( InitialCameraData& camera_data );
  virtual void   trace( const RayGenCameraData& camera_data );
  virtual Buffer getOutputBuffer();

  void createGeometry();

private:
  static unsigned int WIDTH;
  static unsigned int HEIGHT;

  void addShadeTree( int kind, float3 color_a, float3 color_b );
  BasicLight makeLight( const float bright_scale );
  void moveLights( BasicLight* lights );

  std::vector<Program> m_colorPrograms;
  std::vector<Program> m_normalPrograms;
  std::vector<int>     m_normalProgId;    // for each color program, index of normal program
  Program m_missProgram;
  Program m_rayGenProgram;
  Buffer m_light_buffer;

  size_t m_num_lights;
};

unsigned int ShadeTreeScene::WIDTH  = 1024u;
unsigned int ShadeTreeScene::HEIGHT = 1024u;

ShadeTreeScene::ShadeTreeScene(size_t num_lights) : m_num_lights( num_lights )
{
}

void ShadeTreeScene::addShadeTree( int kind, float3 color_a, float3 color_b )
{
  const std::string shader_func_path( ptxpath(TARGET_NAME, "shader_functions.cu") );

  if (m_normalPrograms.empty()) {
    // First normal program is passthrough, as default.
    Program pass = m_context->createProgramFromPTXFile( shader_func_path, "float3_passthrough" );
    m_normalPrograms.push_back( pass );
  }

  Program shade_tree;
  int normal_program_id = 0;

  Program constant_a = m_context->createProgramFromPTXFile( shader_func_path, "color_constant" );
  Program constant_b = m_context->createProgramFromPTXFile( shader_func_path, "color_constant" );
  constant_a["color"]->setFloat( color_a );
  constant_b["color"]->setFloat( color_b );
  switch (kind) {

  case 0: // flat color
    {
      Program pass = m_context->createProgramFromPTXFile( shader_func_path, "color_passthrough" );
      pass["color_input"]->set( constant_b );
      shade_tree = pass;
    }
    break;

  case 1: // checker
    {
      Program checker = m_context->createProgramFromPTXFile( shader_func_path, "scalar_checker" );
      checker["scalar_checker_count"]->setFloat( 7.f );
      Program blend = m_context->createProgramFromPTXFile( shader_func_path, "color_blend" );
      blend["color_blend_a"]->set( constant_a );
      blend["color_blend_b"]->set( constant_b );
      blend["color_blend_factor"]->set( checker );
      shade_tree = blend;
    }
    break;

  case 2: // simplex noise
    {
      Program noise = m_context->createProgramFromPTXFile( shader_func_path, "scalar_snoise" );
      noise["frequency"]->setFloat( 5.f );
      Program blend = m_context->createProgramFromPTXFile( shader_func_path, "color_blend" );
      blend["color_blend_a"]->set( constant_a );
      blend["color_blend_b"]->set( constant_b );
      blend["color_blend_factor"]->set( noise );
      shade_tree = blend;
    }
    break;

  case 3: // simplex turbulence
    {
      Program turb = m_context->createProgramFromPTXFile( shader_func_path, "scalar_sturbulence" );
      turb["octaves"]->setInt( 8 );
      Program blend = m_context->createProgramFromPTXFile( shader_func_path, "color_blend" );
      blend["color_blend_a"]->set( constant_a );
      blend["color_blend_b"]->set( constant_b );
      blend["color_blend_factor"]->set( turb );
      shade_tree = blend;
    }
    break;

  case 4: // checker with noise
    {
      // This does not work, because "color_blend" is attached to another "color_blend".

      Program noise = m_context->createProgramFromPTXFile( shader_func_path, "scalar_snoise" );

      Program blend4 = m_context->createProgramFromPTXFile( shader_func_path, "color_blend" );
      blend4["color_blend_a"]->set( constant_a );
      blend4["color_blend_b"]->set( constant_b );
      blend4["color_blend_factor"]->set( noise );

      Program checker2 = m_context->createProgramFromPTXFile( shader_func_path, "scalar_checker" );
      checker2["scalar_checker_count"]->setFloat( 5.f );

      Program blend = m_context->createProgramFromPTXFile( shader_func_path, "color_blend" );
      blend["color_blend_a"]->set( constant_a );
      blend["color_blend_b"]->set( blend4 );
      blend["color_blend_factor"]->set( checker2 );

      Program pass = m_context->createProgramFromPTXFile( shader_func_path, "color_passthrough" );
      pass["color_input"]->set( blend );

      shade_tree = pass;
    }
    break;

  case 5: //scalar_cellular1
    {
      const float freq = 15.f;
      Program noise = m_context->createProgramFromPTXFile( shader_func_path, "scalar_cellular1" );
      noise["frequency"]->setFloat( freq );
      Program blend = m_context->createProgramFromPTXFile( shader_func_path, "color_blend" );
      blend["color_blend_a"]->set( constant_a );
      blend["color_blend_b"]->set( constant_b );
      blend["color_blend_factor"]->set( noise );
      shade_tree = blend;

      Program bump = m_context->createProgramFromPTXFile( shader_func_path, "bump1" );
      bump["frequency"]->setFloat( freq );
      bump["amplitude"]->setFloat( 1.f );
      m_normalPrograms.push_back( bump );
      normal_program_id = (int)m_normalPrograms.size() - 1;
    }
    break;

  case 6: // bump cellular
    {
      const float freq = 20.f;
      Program pass = m_context->createProgramFromPTXFile( shader_func_path, "color_passthrough" );
      pass["color_input"]->set( constant_b );
      shade_tree = pass;

      Program bump = m_context->createProgramFromPTXFile( shader_func_path, "bump1" );
      bump["frequency"]->setFloat( freq );
      m_normalPrograms.push_back( bump );
      normal_program_id = (int)m_normalPrograms.size() - 1;
    }
    break;


  case 100: // debug color
    {
      Program debug_color = m_context->createProgramFromPTXFile( shader_func_path, "show_uv" );
      shade_tree = debug_color;
    }
    break;

  case 101: // debug cellular
    {
      Program noise = m_context->createProgramFromPTXFile( shader_func_path, "float2_cellular" );
      noise["frequency"]->setFloat( 10.f );
      Program show = m_context->createProgramFromPTXFile( shader_func_path, "show_float2" );
      show["float2_input"]->set( noise );

      shade_tree = show;
    }
    break;

  default: return;
  }

  if (shade_tree ) {
    m_colorPrograms.push_back( shade_tree );
    m_normalProgId.push_back( normal_program_id );
  }
}

namespace {
  inline float FRand() { return float( rand() ) / ( float( RAND_MAX ) ); } // A random number on 0.0 to 1.0.
};

BasicLight ShadeTreeScene::makeLight( const float bright_scale )
{
  BasicLight bl;
  bl.casts_shadow = 1;
  bl.color = make_float3( FRand(), FRand(), FRand() ) * bright_scale;
  bl.pos = make_float3( FRand() * 40.0f - 20.0f, FRand() * 20.0f + 0.0f, FRand() * 40.0f - 20.0f );
  float th = FRand() * 0.1f - 0.05f; // radians of rotation per frame
  bl.padding = float_as_int( th );

  return bl;
}

void ShadeTreeScene::moveLights( BasicLight* lights )
{
  for(size_t l=0; l<m_num_lights; l++) {
    // rotate the light about the vertical axis at a rate of th radians per frame
    float x = lights[l].pos.x;
    float z = lights[l].pos.z;
    float th = int_as_float( lights[l].padding );

    lights[l].pos.x = cosf(th) * x - sinf(th) * z;
    lights[l].pos.z = sinf(th) * x + cosf(th) * z;
  }
}

void ShadeTreeScene::initScene( InitialCameraData& camera_data )
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
    camera_data = InitialCameraData( make_float3(28.878f, 28.8313f, -5.81585f ), // eye
                                     make_float3(3.9701f, -3.0007f, -1.39967f), // lookat
                                     make_float3(-0.490587f, 0.868221f, 0.0742742f), // up
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
    //m_context["environment_color"]->setFloat( 0.462f, 0.725f, 0.0f );

    // Exception program
    Program exception_program = m_context->createProgramFromPTXFile( ptx_path, "exception" );
    m_context->setExceptionProgram( 0, exception_program );
    m_context["bad_color"]->setFloat( 1.0f, 1.0f, 0.0f );

    // Miss program
    m_missProgram = m_context->createProgramFromPTXFile( ptxpath( TARGET_NAME, "camera.cu" ), "miss" );
    m_context->setMissProgram( 0, m_missProgram );

    // Setup lights
    m_context["ambient_light_color"]->setFloat(0.3f, 0.3f, 0.3f);

    m_light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
    m_light_buffer->setFormat(RT_FORMAT_USER);
    m_light_buffer->setElementSize(sizeof(BasicLight));
    m_light_buffer->setSize( m_num_lights );

    BasicLight* lights = reinterpret_cast<BasicLight*>( m_light_buffer->map() );
    for(size_t l=0; l<m_num_lights; l++) {
      lights[l] = makeLight( 2.5f / static_cast<float>(m_num_lights) );
    }
    m_light_buffer->unmap();

    m_context["lights"]->set(m_light_buffer);

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

Buffer ShadeTreeScene::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}

void ShadeTreeScene::trace( const RayGenCameraData& camera_data )
{
  m_context["eye"]->setFloat( camera_data.eye );
  m_context["U"]->setFloat( camera_data.U );
  m_context["V"]->setFloat( camera_data.V );
  m_context["W"]->setFloat( camera_data.W );

  BasicLight* lights = reinterpret_cast<BasicLight*>( m_light_buffer->map() );
  moveLights(lights);
  m_light_buffer->unmap();

  Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );

  m_context->launch( 0, buffer_width, buffer_height );
}

void ShadeTreeScene::createGeometry()
{
  // Sphere array
  const uint2 nspheres = make_uint2(11, 11);
  const float2 spacing = make_float2( 3.0f, 3.0f );
  Geometry sphere = m_context->createGeometry();
  sphere->setPrimitiveCount( nspheres.x * nspheres.y );
  sphere->setBoundingBoxProgram( m_context->createProgramFromPTXFile( ptxpath( TARGET_NAME, "sphere_array.cu" ), "bounds" ) );
  sphere->setIntersectionProgram( m_context->createProgramFromPTXFile( ptxpath( TARGET_NAME, "sphere_array.cu" ), "robust_intersect" ) );
  sphere["sphere_0"]->setFloat( 
    nspheres.x / 2 * -spacing.x,  
    1.2f, 
    nspheres.y / 2 * -spacing.y, 
    1.7f );
  sphere["sphere_count"]->setUint( nspheres );
  sphere["sphere_spacing"]->setFloat( spacing );

  Program sphere_shader = m_context->createProgramFromPTXFile( ptxpath( TARGET_NAME, "materials.cu" ), "shade_tree_material" );

  const int ncolors = 16;
  for( int i=0; i < ncolors; ++i ) {
    float3 color_a = make_contrast_color(i);
    float3 color_b = ( make_float3(1.f) - 0.9f * (color_a * color_a) );
    addShadeTree( 0, color_b, color_a ); // flat color
    addShadeTree( 1, color_a, color_b ); // checker
    addShadeTree( 2, color_a, color_b ); // simplex noise
    addShadeTree( 3, color_a, color_b ); // simplex turbulence
    addShadeTree( 4, color_a, color_b ); // checker simplex turbulence
    addShadeTree( 5, color_a, color_b ); // cellular
    addShadeTree( 6, color_a, color_b ); // bumpcellular

    //addShadeTree( 100, color_a, color_b ); // debug color
    //addShadeTree( 101, color_a, color_b ); // debug cellular
  }
  
  // materials

  uint nmaterials = nspheres.x * nspheres.y;
  if (nmaterials > 1000)  nmaterials = 1000;
  std::vector< Material > sphere_mats( nmaterials );
  for( uint i=0; i < nmaterials; ++i ) {
    sphere_mats[i] = m_context->createMaterial();
    sphere_mats[i]->setClosestHitProgram( 0, sphere_shader );
    const int prog_id = i % m_colorPrograms.size();
    sphere_mats[i]["colorShader"]->set( m_colorPrograms[prog_id]);
    sphere_mats[i]["normalShader"]->set( m_normalPrograms[ m_normalProgId[prog_id] ]);
  }
  sphere["material_count"]->setUint( nmaterials );


  // Floor
  Geometry parallelogram = m_context->createGeometry();
  parallelogram->setPrimitiveCount( 1u );

  std::string ptx_path( ptxpath( TARGET_NAME, "parallelogram.cu" ) );
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

  Material floor_matl;
  {
    Program floor = m_context->createProgramFromPTXFile( ptxpath( TARGET_NAME, "materials.cu" ), "flat_modulated" );

    const std::string shader_func_path( ptxpath(TARGET_NAME, "shader_functions.cu") );

    Program constant_a = m_context->createProgramFromPTXFile( shader_func_path, "color_constant" );
    Program constant_b = m_context->createProgramFromPTXFile( shader_func_path, "color_constant" );
    constant_a["color"]->setFloat( make_float3(0.1f, 0.5f, 0.1f));
    constant_b["color"]->setFloat( make_float3(0.1f, 0.3f, 0.1f));
    Program turb = m_context->createProgramFromPTXFile( shader_func_path, "scalar_sturbulence" );
    turb["octaves"]->setInt( 8 );
    turb["frequency"]->setFloat( 30.f );
    Program blend = m_context->createProgramFromPTXFile( shader_func_path, "color_blend" );
    blend["color_blend_a"]->set( constant_a );
    blend["color_blend_b"]->set( constant_b );
    blend["color_blend_factor"]->set( turb );

    floor_matl = m_context->createMaterial();
    floor_matl->setClosestHitProgram( 0, floor );
    floor_matl["colorShader"]->set( blend );
  }

  // Place geometry into hierarchy
  std::vector<GeometryInstance> gis;
  gis.push_back( m_context->createGeometryInstance( parallelogram, &floor_matl, &floor_matl+1 ) );
  gis.push_back( m_context->createGeometryInstance( sphere,        sphere_mats.begin(), sphere_mats.end() ) );

  GeometryGroup geometrygroup = m_context->createGeometryGroup(gis.begin(), gis.end());
  geometrygroup->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );
  
  m_context["top_object"]->set( geometrygroup );
}



void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                               Print this usage message\n"
    << "        --lights N                           How many lights in the scene\n"
    << std::endl;
  GLUTDisplay::printUsage();


  if ( doExit ) exit(1);
}



int main( int argc, char** argv )
{
  GLUTDisplay::init( argc, argv );
  size_t num_lights = 4;

  for(int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (false) {
    } else if( arg == "-h" || arg == "--help" ) {
      printUsageAndExit( argv[0] );
    } else if( arg == "--seed" ) {
      srand( atoi( argv[++i] ) );
    } else if( arg == "--lights" ) {
      if ( i == argc-1 ) {
        printUsageAndExit( argv[0] );
      }
      num_lights = atoi( argv[++i] );
    } else {
      std::cerr << "Unknown option '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

  ShadeTreeScene scene( num_lights );
  GLUTDisplay::run( "ShadeTreeScene", &scene );
  return 0;
}

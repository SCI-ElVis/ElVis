
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

//-------------------------------------------------------------------------------
//
//  rayDifferentials.cpp -- sample demonstrating mip mapping via ray differentials 
//
//-------------------------------------------------------------------------------

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <iostream>
#include <GLUTDisplay.h>
#include "commonStructs.h"
#include <cstdlib>
#include <cstring>
#include <sstream>
#include "ImageLoader.h"

using namespace optix;

//-----------------------------------------------------------------------------
// 
// RayDifferentials Scene
//
//-----------------------------------------------------------------------------

enum MIPMode {
  MIP_Disable = 0,
  MIP_Enable,
  MIP_FalseColor,

  NumMIPModes
};



class RayDifferentialsScene : public SampleScene
{
public:
  RayDifferentialsScene(const std::string& tex_filename) 
  : SampleScene()
  , m_width(1024u)
  , m_height(720u)
  , m_tex_filename(tex_filename) 
  , m_mip_mode(MIP_Enable)
  , m_tex_scale(1000.f)
  {}

  // From SampleScene
  void   initScene( InitialCameraData& camera_data );
  void   trace( const RayGenCameraData& camera_data );
  void   setDimensions( const unsigned int w, const unsigned int h ) { m_width = w; m_height = h; }
  Buffer getOutputBuffer();

  virtual bool keyPressed(unsigned char key, int x, int y);

private:
  void createGeometry();
  void loadTextures(Material matl);

  unsigned int m_width;
  unsigned int m_height;
  std::string  m_tex_filename;
  MIPMode      m_mip_mode;
  float        m_tex_scale;
};

bool RayDifferentialsScene::keyPressed(unsigned char key, int x, int y)
{
  bool update = false;
  switch(key) {
    case 'l':
    case 'L': {
      if(key == 'l') {
        m_mip_mode = (MIPMode)((m_mip_mode+1)%NumMIPModes);
      } else {
        m_mip_mode = (MIPMode)((NumMIPModes+m_mip_mode-1)%NumMIPModes);
      }
      m_context["mip_mode"]->setInt((int)m_mip_mode);
      update = true;
      std::cout << "MIP mapping: ";
      switch(m_mip_mode) {
      case MIP_Enable:
        std::cout << "ON";
        break;

      case MIP_Disable:
        std::cout << "OFF";
        break;

      case MIP_FalseColor:
        std::cout << "False Colors";
        break;

      default:
        std::cout << "Unknown["<<m_mip_mode<<']';
        break;
      }
      std::cout << '\n';
    } break;

    case 't': {
      m_tex_scale -= .1f;
      m_context["tex_scale"]->setFloat(m_tex_scale, m_tex_scale);
      update = true;
    } break;

    case 'T': {
      m_tex_scale += .1f;
      m_context["tex_scale"]->setFloat(m_tex_scale, m_tex_scale);
      update = true;
    } break;
  }

  return update;
}


void RayDifferentialsScene::initScene( InitialCameraData& camera_data )
{
  // context 
  m_context->setRayTypeCount( 2 );
  m_context->setEntryPointCount( 1 );
  m_context->setStackSize( 2240 );

  m_context["max_depth"]->setInt(5);
  m_context["radiance_ray_type"]->setUint(0);
  m_context["shadow_ray_type"]->setUint(1);
  m_context["frame_number"]->setUint( 0u );
  m_context["scene_epsilon"]->setFloat( 1.e-4f );
  m_context["ambient_light_color"]->setFloat( 0.4f, 0.4f, 0.4f );

  m_context["output_buffer"]->set( createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, m_width, m_height) );

  m_context["mip_mode"]->setInt(m_mip_mode);
  m_context["tex_scale"]->setFloat(m_tex_scale, m_tex_scale);

  // Ray gen program
  std::string rgp_ptx_path = ptxpath( "rayDifferentials", "pinhole_camera_differentials.cu" );
  Program ray_gen_program = m_context->createProgramFromPTXFile( rgp_ptx_path, "pinhole_camera" );
  m_context->setRayGenerationProgram( 0, ray_gen_program );

  // Exception / miss programs
  Program exception_program = m_context->createProgramFromPTXFile( rgp_ptx_path, "exception" );
  m_context->setExceptionProgram( 0, exception_program );
  m_context["bad_color"]->setFloat( 0.0f, 1.0f, 0.0f );
  m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptxpath( "rayDifferentials", "constantbg.cu" ), "miss" ) );
  m_context["bg_color"]->setFloat( make_float3( 0.462f, 0.725f, 0.0f ) );
  //m_context["bg_color"]->setFloat( make_float3( 0.9f, 0.9f, 0.9f ) );

  // Lights
  BasicLight lights[] = { 
    { make_float3( 60.0f, 40.0f, 0.0f ), make_float3( 1.0f, 1.0f, 1.0f ), 1 }
  };

  Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
  light_buffer->setFormat(RT_FORMAT_USER);
  light_buffer->setElementSize(sizeof(BasicLight));
  light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
  memcpy(light_buffer->map(), lights, sizeof(lights));
  light_buffer->unmap();

  m_context["lights"]->set(light_buffer);

  // Set up camera
  camera_data = InitialCameraData( make_float3( 6.0f, 2.3f, 1.0f ), // eye
                                   make_float3( 0.0f, 2.3f, 1.0f ), // lookat
                                   make_float3( 0.0f, 1.0f,  0.0f ), // up
                                   60.0f );                          // vfov

  m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

  // Populate scene hierarchy
  createGeometry();

  // Prepare to run
  m_context->validate();
  m_context->compile();
}


Buffer RayDifferentialsScene::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}


void RayDifferentialsScene::trace( const RayGenCameraData& camera_data )
{
  m_context["eye"]->setFloat( camera_data.eye );
  m_context["U"]->setFloat( camera_data.U );
  m_context["V"]->setFloat( camera_data.V );
  m_context["W"]->setFloat( camera_data.W );

  Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );

  m_context->launch( 0, static_cast<unsigned int>(buffer_width),
                      static_cast<unsigned int>(buffer_height) );
}

void RayDifferentialsScene::loadTextures(Material matl)
{
  TextureSampler tex0 = loadTexture(matl->getContext(), m_tex_filename, make_float3(0.f));
  RTsize width0, height0;
  tex0->getBuffer(0,0)->getSize(width0, height0);

  tex0->setWrapMode( 0, RT_WRAP_REPEAT );
  tex0->setWrapMode( 1, RT_WRAP_REPEAT );

  if(width0 != height0) {
    std::cerr << "Input texture must be square\n";
    exit(EXIT_FAILURE);
  }

  // check dimension is power of two
  if( !((width0 & (width0-1)) == 0) || width0 == 0 ) {
    std::cerr << "Input texture size must be a power of two\n";
    exit(EXIT_FAILURE);
  }

  matl["tex0_dim"]->setInt((int)width0, (int)height0);

  int num_levels = 1;
  int dim = (int)width0;
  matl["tex0"]->set(tex0);
  Buffer previous_tex = tex0->getBuffer(0,0);
  
  TextureSampler samplers[16];
  samplers[0] = tex0;

  do {
    dim >>= 1;

    TextureSampler sampler = matl->getContext()->createTextureSampler();
    sampler->setWrapMode( 0, tex0->getWrapMode(0) );
    sampler->setWrapMode( 1, tex0->getWrapMode(1) );
    sampler->setWrapMode( 2, tex0->getWrapMode(2) );
    sampler->setIndexingMode( tex0->getIndexingMode() );
    sampler->setReadMode( tex0->getReadMode() );
    sampler->setMaxAnisotropy( tex0->getMaxAnisotropy() );
    sampler->setMipLevelCount( 1u );
    sampler->setArraySize( 1u );
    sampler->setFilteringModes( RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE );

    Buffer buffer = matl->getContext()->createBuffer(RT_BUFFER_INPUT);
    buffer->setFormat(RT_FORMAT_UNSIGNED_BYTE4);
    buffer->setSize(dim, dim);

    uchar4* previous_data = (uchar4*)previous_tex->map();
    uchar4* new_data = (uchar4*)buffer->map();
    for(int j = 0; j < dim; ++j) {
      for(int i = 0; i < dim; ++i) {
        
        uchar4 texel0 = previous_data[(i*2+0) + (j*2+0)*(dim*2)];
        uchar4 texel1 = previous_data[(i*2+1) + (j*2+0)*(dim*2)];
        uchar4 texel2 = previous_data[(i*2+0) + (j*2+1)*(dim*2)];
        uchar4 texel3 = previous_data[(i*2+1) + (j*2+1)*(dim*2)];

        float4 ftexel0 = make_float4(texel0.x/255.f, texel0.y/255.f, texel0.z/255.f, texel0.w/255.f);
        float4 ftexel1 = make_float4(texel1.x/255.f, texel1.y/255.f, texel1.z/255.f, texel1.w/255.f);
        float4 ftexel2 = make_float4(texel2.x/255.f, texel2.y/255.f, texel2.z/255.f, texel2.w/255.f);
        float4 ftexel3 = make_float4(texel3.x/255.f, texel3.y/255.f, texel3.z/255.f, texel3.w/255.f);

        float4 fnew_texel = (ftexel0 + ftexel1 + ftexel2 + ftexel3)/make_float4(4.f, 4.f, 4.f, 4.f);

        uchar4 new_texel = make_uchar4( (unsigned char)(fnew_texel.x*255.f),
                                        (unsigned char)(fnew_texel.y*255.f),
                                        (unsigned char)(fnew_texel.z*255.f),
                                        (unsigned char)(fnew_texel.w*255.f) );

        new_data[i + j*dim] = new_texel;
      }
    }
    buffer->unmap();
    previous_tex->unmap();

    sampler->setBuffer(0,0,buffer);
    samplers[num_levels] = sampler;

    num_levels++;
    previous_tex = buffer;
  } while(dim != 1);

  matl["num_mip_levels"]->setInt(num_levels);

  // Get texture IDs for all levels and store them in a buffer so we can use them on the device
  Buffer tex_ids = matl->getContext()->createBuffer(RT_BUFFER_INPUT,RT_FORMAT_INT, num_levels);
  m_context["tex_ids"]->set( tex_ids );
  int *ids = (int*)tex_ids->map();
  for(int l = 0; l < num_levels; ++l)
    ids[l] = samplers[l]->getId();

  tex_ids->unmap();
}

void RayDifferentialsScene::createGeometry()
{
  // Create sphere programs
  std::string sphere_ptx( ptxpath( "rayDifferentials", "sphere_differentials.cu" ) );
  Program sphere_intersect = m_context->createProgramFromPTXFile( sphere_ptx, "robust_intersect" );
  Program sphere_bounds    = m_context->createProgramFromPTXFile( sphere_ptx, "bounds" );

  Geometry sphere = m_context->createGeometry();
  sphere->setPrimitiveCount( 1u );
  sphere->setBoundingBoxProgram( sphere_bounds );
  sphere->setIntersectionProgram( sphere_intersect );

  // Floor geometry
  std::string pgram_ptx( ptxpath( "rayDifferentials", "parallelogram_differentials.cu" ) );
  Geometry parallelogram = m_context->createGeometry();
  parallelogram->setPrimitiveCount( 1u );
  parallelogram->setBoundingBoxProgram( m_context->createProgramFromPTXFile( pgram_ptx, "bounds" ) );
  parallelogram->setIntersectionProgram( m_context->createProgramFromPTXFile( pgram_ptx, "intersect" ) );
  float3 anchor = make_float3( -32000.0f, 0.01f, -32000.0f );
  float3 v1 = make_float3( 64000.0f, 0.0f, 0.0f );
  float3 v2 = make_float3( 0.0f, 0.0f, 64000.0f );
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

  // Glass material
  Program glass_ch = m_context->createProgramFromPTXFile( ptxpath( "rayDifferentials", "glass_mip.cu" ), "closest_hit_radiance" );
  Program glass_ah = m_context->createProgramFromPTXFile( ptxpath( "rayDifferentials", "glass_mip.cu" ), "any_hit_shadow" );
  Material glass_matl = m_context->createMaterial();
  glass_matl->setClosestHitProgram( 0, glass_ch );
  glass_matl->setAnyHitProgram( 1, glass_ah );

  glass_matl["importance_cutoff"]->setFloat( 1e-2f );
  glass_matl["cutoff_color"]->setFloat( 0.34f, 0.55f, 0.85f );
  glass_matl["fresnel_exponent"]->setFloat( 3.0f );
  glass_matl["fresnel_minimum"]->setFloat( 0.1f );
  glass_matl["fresnel_maximum"]->setFloat( 1.0f );
  glass_matl["refraction_index"]->setFloat( 1.4f );
  glass_matl["refraction_color"]->setFloat( 1.0f, 1.0f, 1.0f );
  glass_matl["reflection_color"]->setFloat( 1.0f, 1.0f, 1.0f );
  glass_matl["refraction_maxdepth"]->setInt( 10 );
  glass_matl["reflection_maxdepth"]->setInt( 5 );

  // Metal material
  Program phong_ch = m_context->createProgramFromPTXFile( ptxpath( "rayDifferentials", "phong_mip.cu" ), "closest_hit_radiance" );
  Program phong_ah = m_context->createProgramFromPTXFile( ptxpath( "rayDifferentials", "phong_mip.cu" ), "any_hit_shadow" );

  Material metal_matl = m_context->createMaterial();
  metal_matl->setClosestHitProgram( 0, phong_ch );
  metal_matl->setAnyHitProgram( 1, phong_ah );
  metal_matl["Ka"]->setFloat( 0.2f, 0.2f, 0.2f );
  metal_matl["Kd"]->setFloat( 0.4f, 0.4f, 0.4f );
  metal_matl["Ks"]->setFloat( 0.9f, 0.9f, 0.9f );
  metal_matl["phong_exp"]->setFloat( 64 );
  metal_matl["reflectivity"]->setFloat( 0.6f,  0.6f,  0.6f);

  // Checker material for floor
  Program check_ch = m_context->createProgramFromPTXFile( ptxpath( "rayDifferentials", "flat_tex_mip.cu" ), "closest_hit_radiance" );
  Program check_ah = m_context->createProgramFromPTXFile( ptxpath( "rayDifferentials", "flat_tex_mip.cu" ), "any_hit_shadow" );
  Material floor_matl = m_context->createMaterial();
  floor_matl->setClosestHitProgram( 0, check_ch );
  floor_matl->setAnyHitProgram( 1, check_ah );

  floor_matl["Kd1"]->setFloat( 0.8f, 0.3f, 0.15f);
  floor_matl["Ka1"]->setFloat( 0.8f, 0.3f, 0.15f);
  floor_matl["Ks1"]->setFloat( 0.0f, 0.0f, 0.0f);
  floor_matl["Kd2"]->setFloat( 0.9f, 0.85f, 0.05f);
  floor_matl["Ka2"]->setFloat( 0.9f, 0.85f, 0.05f);
  floor_matl["Ks2"]->setFloat( 0.0f, 0.0f, 0.0f);
  floor_matl["inv_checker_size"]->setFloat( 32.0f, 16.0f, 1.0f );
  floor_matl["phong_exp1"]->setFloat( 0.0f );
  floor_matl["phong_exp2"]->setFloat( 0.0f );
  floor_matl["reflectivity1"]->setFloat( 0.0f, 0.0f, 0.0f);
  floor_matl["reflectivity2"]->setFloat( 0.0f, 0.0f, 0.0f);

  loadTextures(floor_matl);

  // Create GIs for each piece of geometry
  std::vector<GeometryInstance> gis;
  gis.push_back( m_context->createGeometryInstance( sphere, &glass_matl, &glass_matl+1 ) );
  gis[0]["sphere"]->setFloat( -1.0f, 2.3f, -2.0f, 2.0f );
  gis.push_back( m_context->createGeometryInstance( sphere,  &metal_matl,  &metal_matl+1 ) );
  gis[1]["sphere"]->setFloat( 0.0f, 2.0f, 1.5f, 1.5f );
  gis.push_back( m_context->createGeometryInstance( parallelogram, &floor_matl,  &floor_matl+1 ) );

  // Place all in group
  GeometryGroup geometrygroup = m_context->createGeometryGroup();
  geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()) );
  geometrygroup->setChild( 0, gis[0] );
  geometrygroup->setChild( 1, gis[1] );
  geometrygroup->setChild( 2, gis[2] );
  geometrygroup->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );

  m_context["top_object"]->set( geometrygroup );
  m_context["top_shadower"]->set( geometrygroup );
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
    << "  -t  | --texture                            Specify input texture\n"
    << std::endl;
  GLUTDisplay::printUsage();

  std::cerr
    << "App keystrokes:\n"
    << "  l Decrease MIP mapping mode\n"
    << "  L Increase MIP mapping mode\n"
    << "  t Decrease Texture scaling factor\n"
    << "  T Increase Texture scaling factor\n"
    << std::endl;

  if ( doExit ) exit(1);
}


int main( int argc, char** argv )
{
  GLUTDisplay::init( argc, argv );

  unsigned int width = 512u, height = 512u;

  std::string tex_filename = std::string(sutilSamplesDir()) + "/rayDifferentials/tex0.ppm";

  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "--help" || arg == "-h" ) {
      printUsageAndExit( argv[0] );
    } else if ( arg == "--texture" || arg == "-t" ) {
      if ( i == argc-1 ) {
        printUsageAndExit( argv[0] );
      }
      tex_filename = argv[++i];
    } else if ( arg.substr( 0, 6 ) == "--dim=" ) {
      std::string dims_arg = arg.substr(6);
      if ( sutilParseImageDimensions( dims_arg.c_str(), &width, &height ) != RT_SUCCESS ) {
        std::cerr << "Invalid window dimensions: '" << dims_arg << "'" << std::endl;
        printUsageAndExit( argv[0] );
      }
    } else {
      std::cerr << "Unknown option: '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

  try {
    RayDifferentialsScene scene(tex_filename);
    scene.setDimensions( width, height );

    GLUTDisplay::run( "RayDifferentialsScene", &scene );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }
  return 0;
}

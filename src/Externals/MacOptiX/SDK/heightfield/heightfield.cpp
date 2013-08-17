
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
//  heightfield.cpp: render a simple heightfield
//
//  Options        : [ sinxy | sync | plane | plane2 | file_name ]
//
//-----------------------------------------------------------------------------


#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <fstream>
#include <iostream>
#include <float.h>
#include <GLUTDisplay.h>
#include "commonStructs.h"
#include <cstdlib>
#include <cstring>

using namespace optix;

//-----------------------------------------------------------------------------
//
// Heightfield data creation routines
//
//-----------------------------------------------------------------------------

namespace {
  inline float plane(float x, float y)
  {
    return 0.5f;
  }

  inline float plane2(float x, float y)
  {
    return y;
  }

  inline float sinxy(float x, float y)
  {
    float r2 = 8.0f*(x*x+y*y);
    if(r2 > -1.e-12f && r2 < 1.e-12f)
      return 1.0f;
    else
      return (1.2f*sinf(x*15.0f)*sinf(y*30.0f)/(r2*sqrtf(r2)+0.47f));
  }

  inline float sinc(float x, float y)
  {
    float r = sqrtf(x*x+y*y)*20;
    return (r==0.0f?1.0f:sinf(r)/r)+1.0f;
  }

  template<typename F>
    static void fillBuffer(F& f, Context& context, Buffer& databuffer, int nx, int ny)
    {
      databuffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT, nx+1, ny+1 );
      float* p = reinterpret_cast<float*>(databuffer->map());
      for(int i = 0; i<= nx; i++){
        float x = float(i)/nx * 2 - 1;
        for(int j = 0; j<= ny; j++){
          float y = float(j)/ny * 2 - 1;
          *p++ = f(x, y);
        }
      }
      databuffer->unmap();
    }
}


//-----------------------------------------------------------------------------
// 
// Heightfield Scene
//
//-----------------------------------------------------------------------------

class HeightfieldScene : public SampleScene
{
public:
  HeightfieldScene(const std::string& dataname) : dataname(dataname) {}

  // From SampleScene
  virtual void   initScene( InitialCameraData& camera_data );
  virtual void   trace( const RayGenCameraData& camera_data );
  virtual Buffer getOutputBuffer();

  void createGeometry();
  void createData();

private:
  Buffer      databuffer;
  float       ymin, ymax;
  std::string dataname;

  static unsigned int WIDTH;
  static unsigned int HEIGHT;
};

unsigned int HeightfieldScene::WIDTH  = 512u;
unsigned int HeightfieldScene::HEIGHT = 384u;


void HeightfieldScene::initScene( InitialCameraData& camera_data )
{
  try {
    // Setup state
    m_context->setRayTypeCount( 2 );
    m_context->setEntryPointCount( 1 );
    m_context->setStackSize(560);

    m_context["radiance_ray_type"]->setUint( 0u );
    m_context["scene_epsilon"]->setFloat( 1.e-3f );
    m_context["max_depth"]->setInt(5);
    m_context["radiance_ray_type"]->setUint(0);
    m_context["shadow_ray_type"]->setUint(1);
    m_context["output_buffer"]->set( createOutputBuffer( RT_FORMAT_UNSIGNED_BYTE4, WIDTH, HEIGHT ) );

    // Set up camera
    camera_data = InitialCameraData( make_float3( 4.0f, 4.0f, 4.0f ), // eye
                                     make_float3( 0.0f, 0.0f, 0.3f ), // lookat
                                     make_float3( 0.0f, 1.0f, 0.0f ), // up
                                     45.0f );                         // vfov

    // Declare camera variables.  The values do not matter, they will be overwritten in trace.
    m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

    m_context["bad_color"]->setFloat( 1.0f, 1.0f, 0.0f );
    m_context["bg_color"]->setFloat( make_float3(.1f, 0.2f, 0.4f) * 0.5f );

    // Ray gen program
    std::string ptx_path = ptxpath( "heightfield", "pinhole_camera.cu" );
    Program ray_gen_program = m_context->createProgramFromPTXFile( ptx_path, "pinhole_camera" );
    m_context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception program
    Program exception_program = m_context->createProgramFromPTXFile( ptx_path, "exception" );
    m_context->setExceptionProgram( 0, exception_program );
    m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptxpath( "heightfield", "constantbg.cu" ), "miss" ) );

    // Set up light buffer
    m_context["ambient_light_color"]->setFloat( 1.0f, 1.0f, 1.0f );
    BasicLight lights[] = { 
      { { 4.0f, 12.0f, 10.0f }, { 1.0f, 1.0f, 1.0f }, 1 }
    };

    Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
    light_buffer->setFormat(RT_FORMAT_USER);
    light_buffer->setElementSize(sizeof(BasicLight));
    light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
    memcpy(light_buffer->map(), lights, sizeof(lights));
    light_buffer->unmap();

    m_context["lights"]->set(light_buffer);

    createData();
    createGeometry();

    // Finalize
    m_context->validate();
    m_context->compile();

  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }
}


Buffer HeightfieldScene::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}


void HeightfieldScene::trace( const RayGenCameraData& camera_data )
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

void HeightfieldScene::createGeometry()
{
  Geometry heightfield = m_context->createGeometry();
  heightfield->setPrimitiveCount( 1u );

  heightfield->setBoundingBoxProgram( m_context->createProgramFromPTXFile( ptxpath( "heightfield", "heightfield.cu" ), "bounds" ) );
  heightfield->setIntersectionProgram( m_context->createProgramFromPTXFile( ptxpath( "heightfield", "heightfield.cu" ), "intersect" ) );
  float3 min = make_float3(-2, ymin, -2);
  float3 max = make_float3( 2, ymax,  2);
  RTsize nx, nz;
  databuffer->getSize(nx, nz);
  
  // If buffer is nx by nz, we have nx-1 by nz-1 cells;
  float3 cellsize = (max - min) / (make_float3(static_cast<float>(nx-1), 1.0f, static_cast<float>(nz-1)));
  cellsize.y = 1;
  float3 inv_cellsize = make_float3(1)/cellsize;
  heightfield["boxmin"]->setFloat(min);
  heightfield["boxmax"]->setFloat(max);
  heightfield["cellsize"]->setFloat(cellsize);
  heightfield["inv_cellsize"]->setFloat(inv_cellsize);
  heightfield["data"]->setBuffer(databuffer);

  // Create material
  Program phong_ch = m_context->createProgramFromPTXFile( ptxpath( "heightfield", "phong.cu" ), "closest_hit_radiance" );
  Program phong_ah = m_context->createProgramFromPTXFile( ptxpath( "heightfield", "phong.cu" ), "any_hit_shadow" );
  Material heightfield_matl = m_context->createMaterial();
  heightfield_matl->setClosestHitProgram( 0, phong_ch );
  heightfield_matl->setAnyHitProgram( 1, phong_ah );

  heightfield_matl["Ka"]->setFloat(0.0f, 0.3f, 0.1f);
  heightfield_matl["Kd"]->setFloat(0.1f, 0.7f, 0.2f);
  heightfield_matl["Ks"]->setFloat(0.6f, 0.6f, 0.6f);
  heightfield_matl["phong_exp"]->setFloat(132);
  heightfield_matl["reflectivity"]->setFloat(0, 0, 0);

  GeometryInstance gi = m_context->createGeometryInstance( heightfield, &heightfield_matl, &heightfield_matl+1 );
  
  GeometryGroup geometrygroup = m_context->createGeometryGroup();
  geometrygroup->setChildCount( 1 );
  geometrygroup->setChild( 0, gi );
  geometrygroup->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );
  
  m_context["top_object"]->set( geometrygroup );
  m_context["top_shadower"]->set( geometrygroup );
}


void HeightfieldScene::createData()
{
  if(dataname == "sinxy"){
    fillBuffer(sinxy, m_context, databuffer, 100, 100);
  } else if(dataname == "plane"){
    fillBuffer(plane, m_context, databuffer, 10, 10);
  } else if(dataname == "plane2"){
    fillBuffer(plane2, m_context, databuffer, 10, 10);
  } else if(dataname == "sinc"){
    fillBuffer(sinc, m_context, databuffer, 50, 50);
  } else {
    // Try to open as a file
    std::ifstream in(dataname.c_str(), std::ios::binary);
    if(!in){
      std::cerr << "Error opening '" << dataname << "'\n";
      exit(1);
    }
    int nx, nz;
    in >> nx >> nz;
    if(!in){
      std::cerr << "Error reading header from '" << dataname << "'\n";
      exit(1);
    }
    in.get();
    databuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT, nx+1, nz+1 );
    float* p = reinterpret_cast<float*>(databuffer->map());
    in.read(reinterpret_cast<char*>(p), sizeof(float)*(nx+1)*(nz+1));
    if(!in){
      std::cerr << "Error reading data from '" << dataname << "'\n";
      exit(1);
    }
    databuffer->unmap();
  }

  // Compute data range
  ymin = FLT_MAX;
  ymax = -FLT_MAX;
  RTsize width, height;
  databuffer->getSize(width, height);
  RTsize size = width * height;
  float* p = reinterpret_cast<float*>(databuffer->map());
  for(RTsize i=0;i<size; i++){
    float value = *p++;
    ymin = fminf(ymin, value);
    ymax = fmaxf(ymax, value);
  }
  ymin -= 1.e-6f;
  ymax += 1.e-6f;
  databuffer->unmap();
}


//-----------------------------------------------------------------------------
//
// main 
//
//-----------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"

    << "  -h  | --help                               Print this usage message\n"
    << "  -ds | --dataset <data set>                 Specify data set to render\n"
    << "\n"
    << "<data set>: sinxy | plane | plane2 | sinc | filename\n"
    << std::endl;
  GLUTDisplay::printUsage();

  if ( doExit ) exit(1);
}


int main( int argc, char** argv )
{
  GLUTDisplay::init( argc, argv );

  // Process command line options
  std::string dataset( "sinxy" );
  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "--help" || arg == "-h" ) {
      printUsageAndExit( argv[0] );
    } else if ( arg == "-ds" || arg == "--dataset" ) {
      if ( i == argc-1 ) {
        printUsageAndExit( argv[0] );
      }
      dataset = argv[++i];
    } else {
      std::cerr << "Unknown option: '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

  try {
    HeightfieldScene scene( dataset );
    GLUTDisplay::run( "HeightfieldScene", &scene );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }

  return 0;
}


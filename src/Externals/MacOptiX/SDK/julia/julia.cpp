
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

#include <optix.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sutil.h>
#include <optixu/optixpp_namespace.h>
#include <GLUTDisplay.h>
#include "commonStructs.h"
#include <Mouse.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

static const unsigned int WIDTH = 1280;
static const unsigned int HEIGHT = 720;

enum AnimState {
  ANIM_ALL,       // full auto
  ANIM_JULIA,     // manual camera
  ANIM_NONE       // pause all
};
static AnimState animstate = ANIM_ALL;

// Encapulates the state of the floor
struct FloorState
{
  FloorState()
    : m_t( 0 )
  {}

  void update( double t )
  {
    if( animstate==ANIM_NONE )
      return;
    m_t += t * 0.3f;
  }

  double m_t;
};

// Moving force particle.
struct Particle
{
  Particle()
    : m_pos( make_float3(1,1,0) )
    , m_t( 0 )
  {}

  void update( double t )
  {
    if( animstate==ANIM_NONE )
      return;
    m_t += t * 0.3f;
    m_pos.x = (float)( sin( m_t ) * cos( m_t*0.2 ) );
    m_pos.y = (float)sin( m_t*2.5 );
    m_pos.z = (float)( cos( m_t*1.8 ) * sin( m_t ) );
  }

  float3 m_pos;
  double m_t;
};

// Animated parameter quaternion.
static const int nposes = 12;
static const float4 poses[nposes] = {
  {-0.5f, 0.1f, 0.2f, 0.3f },
  {-0.71f, 0.31f, -0.02f, 0.03f },
  {-0.5f, 0.1f, 0.59f, 0.03f },
  { -0.5f, -0.62f, 0.2f, 0.3f },
  {-0.57f, 0.04f, -0.17f, 0.36f },
  {0.0899998f, -0.71f, -0.02f, 0.08f },
  {-0.19f, -0.22f, -0.79f, 0.03f },
  {0.49f, 0.48f, -0.38f, -0.11f },
  {-0.19f, 0.04f, 0.0299999f, 0.77f },
  { 0.0299998f, -1.1f, -0.03f, -0.1f },
  {0.45f, 0.04f, 0.56f, -0.00999998f },
  { -0.5f, -0.61f, -0.08f, -0.00999998f }
};
struct ParamQuat
{
  ParamQuat()
    : m_c( make_float4( -0.5f, 0.1f, 0.2f, 0.3f ) )
    , m_t( 0 )
  {}

  void update( double t )
  {
    if( animstate==ANIM_NONE )
      return;
    m_t += t * 0.03f;
    const float rem   = fmodf( (float)m_t, (float)nposes );
    const int   p0    = (int)rem;
    const int   p1    = (p0+1) % nposes;
    const float lin   = rem - (float)p0;
    const float blend = smoothstep( 0.0f, 1.0f, lin );
    m_c = lerp( poses[p0], poses[p1], blend );
  }
  
  float4 m_c;
  double m_t;
};

// Animated camera.
struct AnimCamera
{
  AnimCamera()
    : m_pos( make_float3(0) )
    , m_aspect( (float)WIDTH/(float)HEIGHT )
    , m_t( 0 )
  {}

  void update( double t )
  {
    m_t += t * 0.1;
    m_pos.y = (float)( 2 + sin( m_t*1.5 ) );
    m_pos.x = (float)( 2.3*sin( m_t ) );
    m_pos.z = (float)( 0.5+2.1*cos( m_t ) );
  }

  void apply( Context context )
  {
    PinholeCamera pc( m_pos, make_float3(0), make_float3(0,1,0), 60.f, 60.f/m_aspect );
    float3 eye, u, v, w;
    pc.getEyeUVW( eye, u, v, w );
    context["eye"]->setFloat( eye );
    context["U"]->setFloat( u );
    context["V"]->setFloat( v );
    context["W"]->setFloat( w );
  }

  float3 m_pos;
  float  m_aspect;
  double m_t;
};

class JuliaScene : public SampleScene
{
public:

  JuliaScene()
    : m_alpha( 0.003f ),
      m_delta( 1.0f ),
      m_DEL( 0.02f), // relatively large value here gives the candy-look with normal shader
      m_max_iterations( 12 )
  {
    sutilCurrentTime(&m_previousFrameTime);
  }

  // From SampleScene
  virtual void   initScene( InitialCameraData& camera_data );
  virtual void   trace( const RayGenCameraData& camera_data );
  virtual Buffer getOutputBuffer();
  virtual bool   keyPressed(unsigned char key, int x, int y);

  virtual void resize(unsigned int width, unsigned int height);

  void createGeometry();

  static bool m_useGLBuffer;

private:
  void createFloor(std::vector<GeometryInstance> &geometryInstances);

  // parameters of Equation 16 of [Hart et al., 1989]
  float  m_alpha;
  float  m_delta;

  float m_DEL; 

  unsigned int m_max_iterations;

  FloorState  m_floorstate; // the state of the floor
  Particle    m_particle;   // moving force particle
  ParamQuat   m_param;      // julia set parameter
  AnimCamera  m_cam;        // camera animation

  double m_previousFrameTime;

  GeometryGroup m_geometrygroup;
};

// make PBOs default since it's a lot of faster
bool         JuliaScene::m_useGLBuffer = true;

void JuliaScene::resize(unsigned int width, unsigned int height)
{
  Buffer buffer = getOutputBuffer();
  SampleScene::resize(width, height);
  m_cam.m_aspect = (float)width / (float)height;
}


void JuliaScene::initScene( InitialCameraData& camera_data )
{
  try {
    // Setup state
    m_context->setRayTypeCount( 2 );
    m_context->setEntryPointCount( 1 );
    m_context->setStackSize(1280);

    m_context["max_depth"]->setInt( 5 );
    m_context["radiance_ray_type"]->setUint( 0u );
    m_context["shadow_ray_type"]->setUint( 1u );
    m_context["scene_epsilon"]->setFloat( 1.e-4f );
    m_context["color_t"]->setFloat( 0.0f );
    m_context["shadowsActive"]->setUint( 0u );

    Variable output_buffer = m_context["output_buffer"];

    output_buffer->set(createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, WIDTH, HEIGHT));

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
    std::string ptx_path( ptxpath( "julia", "pinhole_camera.cu" ) );
    Program ray_gen_program = m_context->createProgramFromPTXFile( ptx_path, "pinhole_camera" );
    m_context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception
    Program exception_program = m_context->createProgramFromPTXFile( ptx_path, "exception" );
    m_context->setExceptionProgram( 0, exception_program );
    m_context["bad_color"]->setFloat( 1.0f, 1.0f, 0.0f );

    // Miss program
    m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptxpath( "julia", "constantbg.cu" ), "miss" ) );
    m_context["bg_color"]->setFloat( make_float3(108.0f/255.0f, 166.0f/255.0f, 205.0f/255.0f) * 0.5f );

    // Setup lights
    m_context["ambient_light_color"]->setFloat(0.1f,0.1f,0.3f);
    BasicLight lights[] = { 
      { { 0.0f, 8.0f, -5.0f }, { .8f, .8f, .6f }, 1 },
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


Buffer JuliaScene::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}


void JuliaScene::trace( const RayGenCameraData& camera_data )
{
  double t;

  // are we doing continuous rendering right now?
  if(GLUTDisplay::getContinuousMode() != GLUTDisplay::CDNone) {
    // note the current time
    sutilCurrentTime(&t);
  } else {
    // set the current time to the previous time
    // to cause elapsed time to be 0
    t = m_previousFrameTime;
  }

  const double time_elapsed = t - m_previousFrameTime;

  // remember what time we traced this frame
  m_previousFrameTime = t;

  // update sphere
  m_particle.update( time_elapsed );
  m_context[ "particle" ]->setFloat( m_particle.m_pos );
  m_context[ "sphere" ]->setFloat( m_particle.m_pos.x, m_particle.m_pos.y, m_particle.m_pos.z, 0.25f );

  // update quaternion
  m_param.update( time_elapsed );
  m_context[ "c4" ]->setFloat( m_param.m_c );

  // update julia set color
  static double col_t = 0;
  static double last_col_change = 0;
  static double col_sign = 1;
  static double col_lerp = 0;
  if( last_col_change == 0 )
    sutilCurrentTime( &last_col_change );
  if( t - last_col_change > 15 )
  {
    col_lerp += time_elapsed * 0.2;
    if( col_lerp >= 1. ) {
      last_col_change = t;
      col_lerp = 0;
      col_t += col_sign;
      if( col_t < 0.5 || col_t > 1.5 )
        col_sign *= -1;
    }
    m_context["color_t"]->setFloat( (float)(col_t+col_sign*col_lerp) );
  }

  // update floor
  m_floorstate.update( time_elapsed );
  m_context[ "floor_time" ]->setFloat( (float)m_floorstate.m_t );

  // update camera
  if( animstate == ANIM_ALL )
  {
    m_cam.update( time_elapsed );
    m_cam.apply( m_context );
  }
  else
  {
    m_context["eye"]->setFloat( camera_data.eye );
    m_context["U"]->setFloat( camera_data.U );
    m_context["V"]->setFloat( camera_data.V );
    m_context["W"]->setFloat( camera_data.W );
  }

  // make sure to rebuild the AS for the moving sphere
  m_geometrygroup->getAcceleration()->markDirty();

  // trace
  Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );
  m_context->launch( 0, 
                   static_cast<unsigned int>(buffer_width),
                   static_cast<unsigned int>(buffer_height)
                   );
}

void JuliaScene::createFloor(std::vector<GeometryInstance> &geometryInstances)
{
  // Normal programs
  Program normal_ch = m_context->createProgramFromPTXFile( ptxpath( "julia", "normal_shader.cu" ), "closest_hit_radiance" );
  Program normal_ah = m_context->createProgramFromPTXFile( ptxpath( "julia", "normal_shader.cu" ), "any_hit_shadow" );

  // Normal material
  Material normal_matl = m_context->createMaterial();
  normal_matl->setClosestHitProgram( 0, normal_ch );
  normal_matl->setAnyHitProgram( 1, normal_ah );

  // box programs
  Program box_bound     = m_context->createProgramFromPTXFile( ptxpath( "julia", "box.cu" ), "box_bounds" );
  Program box_intersect = m_context->createProgramFromPTXFile( ptxpath( "julia", "box.cu" ), "box_intersect" );


  size_t num_repetitions = 32;
  optix::Aabb bounds_of_floor( make_float3(-10.0f, -2.0f, -10.0f),
                                     make_float3( 10.0f, -1.0f,  10.0f));
  float step_x = bounds_of_floor.extent().x / num_repetitions;
  float step_y = bounds_of_floor.extent().y;
  float step_z = bounds_of_floor.extent().z / num_repetitions;

  float3 min = bounds_of_floor.m_min;
  for(unsigned int i = 0; i < num_repetitions; ++i, min.x += step_x) {

    min.z = bounds_of_floor.m_min.z;
    for(unsigned int j = 0; j < num_repetitions; ++j, min.z += step_z) {
      Geometry block = m_context->createGeometry();
      block->setPrimitiveCount( 1u );
      block->setBoundingBoxProgram( box_bound );
      block->setIntersectionProgram( box_intersect );

      float3 max = min + make_float3(step_x, step_y, step_z);

      block["boxmin"]->setFloat( min.x, min.y, min.z );
      block["boxmax"]->setFloat( max.x, max.y, max.z );
      geometryInstances.push_back( m_context->createGeometryInstance( block, &normal_matl, &normal_matl+1 ) );
    }
  }
}


void JuliaScene::createGeometry()
{
  // Julia object
  Geometry julia = m_context->createGeometry();
  julia->setPrimitiveCount( 1u );
  julia->setBoundingBoxProgram( m_context->createProgramFromPTXFile( ptxpath( "julia", "julia.cu" ), "bounds" ) );
  julia->setIntersectionProgram( m_context->createProgramFromPTXFile( ptxpath( "julia", "julia.cu" ), "intersect" ) );
  
  // Sphere
  Geometry sphere = m_context->createGeometry();
  sphere->setPrimitiveCount( 1 );
  sphere->setBoundingBoxProgram( m_context->createProgramFromPTXFile( ptxpath( "julia", "sphere.cu" ), "bounds" ) );
  sphere->setIntersectionProgram( m_context->createProgramFromPTXFile( ptxpath( "julia", "sphere.cu" ), "intersect" ) );
  m_context["sphere"]->setFloat( 1, 1, 1, 0.2f );

  // Floor
  Geometry floor = m_context->createGeometry();
  floor->setPrimitiveCount( 1u );

  std::string ptx_path( ptxpath( "julia", "block_floor.cu" ) );
  floor->setBoundingBoxProgram( m_context->createProgramFromPTXFile( ptx_path, "bounds" ) );
  floor->setIntersectionProgram( m_context->createProgramFromPTXFile( ptx_path, "intersect" ) );

  Program julia_ch = m_context->createProgramFromPTXFile( ptxpath( "julia", "julia.cu" ), "julia_ch_radiance" );
  Program julia_ah = m_context->createProgramFromPTXFile( ptxpath( "julia", "julia.cu" ), "julia_ah_shadow" );
  Program chrome_ch = m_context->createProgramFromPTXFile( ptxpath( "julia", "julia.cu" ), "chrome_ch_radiance" );
  Program chrome_ah = m_context->createProgramFromPTXFile( ptxpath( "julia", "julia.cu" ), "chrome_ah_shadow" );
  Program floor_ch = m_context->createProgramFromPTXFile( ptxpath( "julia", "block_floor.cu" ), "block_floor_ch_radiance" );
  Program floor_ah = m_context->createProgramFromPTXFile( ptxpath( "julia", "block_floor.cu" ), "block_floor_ah_shadow" );
  Program normal_ch = m_context->createProgramFromPTXFile( ptxpath( "julia", "normal_shader.cu" ), "closest_hit_radiance" );

  // Julia material
  Material julia_matl = m_context->createMaterial();
  julia_matl->setClosestHitProgram( 0, julia_ch );
  julia_matl->setAnyHitProgram( 1, julia_ah );

  // Sphere material
  Material sphere_matl = m_context->createMaterial();
  sphere_matl->setClosestHitProgram( 0, chrome_ch );
  sphere_matl->setAnyHitProgram( 1, chrome_ah );

  m_context["Ka"]->setFloat(0.3f,0.3f,0.3f);
  m_context["Kd"]->setFloat(.6f, 0.1f, 0.1f);
  m_context["Ks"]->setFloat(.6f, .6f, .6f);
  m_context["phong_exp"]->setFloat(32);
  m_context["reflectivity"]->setFloat(.4f, .4f, .4f);
  

  // Floor material
  Material floor_matl = m_context->createMaterial();
  floor_matl->setClosestHitProgram( 0, floor_ch );
  floor_matl->setAnyHitProgram( 1, floor_ah );

  // Place geometry into hierarchy
  std::vector<GeometryInstance> gis;
  gis.push_back( m_context->createGeometryInstance( sphere, &sphere_matl, &sphere_matl+1 ) );
  gis.push_back( m_context->createGeometryInstance( julia,        &julia_matl, &julia_matl+1 ) );
  //gis.push_back( m_context->createGeometryInstance( floor, &floor_matl, &floor_matl+1 ) );

  m_geometrygroup = m_context->createGeometryGroup();
  m_geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()) );
  for(size_t i = 0; i < gis.size(); ++i) {
    m_geometrygroup->setChild( (int)i, gis[i] );
  }
  m_geometrygroup->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );

  // GG for floor
  GeometryGroup floor_gg = m_context->createGeometryGroup();
  floor_gg->setChildCount( 1 );
  floor_gg->setChild( 0, m_context->createGeometryInstance( floor, &floor_matl, &floor_matl+1 ) );
  floor_gg->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );


  // Top level group
  Group topgroup = m_context->createGroup();
  topgroup->setChildCount( 2 );
  topgroup->setChild( 0, m_geometrygroup );
  topgroup->setChild( 1, floor_gg );
  topgroup->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );

  m_context["top_object"]->set( topgroup );
  m_context["top_shadower"]->set( m_geometrygroup );

  // set QJS parameters
  m_context[ "c4" ]->setFloat( m_param.m_c );
  m_context[ "alpha" ]->setFloat( m_alpha );
  m_context[ "delta" ]->setFloat( m_delta );
  m_context[ "max_iterations" ]->setUint( m_max_iterations );
  m_context[ "DEL" ]->setFloat( m_DEL );
  m_context[ "particle" ]->setFloat( 0.5f, 0.5f, 0.4f );

  // set floor parameters
  m_context[ "floor_time" ]->setFloat( (float)m_floorstate.m_t );
}

bool JuliaScene::keyPressed(unsigned char key, int x, int y)
{
  float delta = 0.01f;

  float4& cref = m_param.m_c;

   switch (key)
   {
     case 'z': {
                 fprintf(stderr,"watch out..\n");
                 m_context["hansdampf"]->setFloat( 3.f );
                 return true;
               } 
     case 'a':
       m_alpha *= 0.99f;
       std::cerr << "alpha: " << m_alpha << std::endl;
       m_context[ "alpha" ]->setFloat( m_alpha );
       return true;
     case 'A':
       m_alpha *= 1.01f;
       std::cerr << "alpha: " << m_alpha << std::endl;
       m_context[ "alpha" ]->setFloat( m_alpha );
       return true;
     case 'u':
       cref.x += delta;
       std::cerr << cref.x << ", " << cref.y << ", " << cref.z << ", " << cref.w << ", " << std::endl;
       m_context[ "c4" ]->setFloat( cref );
       return true;
     case 'U':
       cref.x -= delta;
       std::cerr << cref.x << ", " << cref.y << ", " << cref.z << ", " << cref.w << ", " << std::endl;
       m_context[ "c4" ]->setFloat( cref );
       return true;
     case 'i':
       cref.y += delta;
       std::cerr << cref.x << ", " << cref.y << ", " << cref.z << ", " << cref.w << ", " << std::endl;
       m_context[ "c4" ]->setFloat( cref );
       return true;
     case 'I':
       cref.y -= delta;
       std::cerr << cref.x << ", " << cref.y << ", " << cref.z << ", " << cref.w << ", " << std::endl;
       m_context[ "c4" ]->setFloat( cref );
       return true;
     case 'o':
       cref.z += delta;
       std::cerr << cref.x << ", " << cref.y << ", " << cref.z << ", " << cref.w << ", " << std::endl;
       m_context[ "c4" ]->setFloat( cref );
       return true;
     case 'O':
       cref.z -= delta;
       std::cerr << cref.x << ", " << cref.y << ", " << cref.z << ", " << cref.w << ", " << std::endl;
       m_context[ "c4" ]->setFloat( cref );
       return true;
     case 'p':
       cref.w += delta;
       std::cerr << cref.x << ", " << cref.y << ", " << cref.z << ", " << cref.w << ", " << std::endl;
       m_context[ "c4" ]->setFloat( cref );
       return true;
     case 'P':
       cref.w -= delta;
       std::cerr << cref.x << ", " << cref.y << ", " << cref.z << ", " << cref.w << ", " << std::endl;
       m_context[ "c4" ]->setFloat( cref );
       return true;
     case 'r':
       // we're about to stop or begin rendering, so note the current time
       sutilCurrentTime(&m_previousFrameTime);
       // return false to cause GLUTDisplay's key handler to be invoked
       return false;
     case 'S':
       m_context[ "shadowsActive" ]->setUint( !!! m_context[ "shadowsActive" ]->getUint() );
       return true;
     case 'l':
       m_max_iterations = std::max(m_max_iterations - 1u, 1u);
       std::cerr << "max_iterations: " << m_max_iterations << std::endl;
       m_context[ "max_iterations" ]->setUint( m_max_iterations );
       return true;
     case 'L':
       m_max_iterations++;
       std::cerr << "max_iterations: " << m_max_iterations << std::endl;
       m_context[ "max_iterations" ]->setUint( m_max_iterations );
       return true;
     case '-':
       m_DEL *= 0.99f;
       std::cerr << "DEL: " << m_DEL << std::endl;
       m_context[ "DEL" ]->setFloat( m_DEL );
       return true;
     case '+':
       m_DEL *= 1.01f;
       std::cerr << "DEL: " << m_DEL << std::endl;
       m_context[ "DEL" ]->setFloat( m_DEL );
       return true;
     case '/': {
       std::ofstream of( "poses.txt", std::ios::app );
       of << "\n  { " << m_param.m_c.x << "f, " << m_param.m_c.y << "f, " << m_param.m_c.z << "f, " << m_param.m_c.w << "f },";
       return true;
       }
     case ' ': {
       animstate = AnimState( (animstate+1) % 3 );
       return true;
      }
   }
   return false;
}

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"

    << "  -h  | --help                               Print this usage message\n"
    << "  -P  | --pbo                                Use OpenGL PBO for output buffer (default)\n"
    << "  -n  | --nopbo                              Use internal output buffer\n"
    << std::endl;
  GLUTDisplay::printUsage();

  std::cerr
    << "App keystrokes:\n"
    << "  a Decrease alpha\n"
    << "  A Increase alpha\n"
    << "  l Decrease Julia set calculation iterations\n"
    << "  L Increase Julia set calculation iterations\n"
    << "  S Toggle the ground plane shadows on and off\n"
    << "\n"
    << "Most easily seen when scene is not animating:\n"
    << "  u,i,o,p  Increase the parameters defining the Julia set\n"
    << "  U,I,O,P  Decrease the parameters defining the Julia set\n"
    << "  space    Cycles camera from animated, to interactive, to interactive with no animation\n"
    << std::endl;

  if ( doExit ) exit(1);
}


int main( int argc, char** argv )
{
  GLUTDisplay::init( argc, argv );

  for(int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if(arg == "-p" || arg == "--pbo") {
      JuliaScene::m_useGLBuffer = true;
    } else if( arg == "-n" || arg == "--nopbo" ) {
      JuliaScene::m_useGLBuffer = false;
    } else if( arg == "-h" || arg == "--help" ) {
      printUsageAndExit(argv[0]);
    } else {
      std::cerr << "Unknown option '" << arg << "'\n";
      printUsageAndExit(argv[0]);
    }
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

  try {
    JuliaScene scene;
    GLUTDisplay::run( "JuliaScene", &scene, GLUTDisplay::CDAnimated );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }

  return 0;
}

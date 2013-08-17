
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

//------------------------------------------------------------------------------
//
//  swimmingShark.cpp -- Renders marine life OBJ models with a time-varying warping transform.
//
//------------------------------------------------------------------------------

// Models and textures copyright Toru Miyazawa, Toucan Corporation. Used by permission.
// http://toucan.web.infoseek.co.jp/3DCG/3ds/FishModelsE.html

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_math_stream.h>
#include <sutil.h>
#include <GLUTDisplay.h>
#include <ObjLoader.h>
#include <ImageLoader.h>
#include <string>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <ctime>

using namespace optix;

inline float FRand( ) { return float( rand( ) ) / ( float( RAND_MAX ) ); } // A random number on 0.0 to 1.0.
inline float FRand( const float high ) { return FRand( ) * high; } // A random number on 0.0 to high.
inline float FRand( const float low, const float high ) { return low + FRand( ) * ( high - low ); } // A random number on low to high.
inline float3 MakeDRand( const float3 low, const float3 high )
{
  return make_float3( FRand( low.x, high.x ), FRand( low.y, high.y ), FRand( low.z, high.z ) );
}

const float SPAN = 50.0f; // meter radius
const optix::Aabb SCENE_BOX( make_float3( -SPAN, 0, -SPAN ), make_float3( SPAN, 12.0f, SPAN ) );
optix::Aabb TargetBox( make_float3( SCENE_BOX.m_min.x * 0.4f, SCENE_BOX.m_min.y, SCENE_BOX.m_max.z * 0.85f ),
                            make_float3( SCENE_BOX.m_max.x * 0.4f, SCENE_BOX.m_max.y, SCENE_BOX.m_max.z ) );

struct Species_t
{
  std::string name;   // String name of OBJ file
  float sizeInObj;    // Size in OBJ, for scaling to max( abs( z ) )==1.0.
  float lengthMeters; // Length in meters
  float centerOffset; // Amount to offset center in -1..1 space
  float maxAmplRad;   // Max angle the head and tail can swing
  float headWaveLen;  // Number of degrees of sine wave from nose to origin
  float tailWaveLen;  // Number of degrees of sine wave from origin to tail
  float speed;        // Speed as a percentage of size
};

// Note that I sorted these fish based on lengthMeters (column 3) instead of file name using:
// column -t speciesInfo.h | sort -k 3 -n > speciesInfo.h
const static Species_t SpeciesInfo[] = {
#include "speciesInfo.h"
};

const int num_species = sizeof( SpeciesInfo ) / sizeof( Species_t );

const Species_t *FindSpecies( std::string &name )
{
  for( int i=0; i<num_species; i++ )
    if( name.find( SpeciesInfo[i].name ) != std::string::npos )
      return SpeciesInfo + i;

  std::cerr << "Couldn't find " << name << ". Using " << SpeciesInfo[0].name << ".\n";

  return SpeciesInfo;
}

//------------------------------------------------------------------------------
//
// Fish_t definition
//
//------------------------------------------------------------------------------

struct Fish_t
{
  Fish_t( std::string &objfilename, Material TankMat, TextureSampler CausticTS,
          Context context, GeometryGroup inGG );

  std::vector<std::vector<float3> > AnimatedPoints;
  Transform          Tr;
  GeometryGroup      GG;
  Geometry           G;
  Buffer             VB;
  optix::Aabb  aabb;
  float3             Pos, Vel, Target;
  int                phase_num;
  int                frames_toward_target; // Number of frames since Target was chosen
  RTsize             num_verts;
  const Species_t *  Species;
  bool               ownGG; // True if this Fish is responsible for updating the shared GG for this species each frame

  const static int   ANIM_STEPS;

  void initAnimation( );
  void updateGeometry( );

private:
  float3 swimPoint( float phase, float3 &P );
  void updatePos( );
};

const int Fish_t::ANIM_STEPS = 31;

Fish_t::Fish_t( std::string &objfilename, Material TankMat, TextureSampler CausticTS,
                Context context, GeometryGroup inGG = GeometryGroup( 0 ) )
  : GG( inGG ), ownGG( false )
{
  static bool printedPermissions = false;
  if( !printedPermissions ) {
    if( objfilename.find( "Fish_OBJ_PPM" ) != std::string::npos ) {
      std::cout << "\nModels and textures copyright Toru Miyazawa, Toucan Corporation. Used by permission.\n";
      std::cout << "http://toucan.web.infoseek.co.jp/3DCG/3ds/FishModelsE.html\n\n";
      printedPermissions = true;
    }
  }

  Species = FindSpecies( objfilename );

  // std::cerr << "Found name: " << Species->name << '\n';

  GeometryInstance GI;

  if( GG.get( ) == 0 ) {
    std::cerr << "Loading " << objfilename << '\n';
    ownGG = true;

    GG = context->createGeometryGroup( );
    ObjLoader FishLoader( objfilename.c_str(), context, GG, TankMat, true );

    float m[16] = {
      0,-1,0,0,
      0,0,1,0,
      -1,0,0,0,
      0,0,0,1
    };
    optix::Matrix4x4 Rot( m );
    optix::Matrix4x4 XForm = Rot;
    XForm = optix::Matrix4x4::scale( make_float3( 1.0f/Species->sizeInObj ) ) * XForm;
    XForm = optix::Matrix4x4::translate( make_float3( 0, 0, Species->centerOffset ) ) * XForm;
    XForm = optix::Matrix4x4::scale( make_float3( Species->lengthMeters ) ) * XForm;
    FishLoader.load( XForm );
    aabb = FishLoader.getSceneBBox( );
    //std::cerr << "AABB: " << aabb.m_min << aabb.m_max << '\n';

    // Set the material properties that differ between the fish and the other scene elements
    for (unsigned int i = 0; i < GG->getChildCount(); ++i)
    {
      GI = GG->getChild( i );
      GI["caustic_map"]->setTextureSampler( CausticTS );
      GI["diffuse_map_scale"]->setFloat( 1.0f );
      GI["emission_color"]->setFloat( 0 );
      GI["Kr"]->setFloat( 0 );
    }

    // Select an AS builder that allows refit, unlike the ObjLoader default.
    Acceleration AS = GG->getAcceleration( );
    AS->setBuilder( "Bvh" );
    AS->setProperty( "refit", "1" );
  } else {
    //std::cerr << "Instancing " << objfilename << '\n';
    GI = GG->getChild( 0 );
  }

  G = GI->getGeometry( );
  VB = G["vertex_buffer"]->getBuffer( );
  VB->getSize( num_verts ); // Query number of vertices in the buffer

  Tr = context->createTransform( );
  Tr->setChild( GG );
}

// Calculation of the rotation angle of the current model point depending on the z-coordinate of the point
inline float3 Fish_t::swimPoint( float phaseDeg, float3 &P )
{
  // z >= 0 is the front of the model, and the movement of the head can be reduced.
  // angle = phase factor * damping * full amplitude.

  const float DtoR = float( M_PI / 180.0 );

  float wave_scale = ( P.z >= 0 ) ? P.z / aabb.m_max.z : -P.z / aabb.m_min.z;
  float wave_len = ( P.z >= 0 ) ? Species->headWaveLen : Species->tailWaveLen;
  float damping = wave_scale; // Scales the wave by distance from origin;
  float rotAngRad = sin( ( -phaseDeg - wave_len * wave_scale ) * DtoR ) * damping * Species->maxAmplRad;

  // Rotate about +Y
  return make_float3( cos( rotAngRad ) * P.x + sin( rotAngRad ) * P.z, P.y, -sin( rotAngRad ) * P.x + cos( rotAngRad ) * P.z );
}

// Precompute all the animation frames
void Fish_t::initAnimation( )
{
  if( ownGG ) {
    // Get a pointer to the initial, non-deformed buffer data
    float3* Verts = ( float3* )VB->map( );

    AnimatedPoints.resize( ANIM_STEPS );

    for ( int ph = 0; ph < ANIM_STEPS; ph++ ) { //
      // Compute this frame of the animation
      float phaseDeg = 360.0f * ph / float( ANIM_STEPS - 1 ); // The phase in degrees
      for ( size_t v = 0; v < num_verts; v++ ) {
        float3 pResult = swimPoint( phaseDeg, Verts[v] );

        AnimatedPoints[ph].push_back( pResult );
      } // v
    }   // ph

    // Unmap buffer
    VB->unmap( );
  }

  Pos = MakeDRand( TargetBox.m_min, TargetBox.m_max );
  Vel = make_float3( 0,0,1 );
  Target = MakeDRand( TargetBox.m_min, TargetBox.m_max );

  phase_num = rand( ) % ANIM_STEPS;
  frames_toward_target = 10000;
}

void Fish_t::updatePos( )
{
  // Update position
#if 0
  Pos = make_float3( 0,50,0 );
  Vel = make_float3( 1,0,0 );
  Target = Pos + Vel * 300.0f;
  return;
#endif
  Pos += Vel;

  // Update velocity
  float range = length( Target - Pos );
  float3 TargetDir = normalize( Target - Pos );
  Vel = normalize( Vel );
  float oldYVel = Vel.y;

  float ang = acos( dot( Vel, TargetDir ) );
  if( ang > 1e-5 ) {
    float3 Axis = normalize( cross( Vel, TargetDir ) );
    const float max_ang = 0.08f;
    ang = fminf( ang, max_ang );
    optix::Matrix4x4 Rot = optix::Matrix4x4::rotate( ang, Axis );
    Vel = make_float3( Rot * make_float4( Vel,1 ) );
  }

  // Prevent them from pitching too quickly
  const float max_dpitch = 0.01f;
  float new_pitch = 0;
  if( Vel.y > oldYVel+max_dpitch ) new_pitch = oldYVel+max_dpitch;
  if( Vel.y < oldYVel-max_dpitch ) new_pitch = oldYVel-max_dpitch;

  if( new_pitch ) {
    Vel.y = 0;
    Vel = normalize( Vel );
    Vel.y = new_pitch;
    Vel = normalize( Vel );
  }

  Vel *= Species->speed * Species->lengthMeters;

  // Update target
  // meters
  if( range > 1.0f && frames_toward_target++ < 40 )
    return;

  // Choose new random target
  Target = MakeDRand( TargetBox.m_min, TargetBox.m_max );
  // std::cerr << Target << '\n';
  frames_toward_target = 0;
}

void Fish_t::updateGeometry( )
{
  if( ownGG ) {
    // We have precomputed every animation pose. Here we copy the relevant one into place.
    assert( AnimatedPoints[phase_num].size( ) == num_verts );
    float3* Verts = ( float3* )VB->map( );
    memcpy( Verts, &( AnimatedPoints[phase_num][0] ), num_verts * sizeof( float3 ) );
    VB->unmap( );

    phase_num++;
    if( phase_num >= ANIM_STEPS ) phase_num = 0;

    // Mark the accel structure and geometry as dirty so they will be rebuilt.
    G->markDirty( );
    GG->getAcceleration( )->markDirty( );
  }

  updatePos( );

  // Transform the fish's position and orientation
  float3 forward = normalize( Vel );
  float3 side = make_float3( forward.z, 0, -forward.x );
  side = normalize( side );
  float3 up = cross( forward, side );

  optix::Matrix<4,4> Rotate;
  Rotate.setCol( 0u, make_float4( side, 0 ) );
  Rotate.setCol( 1u, make_float4( up, 0 ) );
  Rotate.setCol( 2u, make_float4( forward, 0 ) );
  Rotate.setCol( 3u, make_float4( 0,0,0,1 ) );

  optix::Matrix<4,4> Translate = optix::Matrix<4,4>::translate( Pos );
  optix::Matrix<4,4> Comp = Translate * Rotate;

  Tr->setMatrix( false, Comp.getData( ), 0 );
}

//------------------------------------------------------------------------------
//
// Bubbles definition
//
//------------------------------------------------------------------------------

class Bubbles_t
{
public:
    Bubbles_t( const float3 &bubble_origin, Material TankMat, TextureSampler BlankTS, Context context )
        : m_bubble_origin( bubble_origin ),
        m_sphere_rad( SCENE_BOX.extent( 1 )*0.01f ),
        m_num_bubbles( 128 )
    {
        BB = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, m_num_bubbles );
        Buffer MB = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, m_num_bubbles );

        unsigned int *sphere_mats = reinterpret_cast<unsigned int *>( MB->map( ) );
        float4 *spheres = reinterpret_cast<float4 *>( BB->map( ) );

        for( size_t i=0; i<m_num_bubbles; i++ ) {
            float3 B = m_bubble_origin;
            B.y += SCENE_BOX.extent( 1 ) * i / float( m_num_bubbles );
            spheres[i] = make_float4( B, m_sphere_rad );
            sphere_mats[i] = 0;
        }

        MB->unmap( );
        BB->unmap( );

        std::string ptx_path = SampleScene::ptxpath( "swimmingShark", "sphere_list.cu" );
        G = context->createGeometry( );
        G->setBoundingBoxProgram( context->createProgramFromPTXFile( ptx_path, "bounds" ) );
        G->setIntersectionProgram( context->createProgramFromPTXFile( ptx_path, "intersect" ) );
        G->setPrimitiveCount( static_cast<unsigned int>(m_num_bubbles) );

        G["sphere_buffer"]->setBuffer( BB );
        G["material_buffer"]->setBuffer( MB );

        GeometryInstance GI = context->createGeometryInstance( G, &TankMat, &TankMat+1 );
        GI["caustic_map"]->setTextureSampler( BlankTS );
        GI["diffuse_map"]->setTextureSampler( BlankTS );
        GI["diffuse_map_scale"]->setFloat( 1.0f );
        GI["emission_color"]->setFloat( 0 );
        GI["Kr"]->setFloat( 1.0f );

        GG = context->createGeometryGroup( );
        GG->setChildCount( 1u );
        GG->setChild( 0, GI );
        GG->setAcceleration( context->createAcceleration( "MedianBvh", "Bvh" ) );
    }

    void updateGeometry( )
    {
        float4 *spheres = reinterpret_cast<float4 *>( BB->map( ) );

        for( size_t i=0; i<m_num_bubbles; i++ ) {
            const float speed = SCENE_BOX.extent( 1 )*0.01f;
            float3 P = make_float3( spheres[i] );
            P += MakeDRand( make_float3( -speed, speed, -speed ), make_float3( speed, speed*0.9f, speed ) );
            if( P.y > SCENE_BOX.m_max.y ) P.y = m_bubble_origin.y;
            spheres[i] = make_float4( P, spheres[i].w );
        }

        BB->unmap( );

        // Mark the accel structure and geometry as dirty so they will be rebuilt.
        //G->markDirty( );
        GG->getAcceleration( )->markDirty( );
    }

    GeometryGroup GG;

private:
    Geometry G;
    Buffer BB;

    float3 m_bubble_origin;
    float  m_sphere_rad;
    size_t m_num_bubbles;
};

//------------------------------------------------------------------------------
//
// TankScene definition
//
//------------------------------------------------------------------------------

class TankScene : public SampleScene
{
public:
  TankScene( const std::string& objfilename, const std::string& objpath, const std::string& texturepath,
    int num_species_to_load, int fish_per_species )
  : m_objfilename( objfilename ), m_objpath( objpath ), m_texturepath( texturepath ),
    m_num_species_to_load( num_species_to_load ), m_fish_per_species( fish_per_species )
  { }

  ~TankScene( )
  { }

  virtual void   initScene( InitialCameraData& camera_data );
  virtual void   trace( const RayGenCameraData& camera_data );
  virtual Buffer getOutputBuffer( );
  virtual bool   keyPressed( unsigned char key, int x, int y );

private:
  void updateGeometry( );
  void createGeometry( );
  Geometry createHeightField( );
  Geometry createParallelogram( Program pbounds, Program pisect, float3 anchor, float3 v1, float3 v2 );
  void createGroundData( );

  Group         TopGroup;
  Buffer        GroundBuf;
  Material      TankMat;
  TextureSampler CausticTS;

  std::vector<Fish_t *> Fish;
  Bubbles_t *   Bubbles;

  std::string   m_objfilename, m_objpath, m_texturepath;
  float         m_scene_epsilon;
  float         m_ground_ymin, m_ground_ymax;
  int           m_num_species_to_load, m_fish_per_species;
  bool          m_animate;

  const static int         WIDTH;
  const static int         HEIGHT;
  const static int         GROUND_WID;
  const static int         GROUND_HGT;
  const static float3      WATER_BLUE;
};

const int TankScene::WIDTH  = 800;
const int TankScene::HEIGHT = 600;
const int TankScene::GROUND_WID = 15;
const int TankScene::GROUND_HGT = 15;
const float3 TankScene::WATER_BLUE = make_float3( 0.192f, 0.498f, 0.792f );

void TankScene::createGeometry( )
{
  // Make fish models
  if( !m_objfilename.empty( ) ) {
    Fish.push_back( new Fish_t( m_objfilename, TankMat, CausticTS, m_context ) );
    for( int f=1; f<m_fish_per_species; f++ ) {
      Fish.push_back( new Fish_t( m_objfilename, TankMat, CausticTS, m_context, Fish.back( )->GG ) );
    }
  }

  // We want a jittered sampling of fish if the array is sorted by size.  Divide
  // num_species by m_num_species_to_load to get the bins to sample from, then sample
  // within that bin.
  float species_per_bin = static_cast<float>(num_species)/m_num_species_to_load;
  
  for( int s=0; s<m_num_species_to_load; s++ ) {
    int sp = static_cast<int>(species_per_bin * ( s + FRand()));
    if (sp >= num_species) sp = num_species-1;
    std::string fname = m_objpath + SpeciesInfo[sp].name;

    Fish.push_back( new Fish_t( fname, TankMat, CausticTS, m_context ) );
    for( int f=1; f<m_fish_per_species; f++ ) {
      Fish.push_back( new Fish_t( fname, TankMat, CausticTS, m_context, Fish.back( )->GG ) );
    }
  }

  // Make tank
  std::cerr << "Initializing tank ...";

  // Geometry
  std::string ptx_path = SampleScene::ptxpath( "swimmingShark", "parallelogram.cu" );
  Program pbounds = m_context->createProgramFromPTXFile( ptx_path, "bounds" );
  Program pisect = m_context->createProgramFromPTXFile( ptx_path, "intersect" );

  Geometry WaterSurfaceG = createParallelogram( pbounds, pisect,
    make_float3( SCENE_BOX.m_min.x, SCENE_BOX.m_max.y, SCENE_BOX.m_min.z ),
    make_float3( SCENE_BOX.extent( 0 ), 0, 0 ),
    make_float3( 0, 0, SCENE_BOX.extent( 2 ) ) );

  TextureSampler ConstTS = loadTexture( m_context, "", WATER_BLUE );

  // Water Surface
  GeometryInstance WaterSurfaceGI = m_context->createGeometryInstance( WaterSurfaceG, &TankMat, &TankMat+1 );
  WaterSurfaceGI["caustic_map"]->setTextureSampler( ConstTS );
  WaterSurfaceGI["diffuse_map"]->setTextureSampler( CausticTS );
  WaterSurfaceGI["diffuse_map_scale"]->setFloat( 12.0f );
  WaterSurfaceGI["emission_color"]->setFloat( WATER_BLUE*0.7f );
  WaterSurfaceGI["Kr"]->setFloat( 1.0f );

  Geometry GroundG = createHeightField( );

  GeometryInstance GroundGI = m_context->createGeometryInstance( GroundG, &TankMat, &TankMat+1 );
  GroundGI["caustic_map"]->setTextureSampler( ConstTS );
  GroundGI["diffuse_map"]->setTextureSampler( loadTexture( m_context, m_texturepath + "/sand.ppm", make_float3( 1, 1, 0 ) ) );
  GroundGI["diffuse_map_scale"]->setFloat( 18.0f );
  GroundGI["emission_color"]->setFloat( WATER_BLUE*0.4f );
  GroundGI["Kr"]->setFloat( 0 );

  GeometryGroup SceneGG = m_context->createGeometryGroup( );
  SceneGG->setAcceleration( m_context->createAcceleration( "Sbvh","Bvh" ) );
  SceneGG->setChildCount( static_cast<unsigned int>( 2 ) );

  SceneGG->setChild( 0, GroundGI );
  SceneGG->setChild( 1, WaterSurfaceGI );

  // Make bubbles
  Bubbles = new Bubbles_t( make_float3( SCENE_BOX.m_max.x*0.3f, SCENE_BOX.m_min.y, SCENE_BOX.m_max.z*0.5f ), TankMat, ConstTS, m_context );

  unsigned int numFish = static_cast<unsigned int>(Fish.size());

  // Make overall group
  TopGroup = m_context->createGroup( );
  TopGroup->setChildCount( numFish + 1 + ( Bubbles ? 1 : 0 ) ); // Each fish plus the tank plus the bubbles

  for( unsigned int i=0; i<numFish; i++ )
    TopGroup->setChild( i, Fish[i]->Tr );
  TopGroup->setChild( numFish, SceneGG );
  if( Bubbles ) TopGroup->setChild( numFish + 1, Bubbles->GG );
  TopGroup->setAcceleration( m_context->createAcceleration( "MedianBvh","Bvh" ) );

  m_context["top_object"]->set( TopGroup );
  m_context["top_shadower"]->set( TopGroup );

  std::cerr << "finished." << std::endl;
}

void TankScene::initScene( InitialCameraData& camera_data )
{
  // Setup context
  m_context->setRayTypeCount( 2 );
  m_context->setEntryPointCount( 1 );
  m_context->setStackSize( 1350 );
  m_context->setPrintEnabled( false );
  m_context->setPrintBufferSize( 1024 );
  m_context->setPrintLaunchIndex( 400,300 );

  m_context[ "radiance_ray_type" ]->setUint( 0u );
  m_context[ "shadow_ray_type" ]->setUint( 1u );
  m_scene_epsilon = 1.e-3f;
  m_context[ "scene_epsilon" ]->setFloat( m_scene_epsilon );
  m_context[ "max_depth" ]->setInt( 3 );

  // Output buffer
  m_context["output_buffer"]->set( createOutputBuffer( RT_FORMAT_UNSIGNED_BYTE4, WIDTH, HEIGHT ) );

  // Ray generation program
  std::string ptx_path = ptxpath( "swimmingShark", "pinhole_camera.cu" );
  Program ray_gen_program = m_context->createProgramFromPTXFile( ptx_path, "pinhole_camera" );
  m_context->setRayGenerationProgram( 0, ray_gen_program );

  // Exception / miss programs
  m_context->setExceptionProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );
  m_context[ "bad_color" ]->setFloat( 0, 1.0f, 0 );

  m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptxpath( "swimmingShark", "constantbg.cu" ), "miss" ) );
  m_context[ "bg_color" ]->setFloat( WATER_BLUE );

  // Set up the material. Some uses will override certain parameters.
  TankMat = m_context->createMaterial( );
  TankMat->setClosestHitProgram( 0, m_context->createProgramFromPTXFile( ptxpath( "swimmingShark", "tank_material.cu" ), "closest_hit_radiance" ) );
  TankMat->setAnyHitProgram( 1, m_context->createProgramFromPTXFile( ptxpath( "swimmingShark", "tank_material.cu" ), "any_hit_shadow" ) );

  TankMat["ambient_light_color"]->setFloat( WATER_BLUE );
  TankMat["attenuation_color"]->setFloat( WATER_BLUE );
  TankMat["attenuation_density"]->setFloat( -0.045f ); // Must be < 0.
  TankMat["caustic_light_color"]->setFloat( 1.6f, 1.6f, 1.6f );
  TankMat["caustic_map_scale"]->setFloat( 0.3f );
  TankMat["light_dir"]->setFloat( 0, 1.0f, 0 );

  CausticTS = loadTexture( m_context, m_texturepath + "/caustic.ppm", make_float3( 1, 0, 0 ) );

  // Set up geometry
  createGeometry( );

  // Set up camera
  float3 eye = make_float3( 0.0001f, SCENE_BOX.center( 1 ), SCENE_BOX.m_max.z + SCENE_BOX.m_max.y * 0.9f );
  float3 lookat = SCENE_BOX.center( );

  camera_data = InitialCameraData(
    eye,                       // eye
    lookat,                    // lookat
    make_float3( 0, 1.0f, 0 ), // up
    50.0f );                   // vfov

  // Declare camera variables.  The values do not matter, they will be overwritten in trace.
  m_context["eye"]->setFloat( make_float3( 0, 0, 0 ) );
  m_context["U"]->setFloat( make_float3( 0, 0, 0 ) );
  m_context["V"]->setFloat( make_float3( 0, 0, 0 ) );
  m_context["W"]->setFloat( make_float3( 0, 0, 0 ) );

  for( size_t i=0; i<Fish.size( ); i++ )
    Fish[i]->initAnimation( );
  m_animate = true;

  // Prepare to run
  m_context->validate( );
  m_context->compile( );
}

void TankScene::createGroundData( )
{
  int N = ( GROUND_WID+1 ) * ( GROUND_HGT+1 );
  GroundBuf = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT, GROUND_WID+1, GROUND_HGT+1 );
  float* p = reinterpret_cast<float*>( GroundBuf->map( ) );

  // Compute data range
  m_ground_ymin = 1e9;
  m_ground_ymax = -1e9;

  const float ground_scale = 0.8f; // meters

  for( int i = 0; i<N; i++ ) {
    p[i] = FRand( ground_scale );
    m_ground_ymin = fminf( m_ground_ymin, p[i] );
    m_ground_ymax = fmaxf( m_ground_ymax, p[i] );
  }

  m_ground_ymin -= 1.e-6f;
  m_ground_ymax += 1.e-6f;
  GroundBuf->unmap( );
}

Geometry TankScene::createHeightField( )
{
  createGroundData( );

  Geometry HgtFldG = m_context->createGeometry( );
  HgtFldG->setPrimitiveCount( 1u );

  HgtFldG->setBoundingBoxProgram( m_context->createProgramFromPTXFile( ptxpath( "swimmingShark", "heightfield.cu" ), "bounds" ) );
  HgtFldG->setIntersectionProgram( m_context->createProgramFromPTXFile( ptxpath( "swimmingShark", "heightfield.cu" ), "intersect" ) );
  float3 min_corner = make_float3( SCENE_BOX.m_min.x, m_ground_ymin, SCENE_BOX.m_min.z );
  float3 max_corner = make_float3( SCENE_BOX.m_max.x, m_ground_ymax, SCENE_BOX.m_max.z );
  RTsize nx, nz;
  GroundBuf->getSize( nx, nz );

  // If buffer is nx by nz, we have nx-1 by nz-1 cells;
  float3 cellsize = ( max_corner - min_corner ) / ( make_float3( static_cast<float>( nx-1 ), 1.0f, static_cast<float>( nz-1 ) ) );
  cellsize.y = 1;
  float3 inv_cellsize = make_float3( 1 )/cellsize;
  HgtFldG["boxmin"]->setFloat( min_corner );
  HgtFldG["boxmax"]->setFloat( max_corner );
  HgtFldG["cellsize"]->setFloat( cellsize );
  HgtFldG["inv_cellsize"]->setFloat( inv_cellsize );
  HgtFldG["data"]->setBuffer( GroundBuf );

  return HgtFldG;
}

Geometry TankScene::createParallelogram( Program pbounds, Program pisect, float3 anchor, float3 v1, float3 v2 )
{
  Geometry parallelogram = m_context->createGeometry( );
  parallelogram->setPrimitiveCount( 1u );
  parallelogram->setBoundingBoxProgram( pbounds );
  parallelogram->setIntersectionProgram( pisect );

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

  return parallelogram;
}

void TankScene::updateGeometry( )
{
  if( !m_animate )
    return;

  for( size_t i=0; i<Fish.size( ); i++ ) {
    Fish[i]->updateGeometry( );

    // Implement schooling by copying the target position of another fish of the same species
    if( i>0 && !Fish[i]->ownGG && ( rand( ) % 3 == 0 ) )
      Fish[i]->Target = Fish[i-1]->Target;
  }

  if( Bubbles ) Bubbles->updateGeometry( );

  TopGroup->getAcceleration( )->markDirty( );
}

void TankScene::trace( const RayGenCameraData& camera_data )
{
  m_context["eye"]->setFloat( camera_data.eye );
  m_context["U"]->setFloat( camera_data.U );
  m_context["V"]->setFloat( camera_data.V );
  m_context["W"]->setFloat( camera_data.W );

  Buffer buffer = m_context["output_buffer"]->getBuffer( );
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );

  //Update the vertex positions before rendering
  updateGeometry( );

  m_context->launch( 0, buffer_width, buffer_height );
}

Buffer TankScene::getOutputBuffer( )
{
  return m_context["output_buffer"]->getBuffer( );
}

bool TankScene::keyPressed( unsigned char key, int x, int y )
{
  switch ( key ) {
  case 'e':
    m_scene_epsilon /= 10.0f;
    std::cerr << "scene_epsilon: " << m_scene_epsilon << std::endl;
    m_context[ "scene_epsilon" ]->setFloat( m_scene_epsilon );
    return true;
  case 'E':
    m_scene_epsilon *= 10.0f;
    std::cerr << "scene_epsilon: " << m_scene_epsilon << std::endl;
    m_context[ "scene_epsilon" ]->setFloat( m_scene_epsilon );
    return true;
  case 'a':
    m_animate = !m_animate;
    return true;
  }
  return false;
}

//------------------------------------------------------------------------------
//
//  main driver
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                               Print this usage message\n"
    << "  -o  | --obj <obj_file>                     Specify .OBJ model to be rendered\n"
    << "  -f  | --species <num>                      Specify the number of species to load\n"
    << "  -n  | --school-size <num>                  Specify the number of fish of each species to load\n"
    << "  -P  | --objpath <obj_path>                 Specify path to the OBJ models\n"
    << "  -t  | --texpath <tex_path>                 Specify path to the sand, water and caustic textures\n"
    << std::endl;
  GLUTDisplay::printUsage();

  std::cerr
    << "App keystrokes:\n"
    << "  e Decrease scene epsilon size (used for shadow ray offset)\n"
    << "  E Increase scene epsilon size (used for shadow ray offset)\n"
    << "  a Toggle animation\n"
    << std::endl;

  if ( doExit ) exit(1);
}

int main( int argc, char** argv )
{
  GLUTDisplay::init( argc, argv );

  // Process command line options
  std::string objfilename, objpath, texturepath;
  int num_species_to_load = 0, fish_per_species = 7;
  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "--help" || arg == "-h" ) {
      printUsageAndExit( argv[0] );
    } else if ( arg == "--species" || arg == "-f" ) {
      if ( i == argc-1 ) printUsageAndExit( argv[0] );
      num_species_to_load = atoi( argv[++i] );
    } else if ( arg == "--school-size" || arg == "-n" ) {
      if ( i == argc-1 ) printUsageAndExit( argv[0] );
      fish_per_species = atoi( argv[++i] );
    } else if ( arg == "--objpath" || arg == "-P" ) {
      if ( i == argc-1 ) printUsageAndExit( argv[0] );
      objpath = argv[++i];
    } else if ( arg == "--obj" || arg == "-o" ) {
      if ( i == argc-1 ) printUsageAndExit( argv[0] );
      objfilename = argv[++i];
    } else if ( arg == "--texpath" || arg == "-t" ) {
      if ( i == argc-1 ) printUsageAndExit( argv[0] );
      texturepath = argv[++i];
    } else {
      std::cerr << "Unknown option: '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

  if( objfilename.empty( ) && num_species_to_load==0 )
    num_species_to_load = 7;
  if( objpath.empty( ) ) {
    objpath = std::string( sutilSamplesDir() ) + "/swimmingShark/Fish_OBJ_PPM/";
  }
  if( texturepath.empty( ) ) {
    texturepath = std::string( sutilSamplesDir() ) + "/swimmingShark/";
  }

  if( !GLUTDisplay::isBenchmark( ) ) {
    // With Unix rand( ), a small magnitude change in input seed yields a small change in the first random number. Duh!
    unsigned int tim = static_cast<unsigned int>( time( 0 ) );
    unsigned int tim2 = ( ( tim & 0xff ) << 24 ) | ( ( tim & 0xff00 ) << 8 ) | ( ( tim & 0xff0000 ) >> 8 ) | ( ( tim & 0xff000000 ) >> 24 );
    srand( tim2 );
  }

  try {
    TankScene scene( objfilename, objpath, texturepath, num_species_to_load, fish_per_species );
    GLUTDisplay::run( "Swimming Shark", &scene, GLUTDisplay::CDAnimated );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString( ).c_str( ) );
    exit( 1 );
  }

  return 0;
}

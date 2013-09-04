
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
#include <optixu/optixu_math_namespace.h>
#include "ppm.h"
#include "path_tracer.h"
#include "random.h"

using namespace optix;

//
// Scene wide variables
//
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(rtObject,      top_object, , );


//
// Ray generation program
//
rtBuffer<PhotonRecord, 1>        ppass_output_buffer;
rtBuffer<uint2, 2>               photon_rnd_seeds;
rtDeclareVariable(uint,          max_depth, , );
rtDeclareVariable(uint,          max_photon_count, , );
rtDeclareVariable(PPMLight,      light , , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );


static __device__ __inline__ float2 rnd_from_uint2( uint2& prev )
{
  return make_float2(rnd(prev.x), rnd(prev.y));
}

static __device__ __inline__ void generateAreaLightPhoton( const PPMLight& light, const float2& d_sample, float3& o, float3& d)
{
  // Choose a random position on light
  o = light.anchor + 0.5f * ( light.v1 + light.v2);
  
  // Choose a random direction from light
  float3 U, V, W;
  createONB( light.direction, U, V, W);
  sampleUnitHemisphere( d_sample, U, V, W, d );
}

static __device__ __inline__ void generateSpotLightPhoton( const PPMLight& light, const float2& d_sample, float3& o, float3& d)
{
  o = light.position;

/*
  // Choose random dir by sampling disk of radius light.radius and projecting up to unit hemisphere
  float r = atanf( light.radius) * sqrtf( d_sample.x );
  float theta = 2.0f * M_PIf * d_sample.y;

  float x = r*cosf( theta );
  float y = r*sinf( theta );
  float z = sqrtf( fmaxf( 0.0f, 1.0f - x*x - y*y ) );
*/

  // Choose random dir by sampling disk of radius light.radius and projecting up to unit hemisphere
  float2 square_sample = d_sample; 
  mapToDisk( square_sample );
  square_sample = square_sample * atanf( light.radius );
  float x = square_sample.x;
  float y = square_sample.y;
  float z = sqrtf( fmaxf( 0.0f, 1.0f - x*x - y*y ) );

  // Now transform into light space
  float3 U, V, W;
  createONB(light.direction, U, V, W);
  d =  x*U + y*V + z*W;
}


RT_PROGRAM void ppass_camera()
{
  size_t2 size     = photon_rnd_seeds.size();
  uint    pm_index = (launch_index.y * size.x + launch_index.x) * max_photon_count;
  uint2   seed     = photon_rnd_seeds[launch_index]; // No need to reset since we dont reuse this seed

  float2 direction_sample = make_float2(
      ( static_cast<float>( launch_index.x ) + rnd( seed.x ) ) / static_cast<float>( size.x ),
      ( static_cast<float>( launch_index.y ) + rnd( seed.y ) ) / static_cast<float>( size.y ) );
  float3 ray_origin, ray_direction;
  if( light.is_area_light ) {
    generateAreaLightPhoton( light, direction_sample, ray_origin, ray_direction );
  } else {
    generateSpotLightPhoton( light, direction_sample, ray_origin, ray_direction );
  }

  optix::Ray ray(ray_origin, ray_direction, ppass_and_gather_ray_type, scene_epsilon );

  // Initialize our photons
  for(unsigned int i = 0; i < max_photon_count; ++i) {
    ppass_output_buffer[i+pm_index].energy = make_float3(0.0f);
  }

  PhotonPRD prd;
  //  rec.ray_dir = ray_direction; // set in ppass_closest_hit
  prd.energy = light.power;
  prd.sample = seed;
  prd.pm_index = pm_index;
  prd.num_deposits = 0;
  prd.ray_depth = 0;
  rtTrace( top_object, ray, prd );
}

//
// Closest hit material
//
rtDeclareVariable(float3,  Ks, , );
rtDeclareVariable(float3,  Kd, , );
rtDeclareVariable(float3,  emitted, , );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PhotonPRD, hit_record, rtPayload, );

RT_PROGRAM void ppass_closest_hit()
{
  // Check if this is a light source
  float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 ffnormal     = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

  float3 hit_point = ray.origin + t_hit*ray.direction;
  float3 new_ray_dir;

  if( fmaxf( Kd ) > 0.0f ) {
    // We hit a diffuse surface; record hit if it has bounced at least once
    if( hit_record.ray_depth > 0 ) {
      PhotonRecord& rec = ppass_output_buffer[hit_record.pm_index + hit_record.num_deposits];
      rec.position = hit_point;
      rec.normal = ffnormal;
      rec.ray_dir = ray.direction;
      rec.energy = hit_record.energy;
      hit_record.num_deposits++;
    }

    hit_record.energy = Kd * hit_record.energy; 
    float3 U, V, W;
    createONB(ffnormal, U, V, W);
    sampleUnitHemisphere(rnd_from_uint2(hit_record.sample), U, V, W, new_ray_dir);

  } else {
    hit_record.energy = Ks * hit_record.energy;
    // Make reflection ray
    new_ray_dir = reflect( ray.direction, ffnormal );
  }

  hit_record.ray_depth++;
  if ( hit_record.num_deposits >= max_photon_count || hit_record.ray_depth >= max_depth)
    return;

  optix::Ray new_ray( hit_point, new_ray_dir, ppass_and_gather_ray_type, scene_epsilon );
  rtTrace(top_object, new_ray, hit_record);
}


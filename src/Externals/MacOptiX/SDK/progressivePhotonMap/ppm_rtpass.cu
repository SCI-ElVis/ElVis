
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


using namespace optix;

//
// Scene wide variables
//
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(rtObject,      top_object, , );


//
// Ray generation program
//
rtBuffer<HitRecord, 2>           rtpass_output_buffer;
rtBuffer<uint2, 2>               image_rnd_seeds;
rtDeclareVariable(float,         rtpass_default_radius2, , );
rtDeclareVariable(float3,        rtpass_eye, , );
rtDeclareVariable(float3,        rtpass_U, , );
rtDeclareVariable(float3,        rtpass_V, , );
rtDeclareVariable(float3,        rtpass_W, , );
rtDeclareVariable(uint2,      launch_index, rtLaunchIndex, );


RT_PROGRAM void rtpass_camera()
{
  float2 screen = make_float2( rtpass_output_buffer.size() );
  /*
  uint   seed   = image_rnd_seeds[index];                       // If we start writing into this buffer here we will
  float2 sample = make_float2( rnd(seed.x), rnd(seed.y) );      // need to make it an INPUT_OUTPUT buffer.  For now it
  image_rnd_seeds[index] = seed;                                // is just INPUT
  */
  float2 sample = make_float2( 0.5f, 0.5f ); 

  float2 d = ( make_float2(launch_index) + sample ) / screen * 2.0f - 1.0f;
  float3 ray_origin = rtpass_eye;
  float3 ray_direction = normalize(d.x*rtpass_U + d.y*rtpass_V + rtpass_W);
  
  optix::Ray ray(ray_origin, ray_direction, rtpass_ray_type, scene_epsilon);

  HitPRD prd;
  // rec.ray_dir = ray_direction; // set in rtpass_closest_hit
  prd.attenuation = make_float3( 1.0f );
  prd.ray_depth   = 0u; 
  rtTrace( top_object, ray, prd );
}

// 
// Closest hit material
// 
rtDeclareVariable(float3,  Ks, , );
rtDeclareVariable(float3,  Kd, , );
rtDeclareVariable(float3,  grid_color, , );
rtDeclareVariable(uint,    use_grid, , );
rtDeclareVariable(float3,  emitted, , );

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, ); 

rtDeclareVariable(HitPRD, hit_prd, rtPayload, );

rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );

RT_PROGRAM void rtpass_closest_hit()
{
  // Check if this is a light source
  if( fmaxf( emitted ) > 0.0f ) {
    HitRecord& rec = rtpass_output_buffer[ launch_index ];
    rec.attenuated_Kd = emitted*hit_prd.attenuation; 
    rec.flags = 0u;
    return;
  }

  float3 direction    = ray.direction;
  float3 origin       = ray.origin;
  float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 ffnormal     = faceforward( world_shading_normal, -direction, world_geometric_normal );
  float3 hit_point    = origin + t_hit*direction;

  if( fmaxf( Kd ) > 0.0f ) { 
    // We hit a diffuse surface; record hit and return
    HitRecord rec;
    rec.position = hit_point; 
    rec.normal = ffnormal;
    if( !use_grid ) {
      rec.attenuated_Kd = Kd * hit_prd.attenuation;
    } else {
      float grid_size = 50.0f; 
      float line_size = 2.0f; 
      float xx = ( hit_point.x + 1025.0f ) / grid_size;
      xx = ( xx - static_cast<float>( static_cast<int>( xx ) ) ) * grid_size;
      float zz = ( hit_point.z + 1025.0f ) / grid_size;
      zz = ( zz - static_cast<float>( static_cast<int>( zz ) ) ) * grid_size;
      if( xx < line_size  || zz < line_size )
        rec.attenuated_Kd = grid_color * hit_prd.attenuation;
      else
        rec.attenuated_Kd = Kd * hit_prd.attenuation;
    }
    rec.flags = PPM_HIT;

    rec.radius2 = rtpass_default_radius2;
    rec.photon_count = 0;
    rec.accum_atten = 0.0f;
    rec.flux = make_float3(0.0f, 0.0f, 0.0f);
    
    rtpass_output_buffer[launch_index] = rec;
  } else {
    // Make reflection ray
    hit_prd.attenuation = hit_prd.attenuation * Ks;
    hit_prd.ray_depth++;
    float3 R = reflect( direction, ffnormal );
    optix::Ray refl_ray( hit_point, R, rtpass_ray_type, scene_epsilon );
    rtTrace( top_object, refl_ray, hit_prd );
  }
}

//
// Miss program
//
rtTextureSampler<float4, 2> envmap;
RT_PROGRAM void rtpass_miss()
{
  float theta = atan2f( ray.direction.x, ray.direction.z );
  float phi   = M_PIf * 0.5f -  acosf( ray.direction.y );
  float u     = (theta + M_PIf) * (0.5f * M_1_PIf);
  float v     = 0.5f * ( 1.0f + sin(phi) );
  float3 result = make_float3(tex2D(envmap, u, v));

  HitRecord& rec = rtpass_output_buffer[launch_index];
  rec.flags = 0u;
  rec.attenuated_Kd = hit_prd.attenuation * result;
}

/*
rtDeclareVariable(float3, rtpass_bg_color, , );
RT_PROGRAM void rtpass_miss()
{
  HitPRD& prd = hit_prd.reference();
  uint2 index = make_uint2( launch_index.get() );
  HitRecord& rec = rtpass_output_buffer[index];

  rec.flags = 0u;
  rec.attenuated_Kd = prd.attenuation * rtpass_bg_color;
}
*/

//       
// Stack overflow program
//
rtDeclareVariable(float3, rtpass_bad_color, , );
RT_PROGRAM void rtpass_exception()
{
  HitRecord& rec = rtpass_output_buffer[launch_index];

  rec.flags = PPM_OVERFLOW;
  rec.attenuated_Kd = rtpass_bad_color;
}


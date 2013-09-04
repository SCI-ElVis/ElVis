
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

// Models and textures copyright Toru Miyazawa, Toucan Corporation. Used by permission.
// http://toucan.web.infoseek.co.jp/3DCG/3ds/FishModelsE.html

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "helpers.h"

using namespace optix;

struct PerRayData_radiance
{
  float3 result;
  float importance;
  int depth;
};

struct PerRayData_shadow
{
  float3 attenuation;
};

rtDeclareVariable( float,               t_hit,      rtIntersectionDistance, );
rtDeclareVariable( optix::Ray,          ray,        rtCurrentRay, );
rtDeclareVariable( PerRayData_radiance, prd,        rtPayload, );
rtDeclareVariable( PerRayData_shadow,   prd_shadow, rtPayload, );

rtDeclareVariable( float3,       geometric_normal, attribute geometric_normal, );
rtDeclareVariable( float3,       shading_normal,   attribute shading_normal, );
rtDeclareVariable( float3,       texcoord,         attribute texcoord, );

rtDeclareVariable( rtObject,     top_object, , );
rtDeclareVariable( rtObject,     top_shadower, , );
rtDeclareVariable( float,        scene_epsilon, , );
rtDeclareVariable( int,          max_depth, , );
rtDeclareVariable( unsigned int, radiance_ray_type, , );
rtDeclareVariable( unsigned int, shadow_ray_type, , );

rtDeclareVariable( float,        attenuation_density, , ); // Negative of fog density
rtDeclareVariable( float,        caustic_map_scale, , );
rtDeclareVariable( float,        diffuse_map_scale, , );
rtDeclareVariable( float,        Kr, , );
rtDeclareVariable( float3,       emission_color, , );
rtDeclareVariable( float3,       ambient_light_color, , );
rtDeclareVariable( float3,       caustic_light_color, , );
rtDeclareVariable( float3,       attenuation_color, , );
rtDeclareVariable( float3,       light_dir, , );

rtTextureSampler<float4, 2>      diffuse_map; // Corresponds to OBJ mtl params
rtTextureSampler<float, 2>       caustic_map; // A one-channel intensity map that's used instead of a light

RT_PROGRAM void any_hit_shadow( )
{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = make_float3( 0 );
  rtTerminateRay( );
}

RT_PROGRAM void closest_hit_radiance( )
{
  float3 direction              = ray.direction;
  float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 ffnormal               = faceforward( world_shading_normal, -direction, world_geometric_normal );

  float3 hit_point = ray.origin + t_hit * ray.direction;

  // ray.direction is unit length, so t_hit is the ray length.
  float atten_fac = exp(attenuation_density * t_hit);

  float3 Kd = make_float3( tex2D( diffuse_map, texcoord.x*diffuse_map_scale, texcoord.y*diffuse_map_scale ) );

  // Ambient lighting
  float3 result = Kd * ambient_light_color + emission_color;

  // Diffuse lighting
  float nDl = dot( ffnormal, light_dir );

  // If not back facing, light the hit point
  if( nDl > 0 ) {
    float3 caustic_coords = hit_point * caustic_map_scale;
    float intensity = tex2D( caustic_map, caustic_coords.x, caustic_coords.z ); // Hard-wired assumption that light travels in Y.

    float3 Lc = caustic_light_color * intensity;
    result += Kd * nDl * Lc;
  }

  if( Kr > 0 ) {
    // ray tree attenuation
    PerRayData_radiance new_prd;
    new_prd.importance = prd.importance * Kr * atten_fac;
    new_prd.depth = prd.depth + 1;

    // reflection ray
    if( new_prd.importance >= 0.01f && new_prd.depth <= max_depth ) {
      float3 R = reflect( ray.direction, ffnormal );
      optix::Ray refl_ray = optix::make_Ray( hit_point, R, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX );
      rtTrace( top_object, refl_ray, new_prd );
      result += Kr * new_prd.result;
    }
  }

  result = lerp( attenuation_color, result, atten_fac);

  // pass the color back up the tree
  prd.result = result;
}

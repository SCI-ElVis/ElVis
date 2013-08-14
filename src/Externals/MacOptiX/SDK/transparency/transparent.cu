
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
#include "commonStructs.h"
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


rtBuffer<BasicLight>                 lights;
rtDeclareVariable(int,               max_depth, , );
rtDeclareVariable(float3,            ambient_light_color, , );
rtDeclareVariable(unsigned int,      radiance_ray_type, , );
rtDeclareVariable(unsigned int,      shadow_ray_type, , );
rtDeclareVariable(float,             scene_epsilon, , );
rtDeclareVariable(rtObject,          top_object, , );
rtDeclareVariable(rtObject,          top_shadower, , );

rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

rtTextureSampler<float4, 2> transmissive_map;
rtDeclareVariable(float3,   Kd, , );
rtDeclareVariable(float3,   Ks, , );
rtDeclareVariable(float3,   Ka, , );
rtDeclareVariable(float,    refraction_index, , );
rtDeclareVariable(float,    phong_exp, , );

rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 

static __device__ __inline__ float pow5( float x )
{
  float t = x*x;
  return t*t*x;
}

RT_PROGRAM void any_hit_shadow()
{
  float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 ffnormal     = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

  float fresnel_lo = 0.1f;
  float fresnel_hi = 0.5f;
  float fresnel = fresnel_lo + ( fresnel_hi - fresnel_lo )*pow5( 1.0f - dot( ffnormal, -ray.direction ) );


  float3 Kt = make_float3( tex2D( transmissive_map, texcoord.x, texcoord.y ) );
  prd_shadow.attenuation = prd_shadow.attenuation * Kt * ( 1.0f - fresnel );
  
  if ( fmaxf( prd_shadow.attenuation ) < 0.001f )
    rtTerminateRay();
  else 
    rtIgnoreIntersection();
}


RT_PROGRAM void closest_hit_radiance()
{
  float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 ffnormal     = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
  float3 hit_point    = ray.origin + t_hit * ray.direction;

  // ambient contribution
  float3 result = Ka * ambient_light_color;

  // compute direct lighting
  unsigned int num_lights = lights.size();
  for(int i = 0; i < num_lights; ++i) {
    BasicLight light = lights[i];
    float Ldist = length(light.pos - hit_point);
    float3 L = normalize(light.pos - hit_point);
    float nDl = dot( ffnormal, L);

    // cast shadow ray
    float3 light_attenuation = make_float3( static_cast<float>( nDl > 0.0f ) );
    if ( nDl > 0.0f && light.casts_shadow ) {
      PerRayData_shadow prd;
      prd.attenuation = make_float3( 1.0f );
      optix::Ray shadow_ray( hit_point, L, shadow_ray_type, scene_epsilon, Ldist );
      rtTrace(top_shadower, shadow_ray, prd);
      light_attenuation = prd.attenuation;
    }

    // If not completely shadowed, light the hit point
    if( fmaxf( light_attenuation ) > 0.0f ) {
      float3 Lc = light.color * light_attenuation;

      result += Kd * nDl * Lc;

      float3 H = normalize(L - ray.direction);
      float nDh = dot( ffnormal, H );
      if(nDh > 0) {
        float power = pow(nDh, phong_exp);
        result += Ks * power * Lc;
      }
    }
  }

  // Reflection
  if( fmaxf( Ks ) > 0  && prd_radiance.depth < max_depth ) {

    // ray tree attenuation
    PerRayData_radiance new_prd;             
    new_prd.importance = prd_radiance.importance * optix::luminance( Ks );
    new_prd.depth = prd_radiance.depth + 1;

    // reflection ray
    if( new_prd.importance >= 0.01f ) {
      float3 R = reflect( ray.direction, ffnormal );
      optix::Ray refl_ray( hit_point, R, radiance_ray_type, scene_epsilon );
      rtTrace(top_object, refl_ray, new_prd);
      result += Ks * new_prd.result;
    }
  }
  
  // Refraction
  float3 Kt = make_float3( tex2D( transmissive_map, texcoord.x, texcoord.y ) );
  if( fmaxf( Kt ) > 0.0f && prd_radiance.depth < max_depth ) {

    // ray tree attenuation
    PerRayData_radiance new_prd;             
    new_prd.importance = prd_radiance.importance * optix::luminance( Kt );
    new_prd.depth = prd_radiance.depth + 1;

    // refrection ray
    if( new_prd.importance >= 0.01f ) {

      float3 R;
      if ( refract( R, ray.direction, ffnormal, refraction_index) ) {

        optix::Ray refl_ray( hit_point, R, radiance_ray_type, scene_epsilon );
        rtTrace(top_object, refl_ray, new_prd);
        result += Kt * new_prd.result;
      }
    }
  }

  // pass the color back up the tree
  prd_radiance.result = result;
}



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

#include <optix_world.h>

using namespace optix;

// We create multiple spheres, one per primitive.
rtDeclareVariable(float4,  sphere_0, , ) = {0.f, 1.2f, 0.f, 1.f};
rtDeclareVariable(float2,  sphere_spacing, , ) = {3.f, 3.2f};
rtDeclareVariable(uint2,   sphere_count, , ) = {4, 4};
rtDeclareVariable(uint,    material_count, , ) = 1;

rtDeclareVariable(float3, geometric_normal,   attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal,     attribute shading_normal, ); 
rtDeclareVariable(float3, shading_tangent,    attribute shading_tangent, ); 
rtDeclareVariable(float2, uv,                 attribute uv, ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );



__forceinline__ __device__ float2 make_uv( float3 sphere_normal )
{
  return make_float2(
    acosf(sphere_normal.z) * M_1_PIf,
    atan2f(sphere_normal.y, sphere_normal.x) * M_1_PIf);
}

__forceinline__ __device__ float3 make_tangent( float3 sphere_normal )
{
  //float3 t = cross( sphere_normal, make_float3( 0.f, 0.f, 1.f ));
  const float3 t = make_float3( -sphere_normal.y, sphere_normal.x, 0.f );
  const float len = sphere_normal.x * sphere_normal.x + sphere_normal.y * sphere_normal.y;
  return (len < 0.001f) ? make_float3(1.f, 0.f, 0.f) :   // arbitrary pole tangent
    t * rsqrtf(len);
}


template<bool use_robust_method>
__device__
void intersect_sphere( float4 sphere, int primIdx )
{
  float3 center = make_float3(sphere);
  float3 O = ray.origin - center;
  float3 D = ray.direction;
  float radius = sphere.w;

  float b = dot(O, D);
  float c = dot(O, O)-radius*radius;
  float disc = b*b-c;
  if(disc > 0.0f){
    float sdisc = sqrtf(disc);
    float root1 = (-b - sdisc);

    bool do_refine = false;

    float root11 = 0.0f;

    if(use_robust_method && fabsf(root1) > 10.f * radius) {
      do_refine = true;
    }

    if(do_refine) {
      // refine root1
      float3 O1 = O + root1 * ray.direction;
      b = dot(O1, D);
      c = dot(O1, O1) - radius*radius;
      disc = b*b - c;

      if(disc > 0.0f) {
        sdisc = sqrtf(disc);
        root11 = (-b - sdisc);
      }
    }

    bool check_second = true;
    float3 sphere_normal;
    if( rtPotentialIntersection( root1 + root11 ) ) {
      sphere_normal = (O + (root1 + root11)*D)/radius;
      shading_normal = geometric_normal = sphere_normal;
      uv = make_uv( sphere_normal );
      shading_tangent = make_tangent( sphere_normal );
      if(rtReportIntersection( primIdx % (int)material_count ))
        check_second = false;
    } 
    if(check_second) {
      float root2 = (-b + sdisc) + (do_refine ? root1 : 0);
      if( rtPotentialIntersection( root2 ) ) {
        sphere_normal= (O + root2*D)/radius;
        shading_normal = geometric_normal = sphere_normal;
        uv = make_uv( sphere_normal );
        shading_tangent = make_tangent( sphere_normal );
        rtReportIntersection( primIdx % (int)material_count );
      }
    }
  }
}

__forceinline__ __device__ float4 make_sphere(int primIdx)
{
  return make_float4(
    sphere_0.x + primIdx % sphere_count.x * sphere_spacing.x,
    sphere_0.y,
    sphere_0.z + primIdx / sphere_count.x * sphere_spacing.y,
    sphere_0.w );
}

RT_PROGRAM void intersect(int primIdx)
{
  intersect_sphere<false>( make_sphere(primIdx), primIdx );
}

RT_PROGRAM void robust_intersect(int primIdx)
{
  intersect_sphere<true>( make_sphere(primIdx), primIdx );
}


RT_PROGRAM void bounds (int primIdx, float result[6])
{
  const float3 cen = make_float3( make_sphere(primIdx) );
  const float3 rad = make_float3( sphere_0.w );

  optix::Aabb* aabb = (optix::Aabb*)result;
  
  if( rad.x > 0.0f  && !isinf(rad.x) ) {
    aabb->m_min = cen - rad;
    aabb->m_max = cen + rad;
  } else {
    aabb->invalidate();
  }
}


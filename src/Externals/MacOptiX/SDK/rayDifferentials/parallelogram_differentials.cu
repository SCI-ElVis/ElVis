
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
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

struct PerRayData_radiance_differentials
{
  float3 result;
  float  importance;
  int    depth;
  
  float3 origin_dx;
  float3 direction_dx;
  float3 origin_dy;
  float3 direction_dy;
};

rtDeclareVariable(PerRayData_radiance_differentials, prd_radiance, rtPayload, );

rtDeclareVariable(float4, plane, , );
rtDeclareVariable(float3, v1, , );
rtDeclareVariable(float3, v2, , );
rtDeclareVariable(float3, anchor, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(float3, dudP, attribute dudP, ); 
rtDeclareVariable(float3, dvdP, attribute dvdP, ); 

RT_PROGRAM void intersect(int primIdx)
{
  float3 n = make_float3( plane );
  float dt = dot(ray.direction, n );
  float t = (plane.w - dot(n, ray.origin))/dt;
  if( t > ray.tmin && t < ray.tmax ) {
    float3 p = ray.origin + ray.direction * t;
    float3 vi = p - anchor;
    float a1 = dot(v1, vi);
    if(a1 >= 0 && a1 <= 1){
      float a2 = dot(v2, vi);
      if(a2 >= 0 && a2 <= 1){
        if( rtPotentialIntersection( t ) ) {
          shading_normal = geometric_normal = n;
          texcoord = make_float3(a1, a2, 0.f);

          // only correct for rectangles in the XZ plane
          float3 tv1 = v1 / dot( v1, v1 );
          float3 tv2 = v2 / dot( v2, v2 );
          dudP = make_float3(1.f/length(tv1), 0.f, 0.f);
          dvdP = make_float3(0.f, 0.f, 1.f/length(tv2));

          rtReportIntersection( 0 );
        }
      }
    }
  }
}

RT_PROGRAM void bounds (int, float result[6])
{
  // v1 and v2 are scaled by 1./length^2.  Rescale back to normal for the bounds computation.
  float3 tv1 = v1 / dot( v1, v1 );
  float3 tv2 = v2 / dot( v2, v2 );
  float3 p00 = anchor;
  float3 p01 = anchor + tv1;
  float3 p10 = anchor + tv2;
  float3 p11 = anchor + tv1 + tv2;

  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->m_min = fminf( fminf( p00, p01 ), fminf( p10, p11 ) );
  aabb->m_max = fmaxf( fmaxf( p00, p01 ), fmaxf( p10, p11 ) );
}

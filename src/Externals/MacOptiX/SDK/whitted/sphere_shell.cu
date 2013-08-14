
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

rtDeclareVariable(float3,  center, , );
rtDeclareVariable(float,   radius1, , );
rtDeclareVariable(float,   radius2, , );
rtDeclareVariable(float,   scene_epsilon, , );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(float3, front_hit_point, attribute front_hit_point, ); 
rtDeclareVariable(float3, back_hit_point, attribute back_hit_point, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 

RT_PROGRAM void intersect(int primIdx)
{
  float3 O = ray.origin - center;
  float3 D = ray.direction;

  float b = dot(O, D);
  float O_dot_O = dot(O, O);
  float sqr_radius2 = radius2*radius2;

  // check if we are outside of outer sphere
  if ( O_dot_O > sqr_radius2 + scene_epsilon ) { 
    float c = O_dot_O - sqr_radius2; 
    float root = b*b-c;
    if ( root > 0.0f ) {
      float t = -b - sqrtf( root );
      if( rtPotentialIntersection( t ) ) {
        shading_normal = geometric_normal = ( O + t*D ) / radius2;
        float3 hit_p  = ray.origin +t*ray.direction;
        float3 offset = normalize( shading_normal )*scene_epsilon;
        front_hit_point = hit_p + offset;
        back_hit_point = hit_p  - offset;
        rtReportIntersection( 0 );
      } 
    }

    // else we are inside of the outer sphere
  } else {

    float c = O_dot_O - radius1*radius1;
    float root = b*b-c;
    if ( root > 0.0f ) {
      float t = -b - sqrtf( root );
      // do we hit inner sphere from between spheres? 
      if( rtPotentialIntersection( t ) ) {
        shading_normal = geometric_normal = ( O + t*D )/(-radius1);
        float3 hit_p  = ray.origin +t*ray.direction;
        float3 offset = normalize( shading_normal )*scene_epsilon;
        front_hit_point = hit_p - offset;
        back_hit_point  = hit_p  + offset;
        rtReportIntersection( 0 );
      } else { 
        t = -b + sqrtf( root );
        // do we hit inner sphere from within both spheres?
        if( rtPotentialIntersection( t ) ) {
          shading_normal = geometric_normal = ( O + t*D )/(-radius1);
          float3 hit_p  = ray.origin +t*ray.direction;
          float3 offset = normalize( shading_normal )*scene_epsilon;
          front_hit_point = hit_p + offset;
          back_hit_point = hit_p  - offset;
          rtReportIntersection( 0 );
        } else {
          c = O_dot_O - sqr_radius2;
          root = b*b-c;
          t = -b + sqrtf( root );
          // do we hit outer sphere from between spheres?
          if( rtPotentialIntersection( t ) ) {
            shading_normal = geometric_normal = ( O + t*D )/radius2;
            float3 hit_p  = ray.origin +t*ray.direction;
            float3 offset = normalize( shading_normal )*scene_epsilon;
            front_hit_point = hit_p - offset;
            back_hit_point = hit_p  + offset;
            rtReportIntersection( 0 );
          }
        }
      }
    } else { 
      c = O_dot_O - sqr_radius2;
      root = b*b-c;
      float t = -b + sqrtf( root );
      // do we hit outer sphere from between spheres? 
      if( rtPotentialIntersection( t ) ) {
        shading_normal = geometric_normal = ( O + t*D )/radius2;
        float3 hit_p  = ray.origin +t*ray.direction;
        float3 offset = normalize( shading_normal )*scene_epsilon;
        front_hit_point = hit_p - offset;
        back_hit_point = hit_p  + offset;
        rtReportIntersection( 0 );
      }
    }
  }
}


RT_PROGRAM void bounds (int, optix::Aabb* aabb)
{
  float3 rad = make_float3( max(radius1,radius2) );
  aabb->m_min = center - rad;
  aabb->m_max = center + rad;
}

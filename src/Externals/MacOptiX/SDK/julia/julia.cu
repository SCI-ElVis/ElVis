
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
#include "distance_field.h"

using namespace optix;


// References:
// [1] Hart, J. C., Sandin, D. J., and Kauffman, L. H. 1989. Ray tracing deterministic 3D fractals
// [2] http://www.devmaster.net/forums/showthread.php?t=4448


rtDeclareVariable( float4, c4 , , );                // parameter quaternion
rtDeclareVariable( float,  alpha , , );
rtDeclareVariable( float,  delta , , );
rtDeclareVariable( float,  DEL , , );
rtDeclareVariable( float,  color_t , , );           // 0,1,2 are full colors, in between is morph
rtDeclareVariable( uint,   max_iterations , , );    // max iterations for divergence determination
rtDeclareVariable( float3, particle , , );          // position of force particle

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

// julia set object outputs this
rtDeclareVariable(float3, normal, attribute normal, ); 

// sphere outputs this
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 

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

rtDeclareVariable(rtObject,                         top_object, , );
rtDeclareVariable(float, isect_t, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );


// Quaternion helpers.
static __host__ __device__ float4 mul( float4 a, float4 b )
{
  const float3 a2 = make_float3( a.y, a.z, a.w );
  const float3 b2 = make_float3( b.y, b.z, b.w );
  float3 r;
  r = a.x*b2 + b.x*a2 + cross( a2, b2 );
  return make_float4(
    a.x*b.x - dot( a2, b2 ),
    r.x, r.y, r.z );
}
static __host__ __device__ float4 square( float4 a )
{
  float3 a2 = make_float3( a.y, a.z, a.w );
  float3 r;
  r = 2*a.x*a2;
  return make_float4(
    a.x*a.x - dot( a2,a2 ),
    r.x, r.y, r.z );
}

// Intersect the bounding sphere of the Julia set.
static __host__ __device__ bool intersectBoundingSphere( float3 o, float3 d, float& tmin, float &tmax )
{
  const float sq_radius = 4.0f;
  const float b = dot( o, d );
  const float c = dot( o, o ) - sq_radius;
  const float disc = b*b - c;
  
  if( disc > 0.0f )
  {
    const float sdisc = sqrtf( disc );
    tmin = (-b - sdisc);
    tmax = (-b + sdisc);

    if(tmin > tmax)
    {
      const float temp = tmin;
      tmax = tmin;
      tmin = temp;
    }

    return true;
  } else {
    tmin = tmax = 0;
  }
  return false;
}

struct JuliaSet
{
  __host__ __device__
  JuliaSet(const unsigned int max_iterations):m_max_iterations(max_iterations)
  {}

  // Return the approximate lower bound on the distance from x to the set.
  __host__ __device__ __forceinline__
  float operator()( float3 x ) const
  {
    // Warp space around the particle to get the blob-effect.
    const float part_dist = length( particle - x );
    const float force = smoothstep( 0.0f, 1.0f, 0.1f / (part_dist*part_dist) ) * 0.2f;
    const float3 weg = (x-particle) / max(0.01f,part_dist);
    x -= weg * force;

    // Iterated values.
    float4 f_n  = make_float4( x, 0 );
    float4 fp_n = make_float4( 1, 0, 0, 0 );  // start derivative at real 1 (see [2]).

    const float sq_threshold = 16.0f;   // divergence threshold

    // Iterate to compute f_n and fp_n for the distance estimator.
    int i = m_max_iterations;
    while( i-- )
    {
      fp_n = 2.0f * mul( f_n, fp_n );   // z prime in [2]
      f_n = square( f_n ) + c4;         // equation (1) in [1]

      // Stop when we know the point diverges.
      // TODO: removing this condition burns 2 less registers and results in
      //       in a big perf improvement. Can we do something about it?
      if( dot(f_n,f_n) > sq_threshold )
        break;
    }

    // Distance estimation. Equation (8) from [1], with correction mentioned in [2].
    const float norm = length( f_n );
    const float julia_dist = 0.5f * norm * logf( norm ) / length( fp_n );

    return julia_dist;
    //return fminf( julia_dist, part_dist - 0.2f );  // this "renders" the particle as well
  }

  unsigned int m_max_iterations;
};


RT_PROGRAM void intersect(int primIdx)
{
  float tmin, tmax;
  if( intersectBoundingSphere(ray.origin, ray.direction, tmin, tmax) )
  {
    JuliaSet distance( max_iterations );

    // XXX inline the sphere tracing procedure here because nvcc isn't
    //     generating the right code i guess

    float3 ray_direction = ray.direction;
    float3 x = ray.origin + tmin * ray_direction;

    float dist_from_origin = tmin;

    // Compute epsilon using equation (16) of [1].
    //float epsilon = max(0.000001f, alpha * powf(dist_from_origin, delta));
    const float epsilon = 1e-3f;

    float dist = 0;
    for( unsigned int i = 0; i < 800; ++i )
    {
      dist = distance( x );

      // Step along the ray and accumulate the distance from the origin.
      x += dist * ray_direction;
      dist_from_origin += dist;

      // Check if we're close enough or too far.
      if( dist < epsilon || dist_from_origin > tmax  )
         break;
    }

    // Found intersection?
    if( dist < epsilon )
    {
      if( rtPotentialIntersection(dist_from_origin) )
      {
        // color HACK
        distance.m_max_iterations = 14;  // more iterations for normal estimate, to fake some more detail
        normal = estimate_normal(distance, x, DEL);
        rtReportIntersection( 0 );
      }
    }
  }
}

RT_PROGRAM void bounds (int, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  const float sz = 1.4f;
  aabb->m_min = make_float3(-sz);
  aabb->m_max = make_float3(sz);
}


//
// Julia set shader.
//

RT_PROGRAM void julia_ch_radiance()
{
  const float3 p = ray.origin + isect_t * ray.direction;

  // ambient occlusion
  JuliaSet distance( max_iterations );
  float occlusion = 1.f;
  float fact = 7.f;
  const float delta = 0.05f;
  for( int i=0; i<4; ++i ) {
    const float dist = delta * i;
    occlusion -= fact * (dist - distance(p+dist*normal));
    fact *= 0.5f;
  }
  occlusion += 0.3f;
  occlusion *= occlusion;
  occlusion = clamp( occlusion, 0.2f, 1.0f );

  // base colors
  float3 red   = normal*0.5f + make_float3(0.5f);
  float3 green = red;
  float3 blue  = red;
  
  // red/orange
  red.x = abs(normal.x)*0.5f + 0.5f;
  red.x = max( red.x, 0.1f );
  red = red * make_float3( 0.8f, 0.1f+red.x, 0.1f );
  red.y += 0.2f * red.x;
  red.x += 0.6f;
  red.x *= max(occlusion,0.8f);
  red.y *= occlusion;
  red.z *= occlusion;
  
  // green
  green.x = abs(normal.x)*0.5f + 0.5f;
  green.z = -abs(normal.z)*0.5f + 0.5f;
  green.y = green.y * 0.7f + 0.3f;
  green = green * make_float3( 0.9f*green.y*green.y, 1.0f, 0.2f );
  green.x += 0.2f;
  green.x *= green.x;
  green.x *= occlusion;
  green.y = max(0.3f,green.y*occlusion);
  green.z *= occlusion;

  // blue
  blue.x = abs(normal.x)*0.5f + 0.5f;
  blue.y = abs(normal.y)*0.5f + 0.5f;
  blue.z = -abs(normal.z)*0.5f + 0.5f;
  blue.z = blue.z * 0.7f + 0.3f;
  blue.x += 0.2f;
  blue.y += 0.2f;
  blue = blue * make_float3( 0.9f*blue.y*blue.y, 1.0f*blue.z*blue.y, 1.0f );
  blue.z += 0.3f;
  blue.x *= blue.z * max(0.3f,occlusion);
  blue.y *= occlusion;
  blue.z = blue.z * max(0.6f,occlusion);

  // select color
  float3 c0 = green;
  float3 c1 = red;
  float ct = color_t;
  if( color_t > 1.0f ) {
    c0 = red;
    c1 = blue;
    ct -= 1.0f;
  }
  float3 result = dot(p,p) > ct*3.0f ? c0 : c1;

  // add glow close to particle
  const float part_dist = length( p-particle );
  const float glow = 1.0f - smoothstep( 0.0f, 1.0f, part_dist );
  result = result + make_float3(glow*0.4f);

  // add phong highlight
  const float3 l = make_float3( 1, 3, 1 );
  const float3 h = normalize( l - ray.direction );
  const float ndh = dot( normal, h );
  if( ndh > 0.0f ) {
    result = result + make_float3( 0.6f * occlusion * pow(ndh,20.0f) );
  }

  //result = make_float3( occlusion );

  // Reflection (disabled, doesn't look too great)
  /*
  PerRayData_radiance new_prd;             
  new_prd.importance = prd_radiance.importance;
  new_prd.depth = prd_radiance.depth + 1;
  if( prd_radiance.depth < 3 )
  {
    const float3 refl = reflect( ray.direction, normal );
    const Ray refl_ray = make_ray( p, refl, 0, 1e-3f, RT_DEFAULT_MAX );
    rtTrace( top_object, refl_ray, new_prd );
    result += new_prd.result * 0.1f;
  }
  */

  prd_radiance.result = result;
}

RT_PROGRAM void julia_ah_shadow()
{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = make_float3(0);
  rtTerminateRay();
}


//
// Chrome shader for force particle.
//

RT_PROGRAM void chrome_ch_radiance()
{
  float3 dir = ray.direction;
  const float3 hit_point = ray.origin + isect_t * dir;

  if( prd_radiance.depth < 3 )
  {
    PerRayData_radiance new_prd;             
    new_prd.importance = prd_radiance.importance;
    new_prd.depth = prd_radiance.depth + 1;
    
    const float3 refl = reflect( dir, shading_normal );
    const optix::Ray refl_ray = optix::make_Ray( hit_point, refl, 0, 1e-3f, RT_DEFAULT_MAX );
    rtTrace( top_object, refl_ray, new_prd );
    const float fresnel = fresnel_schlick( dot(shading_normal,-dir), 5.0f, 0.3f, 1.0f );
    const float diff = (shading_normal.y+1.f) * 0.5f;
    prd_radiance.result = new_prd.result * fresnel
      + make_float3(diff*diff*diff*0.1f);
  } else {
    prd_radiance.result = make_float3( 0 );
  }
}

RT_PROGRAM void chrome_ah_shadow()
{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = make_float3(0);
  rtTerminateRay();
}

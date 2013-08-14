
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

using namespace optix;

// Turbulence parameters
rtDeclareVariable(float,  frequency, , );
rtDeclareVariable(float,  turbulence, , );
rtDeclareVariable(float,  lambda, , );
rtDeclareVariable(float,  omega, , );
rtDeclareVariable(int,    octaves, , );
rtDeclareVariable(float3, origin, , );

rtDeclareVariable(float, diameter, , );

// Material properties
rtDeclareVariable(float, reflectivity, , );
rtDeclareVariable(float, specular_exp, , );

// Attributes
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 

// Ray tracing
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, isect_dist, rtIntersectionDistance, );

rtTextureSampler<float,  3> noise_texture;

// 3D solid noise texture, values in range [0, 1].
__device__ __inline__ float Noise1f(float3 p)
{
  return tex3D(noise_texture, p.x, p.y, p.z);
}

__device__ __inline__ float Turbulence(float3 point)
{
  float t = Noise1f(point);
  float l = lambda;
  float o = omega;  
  for (int i = 1; i < octaves; i++)
  {
    float3 pos = point * l;
    t += Noise1f(pos) * o;
    l *= lambda;
    o *= omega;
  }
  return t * turbulence;
}


// #############################
// Procedural solid 3D textures.
//
// All these functions return a single float value in the range [0.0, 1.0]
// which can be used as 1D texture coordinate into a color ramp texture.

// Marble, maps the 1D color ramp to the x-axis through the pattern origin.
RT_CALLABLE_PROGRAM float Marble(float3 point)
{
  float3 p = (point - origin) * frequency;
  return p.x + Turbulence(p);
}

// Wood, maps the 1D color map to the distance from the x-axis through the pattern origin.
RT_CALLABLE_PROGRAM float Wood(float3 point)
{
  float3 p = (point - origin) * frequency;
  return sqrtf(p.x * p.x + p.y * p.y) + Turbulence(p);
}

// Onion, maps the 1D color map to the distance from the pattern origin.
RT_CALLABLE_PROGRAM float Onion(float3 point)
{
  float3 p = (point - origin) * frequency;
  return length(p) + Turbulence(p);
}

// NoiseCubed, noise()^3 of range [0, 1] values results in smoother noise distribution. 
RT_CALLABLE_PROGRAM float NoiseCubed(float3 point)
{
  float3 p = (point - origin) * frequency;
  float  t = Noise1f(p);
  return t * t * t + Turbulence(p);
}

// Voronoi, requires a matching 3D lookup texture
// containing the distance from the equidistant surface 
// between the two nearest control points.
RT_CALLABLE_PROGRAM float Voronoi(float3 point)
{
  float3 p = (point - origin) * frequency;
  float  t = Noise1f(p);
  return t + Turbulence(p);
}


// Sphere, 3D sphere of given radius inside cube [0,1] around pattern origin inside is 0.0, outside is 1.0.
RT_CALLABLE_PROGRAM float Sphere(float3 point)
{
  float3 p = (point - origin) * frequency;
  // repeat-map to [0, 1), then scale and bias to [-1, 1).
  p = (p - floor(p)) * 2.0f - 1.0f; 
  return (length(p) > diameter) ? 1.0 : 0.0f;
}

// Checker, 3D unit cubes in alternating colors.
RT_CALLABLE_PROGRAM float Checker(float3 point)
{
  float3 p = (point - origin) * frequency ;
  int    t = (int) (floor(p.x) + floor(p.y) + floor(p.z));
  return (float) (t & 1);
}

// #############################

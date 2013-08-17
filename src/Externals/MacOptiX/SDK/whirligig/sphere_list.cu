
// Copyright NVIDIA Corporation 2008
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

// x,y,z are center. w is radius.
rtBuffer<float4> sphere_buffer;  

rtBuffer<uint>   material_buffer; // per-sphere material index

rtDeclareVariable(float,   scene_epsilon, , );

rtDeclareVariable(float3, front_hit_point, attribute front_hit_point, ); 
rtDeclareVariable(float3, back_hit_point, attribute back_hit_point, ); 

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(float3, texcoord, attribute texcoord, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void intersect(int primIdx)
{
  float3 center = make_float3(sphere_buffer[primIdx]);
  float3 O = ray.origin - center;
  float3 D = ray.direction;
  float radius = sphere_buffer[primIdx].w;

  float b = dot(O, D);
  float c = dot(O, O)-radius*radius;
  float disc = b*b-c;
  if(disc > 0.0f){
    float sdisc = sqrtf(disc);
    float root1 = (-b - sdisc);
    bool check_second = true;
    if( rtPotentialIntersection( root1 ) ) {
      shading_normal = geometric_normal = (O + root1*D)/radius;
      float3 hit_p  = ray.origin +root1*ray.direction;
      float3 offset = shading_normal * scene_epsilon;
      front_hit_point = hit_p + offset;
      back_hit_point = hit_p  - offset;
      if(rtReportIntersection(material_buffer[primIdx]))
        check_second = false;
    } 
    if(check_second) {
      float root2 = (-b + sdisc);
      if( rtPotentialIntersection( root2 ) ) {
        shading_normal = geometric_normal = (O + root2*D)/radius;
        float3 hit_p  = ray.origin +root2*ray.direction;
        float3 offset = shading_normal * scene_epsilon;
        front_hit_point = hit_p - offset;
        back_hit_point = hit_p  + offset;
        texcoord = make_float3(0);
        rtReportIntersection(material_buffer[primIdx]);
      }
    }
  }
}

RT_PROGRAM void bounds (int primIdx, float result[6])
{
  const float3 cen = make_float3( sphere_buffer[primIdx] );
  const float3 rad = make_float3( sphere_buffer[primIdx].w );

  optix::Aabb* aabb = (optix::Aabb*)result;

  if( rad.x > 0.0f && !isinf(rad.x) ) {
    aabb->m_min = cen - rad;
    aabb->m_max = cen + rad;
  } else {
    aabb->invalidate();
  }
}


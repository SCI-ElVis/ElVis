
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "commonStructs.h"
#include "helpers.h"

using namespace optix;

rtDeclareVariable(uint, radiance_ray_type, , );
rtDeclareVariable(uint, shadow_ray_type, , );
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, reflectors, , );
rtDeclareVariable(uint, max_depth, , );

rtBuffer<BasicLight> lights;

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

struct PerRayData_radiance
{
  float3 result;
  float importance;
  int depth;
  float t_hit;
};

struct PerRayData_shadow
{
  float3 attenuation;
};

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

RT_PROGRAM void closest_hit_radiance()
{
  prd_radiance.t_hit = t_hit;

  float3 hit_point = ray.origin + t_hit * ray.direction;
  float3 color = make_float3(.95f, .92f, .6f);

  float fresnel = fresnel_schlick(dot(-ray.direction, shading_normal), 5.f, 0.9f);

  PerRayData_radiance refl_prd;
  refl_prd.importance = prd_radiance.importance * fresnel * optix::luminance(color);
  refl_prd.depth = prd_radiance.depth + 1;

  float3 result;
  if(refl_prd.depth <= max_depth && refl_prd.importance > 0.05) {

    optix::Ray refl_ray = optix::make_Ray(hit_point, reflect(ray.direction, shading_normal), 
                                          radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
    rtTrace(reflectors, refl_ray, refl_prd);
    result = refl_prd.result * fresnel * color;
  } else {
    result = make_float3(0.f,0.f,0.f);
  }
  
  prd_radiance.result = result;
}

RT_PROGRAM void any_hit_shadow()
{
  prd_shadow.attenuation = make_float3(0);
  rtTerminateRay();
}

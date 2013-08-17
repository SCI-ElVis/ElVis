
#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtBuffer<float4, 2>         reflection_buffer;
rtTextureSampler<float4, 2> normal_texture;
rtTextureSampler<float4, 2> request_texture;

rtDeclareVariable(uint, radiance_ray_type, , );
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, reflectors, , );
rtDeclareVariable(float3, eye_pos, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

struct PerRayData_radiance
{
  float3 result;
  float importance;
  int depth;
  float t_hit;
};

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );

RT_PROGRAM void reflection_request()
{
  float3 ray_origin = make_float3(tex2D(request_texture, launch_index.x, launch_index.y));

  PerRayData_radiance prd;
  prd.result = make_float3(0);
  prd.importance = 1.f;
  prd.depth = 0;
  prd.t_hit = -1.f;

  if( !isnan(ray_origin.x) ) {
    float3 V = normalize(ray_origin-eye_pos);
    float3 normal = make_float3(tex2D(normal_texture, launch_index.x, launch_index.y));
    float3 ray_direction = reflect(V, normal);

    optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
    rtTrace(reflectors, ray, prd);
    reflection_buffer[launch_index] = make_float4(prd.result, prd.t_hit);
  }
}

RT_PROGRAM void reflection_exception()
{
  reflection_buffer[launch_index] = make_float4(0.f,0.f,0.f,-1.f);
}

RT_PROGRAM void reflection_miss()
{
  prd_radiance.t_hit = RT_DEFAULT_MAX;
  prd_radiance.result = make_float3(1.f, 1.f, 1.f);
}

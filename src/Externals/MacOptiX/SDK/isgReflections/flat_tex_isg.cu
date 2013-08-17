
#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

struct PerRayData_radiance
{
  float3 result;
  float  importance;
  int    depth;
  float t_hit;
};

struct PerRayData_shadow
{
  float3 attenuation;
};

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

rtTextureSampler<float4, 2> diffuse_texture;
rtDeclareVariable(float3,   texcoord, attribute texcoord, );

RT_PROGRAM void any_hit_shadow()
{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = make_float3(0);
  
  rtTerminateRay();
}

RT_PROGRAM void closest_hit_radiance()
{
  prd_radiance.t_hit = t_hit;

  prd_radiance.result = make_float3(tex2D(diffuse_texture, texcoord.x, texcoord.y));
}

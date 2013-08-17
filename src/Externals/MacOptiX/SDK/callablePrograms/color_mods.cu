
#include <optix_world.h>


RT_CALLABLE_PROGRAM float3 scale_color(float3 input_color, float multiplier)
{
  return multiplier * input_color;
}

// Stubs only needed for sm_1x
#if __CUDA_ARCH__ < 200
__global__ void scale_color_stub() {
  (void) scale_color( make_float3(0,0,0), 0 );
}
#endif

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(float, scale,,);

RT_CALLABLE_PROGRAM float3 checker_color(float3 input_color, float multiplier)
{
  uint2 tile_size = make_uint2(launch_dim.x / 5, launch_dim.y / 5);
  if ((launch_index.x/tile_size.x + launch_index.y/tile_size.y) % 2 == 0)
    return input_color * multiplier;
  else
    return input_color * scale;
}

// Stubs only needed for sm_1x
#if __CUDA_ARCH__ < 200
__global__ void checker_color_stub()
{
  (void) checker_color( make_float3(0,0,0), 0 );
}
#endif

RT_CALLABLE_PROGRAM float3 wavey_color(float3 input_color, float multiplier)
{
  uint2 tile_size = make_uint2(launch_dim.x / 5, launch_dim.y / 5);
  if (((int)(launch_index.x+10*sinf(launch_index.y/10.f))/tile_size.x + launch_index.y/tile_size.y) % 2 == 0)
    return input_color * multiplier;
  else
    return input_color * scale;
  
}

// Stubs only needed for sm_1x
#if __CUDA_ARCH__ < 200
__global__ void wavey_color_stub()
{
  (void) wavey_color( make_float3(0,0,0), 0 );
}
#endif

RT_CALLABLE_PROGRAM float3 return_same_color(float3 input_color)
{
  return input_color;
}

// Stubs only needed for sm_1x
#if __CUDA_ARCH__ < 200
__global__ void return_same_color_stub()
{
  (void) return_same_color( make_float3(0,0,0) );
}
#endif


/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 * 
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure, or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation
 * is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern your
 * use of this NVIDIA software.
 * 
 */

///////////////////////////////////////////////////////////////////////////////

#include "helpers.h"
#include <optix.h>
#include <optix_math.h>


rtDeclareVariable( uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable( uint2, launch_dim,   rtLaunchDim, );

rtDeclareVariable( float, f_exposure, , );

rtBuffer<float4, 2> pre_image;
rtBuffer<uchar4, 2> output_buffer;



RT_PROGRAM void tonemap()
{
  float3 val_Yxy = rgb2Yxy( make_float3( pre_image[ launch_index ] ) );
  
  float Y        = val_Yxy.x; // Y channel is luminance
  float mapped_Y = Y / ( Y + 1.0f );
  
  float3 mapped_Yxy = make_float3( mapped_Y, val_Yxy.y, val_Yxy.z ); 
  float3 mapped_rgb = Yxy2rgb( mapped_Yxy ); 

  output_buffer[ launch_index ] = make_color( mapped_rgb );  
}




/*
 * Copyright (c) 2010 NVIDIA Corporation.  All rights reserved.
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

#ifndef __optix_optix_dx9_interop_h__
#define __optix_optix_dx9_interop_h__

/************************************
 **
 **    DX9 Interop functions
 **
 ***********************************/

/*

On Windows you will need to include windows.h before this file:

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include<windows.h>
#endif
#include <optix_d3d9_interop.h>

*/

#if defined( _WIN32 )

#include "optix.h"

typedef struct IDirect3DDevice9   IDirect3DDevice9;
typedef struct IDirect3DResource9 IDirect3DResource9;

#ifdef __cplusplus
  extern "C" 
  {
#endif

  RTresult RTAPI rtContextSetD3D9Device                ( RTcontext context, IDirect3DDevice9* device );
  RTresult RTAPI rtDeviceGetD3D9Device                 ( int* device, const char* pszAdapterName );
  RTresult RTAPI rtBufferCreateFromD3D9Resource        ( RTcontext context, unsigned int bufferdesc, IDirect3DResource9* resource,  RTbuffer* buffer );
  RTresult RTAPI rtTextureSamplerCreateFromD3D9Resource( RTcontext context, IDirect3DResource9* resource,  RTtexturesampler* textureSampler );  
  RTresult RTAPI rtBufferGetD3D9Resource               ( RTbuffer buffer, IDirect3DResource9** resource );
  RTresult RTAPI rtTextureSamplerGetD3D9Resource       ( RTtexturesampler textureSampler, IDirect3DResource9** pResource );
  RTresult RTAPI rtBufferD3D9Register                  ( RTbuffer buffer );
  RTresult RTAPI rtBufferD3D9Unregister                ( RTbuffer buffer );
  RTresult RTAPI rtTextureSamplerD3D9Register          ( RTtexturesampler textureSampler );
  RTresult RTAPI rtTextureSamplerD3D9Unregister        ( RTtexturesampler textureSampler );

#ifdef __cplusplus
  }
#endif

#endif /* defined( _WIN32 ) */

#endif /* __optix_optix_dx9_interop_h__ */

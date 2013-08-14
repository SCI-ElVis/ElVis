
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

#ifndef __optix_optix_dx11_interop_h__
#define __optix_optix_dx11_interop_h__

/************************************
 **
 **    DX11 Interop functions
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
#include <optix_d3d11_interop.h>

*/

#if defined( _WIN32 )

#include "optix.h"

typedef struct IDXGIAdapter   IDXGIAdapter;
typedef struct ID3D11Device   ID3D11Device;
typedef struct ID3D11Resource ID3D11Resource;

#ifdef __cplusplus
  extern "C" 
  {
#endif

  RTresult RTAPI rtContextSetD3D11Device                ( RTcontext context, ID3D11Device* device );
  RTresult RTAPI rtDeviceGetD3D11Device                 ( int* device, IDXGIAdapter *pAdapter );
  RTresult RTAPI rtBufferCreateFromD3D11Resource        ( RTcontext context, unsigned int bufferdesc, ID3D11Resource* resource,  RTbuffer* buffer );
  RTresult RTAPI rtTextureSamplerCreateFromD3D11Resource( RTcontext context, ID3D11Resource* resource,  RTtexturesampler* textureSampler );  
  RTresult RTAPI rtBufferGetD3D11Resource               ( RTbuffer buffer, ID3D11Resource** resource );
  RTresult RTAPI rtTextureSamplerGetD3D11Resource       ( RTtexturesampler textureSampler, ID3D11Resource** resource );
  RTresult RTAPI rtBufferD3D11Register                  ( RTbuffer buffer );
  RTresult RTAPI rtBufferD3D11Unregister                ( RTbuffer buffer );
  RTresult RTAPI rtTextureSamplerD3D11Register          ( RTtexturesampler textureSampler );
  RTresult RTAPI rtTextureSamplerD3D11Unregister        ( RTtexturesampler textureSampler );

#ifdef __cplusplus
  }

#endif

#endif /* defined( _WIN32 ) */

#endif /* __optix_optix_dx11_interop_h__ */


/*
 * Copyright (c) 2008 - 2010 NVIDIA Corporation.  All rights reserved.
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

#ifndef __optix_optix_gl_interop_h__
#define __optix_optix_gl_interop_h__

#include "optix.h"

/************************************
 **
 **    OpenGL Interop functions
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
#include <optix_gl_interop.h>

*/

#ifdef __cplusplus
extern "C" {
#endif

  RTresult RTAPI rtBufferCreateFromGLBO           ( RTcontext context, unsigned int bufferdesc, unsigned int glId,  RTbuffer* buffer );
  RTresult RTAPI rtTextureSamplerCreateFromGLImage( RTcontext context, unsigned int glId, RTgltarget target, RTtexturesampler* textureSampler );
  RTresult RTAPI rtBufferGetGLBOId                ( RTbuffer buffer, unsigned int* glId );
  RTresult RTAPI rtTextureSamplerGetGLImageId     ( RTtexturesampler textureSampler, unsigned int* glId );
  RTresult RTAPI rtBufferGLRegister               ( RTbuffer buffer );
  RTresult RTAPI rtBufferGLUnregister             ( RTbuffer buffer );
  RTresult RTAPI rtTextureSamplerGLRegister       ( RTtexturesampler textureSampler );
  RTresult RTAPI rtTextureSamplerGLUnregister     ( RTtexturesampler textureSampler );

#if defined(_WIN32)
#if !defined(WGL_NV_gpu_affinity)
  typedef void* HGPUNV;
#endif
  RTresult RTAPI rtDeviceGetWGLDevice(int* device, HGPUNV gpu);
#endif

#ifdef __cplusplus
}
#endif

#endif /* __optix_optix_gl_interop_h__ */

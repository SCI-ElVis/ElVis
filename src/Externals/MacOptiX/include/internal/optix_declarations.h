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


/******************************************************************************\
 *
 * Contains declarations used by both optix host and device headers.
 *
\******************************************************************************/

#ifndef __optix_optix_declarations_h__
#define __optix_optix_declarations_h__

/************************************
 **
 **    Preprocessor macros 
 **
 ***********************************/

#if defined(__CUDACC__) || defined(__CUDABE__)
#  include <host_defines.h> /* For __host__ and __device__ */
#  define RT_HOSTDEVICE __host__ __device__
#else
#  define RT_HOSTDEVICE
#endif


/************************************
 **
 **    Enumerated values
 **
 ***********************************/

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
  RT_FORMAT_UNKNOWN              = 0x100,
  RT_FORMAT_FLOAT,
  RT_FORMAT_FLOAT2,
  RT_FORMAT_FLOAT3,
  RT_FORMAT_FLOAT4,
  RT_FORMAT_BYTE,
  RT_FORMAT_BYTE2,
  RT_FORMAT_BYTE3,
  RT_FORMAT_BYTE4,
  RT_FORMAT_UNSIGNED_BYTE,
  RT_FORMAT_UNSIGNED_BYTE2,
  RT_FORMAT_UNSIGNED_BYTE3,
  RT_FORMAT_UNSIGNED_BYTE4,
  RT_FORMAT_SHORT,
  RT_FORMAT_SHORT2,
  RT_FORMAT_SHORT3,
  RT_FORMAT_SHORT4,
  RT_FORMAT_UNSIGNED_SHORT,
  RT_FORMAT_UNSIGNED_SHORT2,
  RT_FORMAT_UNSIGNED_SHORT3,
  RT_FORMAT_UNSIGNED_SHORT4,
  RT_FORMAT_INT,
  RT_FORMAT_INT2,
  RT_FORMAT_INT3,
  RT_FORMAT_INT4,
  RT_FORMAT_UNSIGNED_INT,
  RT_FORMAT_UNSIGNED_INT2,
  RT_FORMAT_UNSIGNED_INT3,
  RT_FORMAT_UNSIGNED_INT4,
  RT_FORMAT_USER
} RTformat;

typedef enum
{
  RT_OBJECTTYPE_UNKNOWN          = 0x200,
  RT_OBJECTTYPE_GROUP,
  RT_OBJECTTYPE_GEOMETRY_GROUP,
  RT_OBJECTTYPE_TRANSFORM,
  RT_OBJECTTYPE_SELECTOR,
  RT_OBJECTTYPE_GEOMETRY_INSTANCE,
  RT_OBJECTTYPE_BUFFER,
  RT_OBJECTTYPE_TEXTURE_SAMPLER,
  RT_OBJECTTYPE_OBJECT,
  /* RT_OBJECTTYPE_PROGRAM - see below for entry */

  RT_OBJECTTYPE_MATRIX_FLOAT2x2,
  RT_OBJECTTYPE_MATRIX_FLOAT2x3,
  RT_OBJECTTYPE_MATRIX_FLOAT2x4,
  RT_OBJECTTYPE_MATRIX_FLOAT3x2,
  RT_OBJECTTYPE_MATRIX_FLOAT3x3,
  RT_OBJECTTYPE_MATRIX_FLOAT3x4,
  RT_OBJECTTYPE_MATRIX_FLOAT4x2,
  RT_OBJECTTYPE_MATRIX_FLOAT4x3,
  RT_OBJECTTYPE_MATRIX_FLOAT4x4,

  RT_OBJECTTYPE_FLOAT,
  RT_OBJECTTYPE_FLOAT2,
  RT_OBJECTTYPE_FLOAT3,
  RT_OBJECTTYPE_FLOAT4,
  RT_OBJECTTYPE_INT,
  RT_OBJECTTYPE_INT2,
  RT_OBJECTTYPE_INT3,
  RT_OBJECTTYPE_INT4,
  RT_OBJECTTYPE_UNSIGNED_INT,
  RT_OBJECTTYPE_UNSIGNED_INT2,
  RT_OBJECTTYPE_UNSIGNED_INT3,
  RT_OBJECTTYPE_UNSIGNED_INT4,
  RT_OBJECTTYPE_USER,

  RT_OBJECTTYPE_PROGRAM  /* Added in OptiX 3.0 */
} RTobjecttype;

typedef enum
{
  RT_WRAP_REPEAT,
  RT_WRAP_CLAMP_TO_EDGE,
  RT_WRAP_MIRROR,
  RT_WRAP_CLAMP_TO_BORDER
} RTwrapmode;

typedef enum
{
  RT_FILTER_NEAREST,
  RT_FILTER_LINEAR,
  RT_FILTER_NONE
} RTfiltermode;

typedef enum
{
  RT_TEXTURE_READ_ELEMENT_TYPE,
  RT_TEXTURE_READ_NORMALIZED_FLOAT
} RTtexturereadmode;

typedef enum
{
  RT_TARGET_GL_TEXTURE_2D,
  RT_TARGET_GL_TEXTURE_RECTANGLE,
  RT_TARGET_GL_TEXTURE_3D,
  RT_TARGET_GL_RENDER_BUFFER
} RTgltarget;

typedef enum
{
  RT_TEXTURE_INDEX_NORMALIZED_COORDINATES,
  RT_TEXTURE_INDEX_ARRAY_INDEX
} RTtextureindexmode;

typedef enum
{
  RT_BUFFER_INPUT                = 0x1,
  RT_BUFFER_OUTPUT               = 0x2,
  RT_BUFFER_INPUT_OUTPUT         = RT_BUFFER_INPUT | RT_BUFFER_OUTPUT
} RTbuffertype;

typedef enum
{
  RT_BUFFER_GPU_LOCAL            = 0x4,
  RT_BUFFER_COPY_ON_DIRTY        = 0x8
} RTbufferflag;

typedef enum
{
  RT_EXCEPTION_INDEX_OUT_OF_BOUNDS          = 0x3FB,
  RT_EXCEPTION_STACK_OVERFLOW               = 0x3FC,
  RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS   = 0x3FD,
  RT_EXCEPTION_INVALID_RAY                  = 0x3FE,
  RT_EXCEPTION_INTERNAL_ERROR               = 0x3FF,
  RT_EXCEPTION_USER                         = 0x400,

  RT_EXCEPTION_ALL                          = 0x7FFFFFFF
} RTexception;

typedef enum
{
  RT_SUCCESS                           = 0,

  RT_TIMEOUT_CALLBACK                  = 0x100,

  RT_ERROR_INVALID_CONTEXT             = 0x500,
  RT_ERROR_INVALID_VALUE               = 0x501,
  RT_ERROR_MEMORY_ALLOCATION_FAILED    = 0x502,
  RT_ERROR_TYPE_MISMATCH               = 0x503,
  RT_ERROR_VARIABLE_NOT_FOUND          = 0x504,
  RT_ERROR_VARIABLE_REDECLARED         = 0x505,
  RT_ERROR_ILLEGAL_SYMBOL              = 0x506,
  RT_ERROR_INVALID_SOURCE              = 0x507,
  RT_ERROR_VERSION_MISMATCH            = 0x508,

  RT_ERROR_OBJECT_CREATION_FAILED      = 0x600,
  RT_ERROR_NO_DEVICE                   = 0x601,
  RT_ERROR_INVALID_DEVICE              = 0x602,
  RT_ERROR_INVALID_IMAGE               = 0x603,
  RT_ERROR_FILE_NOT_FOUND              = 0x604,
  RT_ERROR_ALREADY_MAPPED              = 0x605,
  RT_ERROR_INVALID_DRIVER_VERSION      = 0x606,
  RT_ERROR_CONTEXT_CREATION_FAILED     = 0x607,

  RT_ERROR_RESOURCE_NOT_REGISTERED     = 0x608,
  RT_ERROR_RESOURCE_ALREADY_REGISTERED = 0x609,

  RT_ERROR_LAUNCH_FAILED               = 0x900,

  RT_ERROR_UNKNOWN                     = ~0
} RTresult;

typedef enum
{
  RT_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
  RT_DEVICE_ATTRIBUTE_CLOCK_RATE,
  RT_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
  RT_DEVICE_ATTRIBUTE_EXECUTION_TIMEOUT_ENABLED,
  RT_DEVICE_ATTRIBUTE_MAX_HARDWARE_TEXTURE_COUNT,
  RT_DEVICE_ATTRIBUTE_NAME,
  RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY,
  RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY,
  RT_DEVICE_ATTRIBUTE_TCC_DRIVER,                 /* sizeof(int) */
  RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL         /* sizeof(int) */
} RTdeviceattribute;

typedef enum
{
  RT_CONTEXT_ATTRIBUTE_MAX_TEXTURE_COUNT,                    /* sizeof(int)    */
  RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS,                      /* sizeof(int)    */
  RT_CONTEXT_ATTRIBUTE_USED_HOST_MEMORY,                     /* sizeof(RTsize) */
  RT_CONTEXT_ATTRIBUTE_GPU_PAGING_ACTIVE,                    /* sizeof(int)    */
  RT_CONTEXT_ATTRIBUTE_GPU_PAGING_FORCED_OFF,                /* sizeof(int)    */
  RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY = 0x10000000  /* sizeof(RTsize) */
} RTcontextattribute;


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* __optix_optix_declarations_h__ */

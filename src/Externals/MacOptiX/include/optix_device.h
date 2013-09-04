
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
 * optix_cuda.h
 *
 * This file provides the nvcc interface for generating PTX that the OptiX is
 * capable of parsing and weaving into the final kernel.  This is included by
 * optix.h automatically if compiling device code.  It can be included explicitly
 * in host code if desired.
 *
\******************************************************************************/

#ifndef __optix_optix_cuda__internal_h__
#define __optix_optix_cuda__internal_h__

#include "internal/optix_datatypes.h"
#include "internal/optix_declarations.h"
#include "internal/optix_internal.h"


/*
  Augment vector types
*/

namespace optix {
  
  template<typename T, int Dim> struct VectorTypes {};
  template<> struct VectorTypes<int, 1> {
    typedef int Type;
    template<class S> static __device__ __forceinline__
      Type make(S s) { return make_int(s); }
  };
  template<> struct VectorTypes<int, 2> {
    typedef int2 Type;
    template<class S> static __device__ __forceinline__
      Type make(S s) { return make_int2(s); }
  };
  template<> struct VectorTypes<int, 3> {
    typedef int3 Type;
    template<class S> static __device__ __forceinline__
      Type make(S s) { return make_int3(s); }
  };
  template<> struct VectorTypes<int, 4> {
    typedef int4 Type;
    template<class S> static __device__ __forceinline__
      Type make(S s) { return make_int4(s); }
  };
  template<> struct VectorTypes<unsigned int, 1> {
    typedef unsigned int Type;
    static __device__ __forceinline__
      Type make(unsigned int s) { return s; }
    template<class S> static __device__ __forceinline__
      Type make(S s) { return (unsigned int)s.x; }
  };
  template<> struct VectorTypes<unsigned int, 2> {
    typedef uint2 Type;
    template<class S> static __device__ __forceinline__
      Type make(S s) { return make_uint2(s); }
  };
  template<> struct VectorTypes<unsigned int, 3> {
    typedef uint3 Type;
    template<class S> static __device__ __forceinline__
      Type make(S s) { return make_uint3(s); }
  };
  template<> struct VectorTypes<unsigned int, 4> {
    typedef uint4 Type;
    template<class S> static __device__ __forceinline__
      Type make(S s) { return make_uint4(s); }
  };
  template<> struct VectorTypes<float, 1> {
    typedef float Type;
    template<class S> static __device__ __forceinline__
      Type make(S s) { return make_float(s); }
  };
  template<> struct VectorTypes<float, 2> {
    typedef float2 Type;
    template<class S> static __device__ __forceinline__
      Type make(S s) { return make_float2(s); }
  };
  template<> struct VectorTypes<float, 3> {
    typedef float3 Type;
    template<class S> static __device__ __forceinline__
      Type make(S s) { return make_float3(s); }
  };
  template<> struct VectorTypes<float, 4> {
    typedef float4 Type;
    template<class S> static __device__ __forceinline__
      Type make(S s) { return make_float4(s); }
  };

#if defined(__APPLE__) || defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
  template<> struct VectorTypes<size_t, 1> {
    typedef size_t Type;
    static __device__ __forceinline__
      Type make(unsigned int s) { return s; }
    template<class S> static __device__ __forceinline__
      Type make(S s) { return (unsigned int)s.x; }
  };
  template<> struct VectorTypes<size_t, 2> {
    typedef size_t2 Type;
    template<class S> static __device__ __forceinline__
      Type make(S s) { return make_size_t2(s); }
  };
  template<> struct VectorTypes<size_t, 3> {
    typedef size_t3 Type;
    template<class S> static __device__ __forceinline__
      Type make(S s) { return make_size_t3(s); }
  };
  template<> struct VectorTypes<size_t, 4> {
    typedef size_t4 Type;
    template<class S> static __device__ __forceinline__
      Type make(S s) { return make_size_t4(s); }
  };
#endif
}


/*
   Variables
*/

struct rtObject {
private:
  unsigned int handle;
};

#define rtDeclareVariable(type, name, semantic, annotation)    \
  namespace rti_internal_typeinfo { \
    __device__ ::rti_internal_typeinfo::rti_typeinfo name = { ::rti_internal_typeinfo::_OPTIX_VARIABLE, sizeof(type)}; \
  } \
  namespace rti_internal_typename { \
    __device__ char name[] = #type; \
  } \
  namespace rti_internal_semantic { \
    __device__ char name[] = #semantic; \
  } \
  namespace rti_internal_annotation { \
    __device__ char name[] = #annotation; \
  } \
  __device__ type name

#define rtDeclareAnnotation(variable, annotation) \
  namespace rti_internal_annotation { \
    __device__ char variable[] = #annotation; \
  }

/* Declares a function that can be set via rtVariableSetObject and called from CUDA C.
   Once declared the function variable can be used as if it were a regular function.  Note
   that the parameters argument to the macro need to have parentheses even if they will be
   empty.

   Example: rtCallableProgram(float, times2, (float));
   Example: rtCallableProgram(float, doStuff, ());
 */

template<typename T> struct rtCallableProgramSizeofWrapper { static const size_t value = sizeof(T); }; 
template<> struct rtCallableProgramSizeofWrapper<void> { static const size_t value = 0; };

#define rtCallableProgram(returnType, name, parameters)         \
  namespace rti_internal_typeinfo {                                     \
    __device__ ::rti_internal_typeinfo::rti_typeinfo name = { ::rti_internal_typeinfo::_OPTIX_VARIABLE, rtCallableProgramSizeofWrapper<returnType>::value }; \
  }                                                                     \
  namespace rti_internal_typename {                                     \
    __device__ char name[] = #returnType;                               \
  }                                                                     \
  namespace rti_internal_semantic {                                     \
    __device__ char name[] = "rt_call";                                 \
  }                                                                     \
  namespace rti_internal_annotation {                                   \
    __device__ char name[] = #parameters;                               \
  }                                                                     \
  __noinline__ __device__ returnType name parameters { typedef returnType localtype; return localtype(); }

/* Helper functions for converting pointers to/from integers so we can support
   call by reference on SM1.x */

namespace optix {

  typedef unsigned int rtPickledLocalPointer;
  static inline __device__ rtPickledLocalPointer rtPickleLocalPointer( void *p ) {
    return optix::rt_pickle_pointer( p );
  }

  static inline __device__ void * rtUnpickleLocalPointer( rtPickledLocalPointer p ) {
    return optix::rt_unpickle_pointer( p );
  }
}

/*
   Buffer
*/

namespace optix {
  template<typename T, int Dim = 1> struct buffer {
    typedef VectorTypes<size_t, Dim> WrapperType;
    typedef typename VectorTypes<size_t, Dim>::Type IndexType;

    __device__ __forceinline__ IndexType size() const {
      return WrapperType::make(rt_buffer_get_size(this, Dim, sizeof(T)));
    }
    __device__ __forceinline__ T& operator[](IndexType i) {
      size_t4 c = make_index(i);
      return *(T*)rt_buffer_get(this, Dim, sizeof(T), c.x, c.y, c.z, c.w);
    }
  private:
    __inline__ __device__ size_t4 make_index(size_t v0) { return make_size_t4(v0, 0, 0, 0); }
    __inline__ __device__ size_t4 make_index(size_t2 v0) { return make_size_t4(v0.x, v0.y, 0, 0); }
    __inline__ __device__ size_t4 make_index(size_t3 v0) { return make_size_t4(v0.x, v0.y, v0.z, 0); }
    __inline__ __device__ size_t4 make_index(size_t4 v0) { return make_size_t4(v0.x, v0.y, v0.z, v0.w); }
  };
}
#define rtBuffer __device__ optix::buffer

/*
   Texture - they are defined in CUDA
*/

#define rtTextureSampler texture

namespace optix {

  typedef int rtTextureId;
 
  template<typename T>
  inline __device__ T rtTex1D(rtTextureId id, float x);
  template<>
  inline __device__ float4 rtTex1D(rtTextureId id, float x)
  {
    return optix::rt_texture_get_f_id(id, 1, x, 0, 0, 0);
  }
  template<>
  inline __device__ int4 rtTex1D(rtTextureId id, float x)
  {
    return optix::rt_texture_get_i_id(id, 1, x, 0, 0, 0);
  }
  template<>
  inline __device__ uint4 rtTex1D(rtTextureId id, float x)
  {
    return optix::rt_texture_get_u_id(id, 1, x, 0, 0, 0);
  }

  template<> inline __device__ unsigned char rtTex1D(rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    return (unsigned char)(tmp.x);
  }
  template<> inline __device__ char rtTex1D(rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    return (char)(tmp.x);
  }
  template<> inline __device__ unsigned short rtTex1D(rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    return (unsigned short)(tmp.x);
  }
  template<> inline __device__ short rtTex1D(rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    return (short)(tmp.x);
  }
  template<> inline __device__ int rtTex1D(rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    return tmp.x;
  }
  template<> inline __device__ unsigned int rtTex1D(rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    return tmp.x;
  }
  template<> inline __device__ uchar1 rtTex1D(rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    return make_uchar1(tmp.x);
  }
  template<> inline __device__ char1 rtTex1D(rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    return make_char1(tmp.x);
  }
  template<> inline __device__ ushort1 rtTex1D(rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    return make_ushort1(tmp.x);
  }
  template<> inline __device__ short1 rtTex1D(rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    return make_short1(tmp.x);
  }
  template<> inline __device__ uint1 rtTex1D(rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    return make_uint1(tmp.x);
  }
  template<> inline __device__ int1 rtTex1D(rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    return make_int1(tmp.x);
  }
  template<> inline __device__ float rtTex1D(rtTextureId id, float x)
  {
    float4 tmp = rtTex1D<float4>(id, x);
    return tmp.x;
  }
  template<> inline __device__ uchar2 rtTex1D(rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    return make_uchar2(tmp.x, tmp.y);
  }
  template<> inline __device__ char2 rtTex1D(rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    return make_char2(tmp.x, tmp.y);
  }
  template<> inline __device__ ushort2 rtTex1D(rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    return make_ushort2(tmp.x, tmp.y);
  }
  template<> inline __device__ short2 rtTex1D(rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    return make_short2(tmp.x, tmp.y);
  }
  template<> inline __device__ uint2 rtTex1D(rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    return make_uint2(tmp.x, tmp.y);
  }
  template<> inline __device__ int2 rtTex1D(rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    return make_int2(tmp.x, tmp.y);
  }
  template<> inline __device__ float2 rtTex1D(rtTextureId id, float x)
  {
    float4 tmp = rtTex1D<float4>(id, x);
    return ::make_float2(tmp.x, tmp.y);
  }
  template<> inline __device__ uchar4 rtTex1D(rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    return make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
  }
  template<> inline __device__ char4 rtTex1D(rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    return make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
  }
  template<> inline __device__ ushort4 rtTex1D(rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    return make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
  }
  template<> inline __device__ short4 rtTex1D(rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    return make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
  }

  inline __device__ void rtTex1D(unsigned char *retVal, rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    *retVal = (unsigned char)(tmp.x);
  }
  inline __device__ void rtTex1D(char *retVal, rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    *retVal = (char)(tmp.x);
  }
  inline __device__ void rtTex1D(unsigned short *retVal, rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    *retVal = (unsigned short)(tmp.x);
  }
  inline __device__ void rtTex1D(short *retVal, rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    *retVal = (short)(tmp.x);
  }
  inline __device__ void rtTex1D(unsigned int *retVal, rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    *retVal = tmp.x;
  }
  inline __device__ void rtTex1D(int *retVal, rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    *retVal = tmp.x;
  }
  inline __device__ void rtTex1D(uchar1 *retVal, rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    *retVal = make_uchar1(tmp.x);
  }
  inline __device__ void rtTex1D(char1 *retVal, rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    *retVal = make_char1(tmp.x);
  }
  inline __device__ void rtTex1D(ushort1 *retVal, rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    *retVal = make_ushort1(tmp.x);
  }
  inline __device__ void rtTex1D(short1 *retVal, rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    *retVal = make_short1(tmp.x);
  }
  inline __device__ void rtTex1D(uint1 *retVal, rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    *retVal = make_uint1(tmp.x);
  }
  inline __device__ void rtTex1D(int1 *retVal, rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    *retVal = make_int1(tmp.x);
  }
  inline __device__ void rtTex1D(float *retVal, rtTextureId id, float x)
  {
    float4 tmp = rtTex1D<float4>(id, x);
    *retVal = tmp.x;
  }
  inline __device__ void rtTex1D(uchar2 *retVal, rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    *retVal = make_uchar2(tmp.x, tmp.y);
  }
  inline __device__ void rtTex1D(char2 *retVal, rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    *retVal = make_char2(tmp.x, tmp.y);
  }
  inline __device__ void rtTex1D(ushort2 *retVal, rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    *retVal = make_ushort2(tmp.x, tmp.y);
  }
  inline __device__ void rtTex1D(short2 *retVal, rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    *retVal = make_short2(tmp.x, tmp.y);
  }
  inline __device__ void rtTex1D(uint2 *retVal, rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    *retVal = make_uint2(tmp.x, tmp.y);
  }
  inline __device__ void rtTex1D(int2 *retVal, rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    *retVal = make_int2(tmp.x, tmp.y);
  }
  inline __device__ void rtTex1D(float2 *retVal, rtTextureId id, float x)
  {
    float4 tmp = rtTex1D<float4>(id, x);
    *retVal = ::make_float2(tmp.x, tmp.y);
  }
  inline __device__ void rtTex1D(uchar4 *retVal, rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
  }
  inline __device__ void rtTex1D(char4 *retVal, rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
  }
  inline __device__ void rtTex1D(ushort4 *retVal, rtTextureId id, float x)
  {
    uint4 tmp = rtTex1D<uint4>(id, x);
    *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
  }
  inline __device__ void rtTex1D(short4 *retVal, rtTextureId id, float x)
  {
    int4 tmp = rtTex1D<int4>(id, x);
    *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
  }

  template<typename T>
  inline __device__ T rtTex2D(rtTextureId id, float x, float y);
  template<>
  inline __device__ float4 rtTex2D(rtTextureId id, float x, float y)
  {
    return optix::rt_texture_get_f_id(id, 2, x, y, 0, 0);
  }
  template<>
  inline __device__ int4 rtTex2D(rtTextureId id, float x, float y)
  {
    return optix::rt_texture_get_i_id(id, 2, x, y, 0, 0);
  }
  template<>
  inline __device__ uint4 rtTex2D(rtTextureId id, float x, float y)
  {
    return optix::rt_texture_get_u_id(id, 2, x, y, 0, 0);
  }

  template<> inline __device__ unsigned char rtTex2D(rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    return (unsigned char)(tmp.x);
  }
  template<> inline __device__ char rtTex2D(rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    return (char)(tmp.x);
  }
  template<> inline __device__ unsigned short rtTex2D(rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    return (unsigned short)(tmp.x);
  }
  template<> inline __device__ short rtTex2D(rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    return (short)(tmp.x);
  }
  template<> inline __device__ int rtTex2D(rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    return tmp.x;
  }
  template<> inline __device__ unsigned int rtTex2D(rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    return tmp.x;
  }
  template<> inline __device__ uchar1 rtTex2D(rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    return make_uchar1(tmp.x);
  }
  template<> inline __device__ char1 rtTex2D(rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    return make_char1(tmp.x);
  }
  template<> inline __device__ ushort1 rtTex2D(rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    return make_ushort1(tmp.x);
  }
  template<> inline __device__ short1 rtTex2D(rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    return make_short1(tmp.x);
  }
  template<> inline __device__ uint1 rtTex2D(rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    return make_uint1(tmp.x);
  }
  template<> inline __device__ int1 rtTex2D(rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    return make_int1(tmp.x);
  }
  template<> inline __device__ float rtTex2D(rtTextureId id, float x, float y)
  {
    float4 tmp = rtTex2D<float4>(id, x, y);
    return tmp.x;
  }
  template<> inline __device__ uchar2 rtTex2D(rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    return make_uchar2(tmp.x, tmp.y);
  }
  template<> inline __device__ char2 rtTex2D(rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    return make_char2(tmp.x, tmp.y);
  }
  template<> inline __device__ ushort2 rtTex2D(rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    return make_ushort2(tmp.x, tmp.y);
  }
  template<> inline __device__ short2 rtTex2D(rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    return make_short2(tmp.x, tmp.y);
  }
  template<> inline __device__ uint2 rtTex2D(rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    return make_uint2(tmp.x, tmp.y);
  }
  template<> inline __device__ int2 rtTex2D(rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    return make_int2(tmp.x, tmp.y);
  }
  template<> inline __device__ float2 rtTex2D(rtTextureId id, float x, float y)
  {
    float4 tmp = rtTex2D<float4>(id, x, y);
    return ::make_float2(tmp.x, tmp.y);
  }
  template<> inline __device__ uchar4 rtTex2D(rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    return make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
  }
  template<> inline __device__ char4 rtTex2D(rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    return make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
  }
  template<> inline __device__ ushort4 rtTex2D(rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    return make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
  }
  template<> inline __device__ short4 rtTex2D(rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    return make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
  }

  inline __device__ void rtTex2D(unsigned char *retVal, rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    *retVal = (unsigned char)(tmp.x);
  }
  inline __device__ void rtTex2D(char *retVal, rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    *retVal = (char)(tmp.x);
  }
  inline __device__ void rtTex2D(unsigned short *retVal, rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    *retVal = (unsigned short)(tmp.x);
  }
  inline __device__ void rtTex2D(short *retVal, rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    *retVal = (short)(tmp.x);
  }
  inline __device__ void rtTex2D(unsigned int *retVal, rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    *retVal = tmp.x;
  }
  inline __device__ void rtTex2D(int *retVal, rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    *retVal = tmp.x;
  }
  inline __device__ void rtTex2D(uchar1 *retVal, rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    *retVal = make_uchar1(tmp.x);
  }
  inline __device__ void rtTex2D(char1 *retVal, rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    *retVal = make_char1(tmp.x);
  }
  inline __device__ void rtTex2D(ushort1 *retVal, rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    *retVal = make_ushort1(tmp.x);
  }
  inline __device__ void rtTex2D(short1 *retVal, rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    *retVal = make_short1(tmp.x);
  }
  inline __device__ void rtTex2D(uint1 *retVal, rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    *retVal = make_uint1(tmp.x);
  }
  inline __device__ void rtTex2D(int1 *retVal, rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    *retVal = make_int1(tmp.x);
  }
  inline __device__ void rtTex2D(float *retVal, rtTextureId id, float x, float y)
  {
    float4 tmp = rtTex2D<float4>(id, x, y);
    *retVal = tmp.x;
  }
  inline __device__ void rtTex2D(uchar2 *retVal, rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    *retVal = make_uchar2(tmp.x, tmp.y);
  }
  inline __device__ void rtTex2D(char2 *retVal, rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    *retVal = make_char2(tmp.x, tmp.y);
  }
  inline __device__ void rtTex2D(ushort2 *retVal, rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    *retVal = make_ushort2(tmp.x, tmp.y);
  }
  inline __device__ void rtTex2D(short2 *retVal, rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    *retVal = make_short2(tmp.x, tmp.y);
  }
  inline __device__ void rtTex2D(uint2 *retVal, rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    *retVal = make_uint2(tmp.x, tmp.y);
  }
  inline __device__ void rtTex2D(int2 *retVal, rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    *retVal = make_int2(tmp.x, tmp.y);
  }  
  inline __device__ void rtTex2D(float2 *retVal, rtTextureId id, float x, float y)
  {
    float4 tmp = rtTex2D<float4>(id, x, y);
    *retVal = ::make_float2(tmp.x, tmp.y);
  }
  inline __device__ void rtTex2D(uchar4 *retVal, rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
  }
  inline __device__ void rtTex2D(char4 *retVal, rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
  }
  inline __device__ void rtTex2D(ushort4 *retVal, rtTextureId id, float x, float y)
  {
    uint4 tmp = rtTex2D<uint4>(id, x, y);
    *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
  }
  inline __device__ void rtTex2D(short4 *retVal, rtTextureId id, float x, float y)
  {
    int4 tmp = rtTex2D<int4>(id, x, y);
    *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
  }

  template<typename T>
  inline __device__ T rtTex3D(rtTextureId id, float x, float y, float z);
  template<>
  inline __device__ float4 rtTex3D(rtTextureId id, float x, float y, float z)
  {
    return optix::rt_texture_get_f_id(id, 3, x, y, z, 0);
  }
  template<>
  inline __device__ int4 rtTex3D(rtTextureId id, float x, float y, float z)
  {
    return optix::rt_texture_get_i_id(id, 3, x, y, z, 0);
  }
  template<>
  inline __device__ uint4 rtTex3D(rtTextureId id, float x, float y, float z)
  {
    return optix::rt_texture_get_u_id(id, 3, x, y, z, 0);
  }

  template<> inline __device__ unsigned char rtTex3D(rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    return (unsigned char)(tmp.x);
  }
  template<> inline __device__ char rtTex3D(rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    return (char)(tmp.x);
  }
  template<> inline __device__ unsigned short rtTex3D(rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    return (unsigned short)(tmp.x);
  }
  template<> inline __device__ short rtTex3D(rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    return (short)(tmp.x);
  }
  template<> inline __device__ int rtTex3D(rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    return tmp.x;
  }
  template<> inline __device__ unsigned int rtTex3D(rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    return tmp.x;
  }
  template<> inline __device__ uchar1 rtTex3D(rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    return make_uchar1(tmp.x);
  }
  template<> inline __device__ char1 rtTex3D(rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    return make_char1(tmp.x);
  }
  template<> inline __device__ ushort1 rtTex3D(rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    return make_ushort1(tmp.x);
  }
  template<> inline __device__ short1 rtTex3D(rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    return make_short1(tmp.x);
  }
  template<> inline __device__ uint1 rtTex3D(rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    return make_uint1(tmp.x);
  }
  template<> inline __device__ int1 rtTex3D(rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    return make_int1(tmp.x);
  }
  template<> inline __device__ float rtTex3D(rtTextureId id, float x, float y, float z)
  {
    float4 tmp = rtTex3D<float4>(id, x, y, z);
    return tmp.x;
  }
  template<> inline __device__ uchar2 rtTex3D(rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    return make_uchar2(tmp.x, tmp.y);
  }
  template<> inline __device__ char2 rtTex3D(rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    return make_char2(tmp.x, tmp.y);
  }
  template<> inline __device__ ushort2 rtTex3D(rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    return make_ushort2(tmp.x, tmp.y);
  }
  template<> inline __device__ short2 rtTex3D(rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    return make_short2(tmp.x, tmp.y);
  }
  template<> inline __device__ uint2 rtTex3D(rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    return make_uint2(tmp.x, tmp.y);
  }
  template<> inline __device__ int2 rtTex3D(rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    return make_int2(tmp.x, tmp.y);
  }  
  template<> inline __device__ float2 rtTex3D(rtTextureId id, float x, float y, float z)
  {
    float4 tmp = rtTex3D<float4>(id, x, y, z);
    return ::make_float2(tmp.x, tmp.y);
  }
  template<> inline __device__ uchar4 rtTex3D(rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    return make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
  }
  template<> inline __device__ char4 rtTex3D(rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    return make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
  }
  template<> inline __device__ ushort4 rtTex3D(rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    return make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
  }
  template<> inline __device__ short4 rtTex3D(rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    return make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
  }

  inline __device__ void rtTex3D(unsigned char *retVal, rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    *retVal = (unsigned char)(tmp.x);
  }
  inline __device__ void rtTex3D(char *retVal, rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    *retVal = (char)(tmp.x);
  }
  inline __device__ void rtTex3D(unsigned short *retVal, rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    *retVal = (unsigned short)(tmp.x);
  }
  inline __device__ void rtTex3D(short *retVal, rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    *retVal = (short)(tmp.x);
  }
  inline __device__ void rtTex3D(unsigned int *retVal, rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    *retVal = tmp.x;
  }
  inline __device__ void rtTex3D(int *retVal, rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    *retVal = tmp.x;
  }
  inline __device__ void rtTex3D(uchar1 *retVal, rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    *retVal = make_uchar1(tmp.x);
  }
  inline __device__ void rtTex3D(char1 *retVal, rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    *retVal = make_char1(tmp.x);
  }
  inline __device__ void rtTex3D(ushort1 *retVal, rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    *retVal = make_ushort1(tmp.x);
  }
  inline __device__ void rtTex3D(short1 *retVal, rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    *retVal = make_short1(tmp.x);
  }
  inline __device__ void rtTex3D(uint1 *retVal, rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    *retVal = make_uint1(tmp.x);
  }
  inline __device__ void rtTex3D(int1 *retVal, rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    *retVal = make_int1(tmp.x);
  }
  inline __device__ void rtTex3D(float *retVal, rtTextureId id, float x, float y, float z)
  {
    float4 tmp = rtTex3D<float4>(id, x, y, z);
    *retVal = tmp.x;
  }
  inline __device__ void rtTex3D(uchar2 *retVal, rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    *retVal = make_uchar2(tmp.x, tmp.y);
  }
  inline __device__ void rtTex3D(char2 *retVal, rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    *retVal = make_char2(tmp.x, tmp.y);
  }
  inline __device__ void rtTex3D(ushort2 *retVal, rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    *retVal = make_ushort2(tmp.x, tmp.y);
  }
  inline __device__ void rtTex3D(short2 *retVal, rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    *retVal = make_short2(tmp.x, tmp.y);
  }
  inline __device__ void rtTex3D(uint2 *retVal, rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    *retVal = make_uint2(tmp.x, tmp.y);
  }
  inline __device__ void rtTex3D(int2 *retVal, rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    *retVal = make_int2(tmp.x, tmp.y);
  }
  inline __device__ void rtTex3D(float2 *retVal, rtTextureId id, float x, float y, float z)
  {
    float4 tmp = rtTex3D<float4>(id, x, y, z);
    *retVal = ::make_float2(tmp.x, tmp.y);
  }
  inline __device__ void rtTex3D(uchar4 *retVal, rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    *retVal = make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w);
  }
  inline __device__ void rtTex3D(char4 *retVal, rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    *retVal = make_char4(tmp.x, tmp.y, tmp.z, tmp.w);
  }
  inline __device__ void rtTex3D(ushort4 *retVal, rtTextureId id, float x, float y, float z)
  {
    uint4 tmp = rtTex3D<uint4>(id, x, y, z);
    *retVal = make_ushort4(tmp.x, tmp.y, tmp.z, tmp.w);
  }
  inline __device__ void rtTex3D(short4 *retVal, rtTextureId id, float x, float y, float z)
  {
    int4 tmp = rtTex3D<int4>(id, x, y, z);
    *retVal = make_short4(tmp.x, tmp.y, tmp.z, tmp.w);
  }

};

/*
   Program
*/

#define RT_PROGRAM __global__

/* This is used to declare programs that can be attached to variables and called from
 * within other RT_PROGRAMS.
 *
 * There are some limitations with PTX that is targetted at sm_1x devices.
 *
 * 1. Functions declared with RT_CALLABLE_PROGRAM will not be emitted in the PTX unless
 *    another function calls it.  This can be fixed by declaring a __global__ helper
 *    function that calls the desired function.
 *
 *    RT_CALLABLE_PROGRAM
 *    float3 simple_shade(float multiplier,  float3 input_color)
 *    {
 *      return multiplier * input_color;
 *    }
 *
 *    #if __CUDA_ARCH__ < 200
 *    __global__ void stub() {
 *       (void) simple_shade( 0, make_float3(0,0,0) );
 *    }
 *    #endif
 *
 * 2. You can't pass pointers to functions or use integers for pointers.  In the first
 *    case CUDA will force the inline of the proxy function removing the call altogether,
 *    and in the case of passing pointers as integers, CUDA will assume that any pointer
 *    that was cast from an integer will point to global memory and could cause errors
 *    when loading from that pointer.  If you need to pass pointers, you should target
 *    sm_20.
 */
   
#define RT_CALLABLE_PROGRAM __device__ __noinline__



/*
   Functions
*/

template<class T>
static inline __device__ void rtTrace( rtObject topNode, optix::Ray ray, T& prd )
{
  optix::rt_trace(*(unsigned int*)&topNode, ray.origin, ray.direction, ray.ray_type, ray.tmin, ray.tmax, &prd, sizeof(T));
}

static inline __device__ bool rtPotentialIntersection( float tmin )
{
  return optix::rt_potential_intersection( tmin );
}

static inline __device__ bool rtReportIntersection( unsigned int material )
{
  return optix::rt_report_intersection( material );
}

static inline __device__ void rtIgnoreIntersection()
{
  optix::rt_ignore_intersection();
}

static inline __device__ void rtTerminateRay()
{
  optix::rt_terminate_ray();
}

static inline __device__ void rtIntersectChild( unsigned int index )
{
  optix::rt_intersect_child( index );
}

static inline __device__ float3 rtTransformPoint( RTtransformkind kind, const float3& p )
{
  return optix::rt_transform_point( kind, p );
}

static inline __device__ float3 rtTransformVector( RTtransformkind kind, const float3& v )
{
  return optix::rt_transform_vector( kind, v );
}

static inline __device__ float3 rtTransformNormal( RTtransformkind kind, const float3& n )
{
  return optix::rt_transform_normal( kind, n );
}

static inline __device__ void rtGetTransform( RTtransformkind kind, float matrix[16] )
{
  return optix::rt_get_transform( kind, matrix );
}


/*
   Printing
*/

static inline __device__ void rtPrintf( const char* fmt )
{
  _RT_PRINTF_1();
  optix::rt_print_start(fmt,sz);
}

template<typename T1>
static inline __device__ void rtPrintf( const char* fmt, T1 arg1 )
{
  _RT_PRINTF_1();
  _RT_PRINTF_ARG_1( arg1 );
  _RT_PRINTF_2();
  _RT_PRINTF_ARG_2( arg1 );
}

template<typename T1, typename T2>
static inline __device__ void rtPrintf( const char* fmt, T1 arg1, T2 arg2 )
{
  _RT_PRINTF_1();
  _RT_PRINTF_ARG_1( arg1 );
  _RT_PRINTF_ARG_1( arg2 );
  _RT_PRINTF_2();
  _RT_PRINTF_ARG_2( arg1 );
  _RT_PRINTF_ARG_2( arg2 );
}

template<typename T1, typename T2, typename T3>
static inline __device__ void rtPrintf( const char* fmt, T1 arg1, T2 arg2, T3 arg3 )
{
  _RT_PRINTF_1();
  _RT_PRINTF_ARG_1( arg1 );
  _RT_PRINTF_ARG_1( arg2 );
  _RT_PRINTF_ARG_1( arg3 );
  _RT_PRINTF_2();
  _RT_PRINTF_ARG_2( arg1 );
  _RT_PRINTF_ARG_2( arg2 );
  _RT_PRINTF_ARG_2( arg3 );
}

template<typename T1, typename T2, typename T3, typename T4>
static inline __device__ void rtPrintf( const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4 )
{
  _RT_PRINTF_1();
  _RT_PRINTF_ARG_1( arg1 );
  _RT_PRINTF_ARG_1( arg2 );
  _RT_PRINTF_ARG_1( arg3 );
  _RT_PRINTF_ARG_1( arg4 );
  _RT_PRINTF_2();
  _RT_PRINTF_ARG_2( arg1 );
  _RT_PRINTF_ARG_2( arg2 );
  _RT_PRINTF_ARG_2( arg3 );
  _RT_PRINTF_ARG_2( arg4 );
}

template<typename T1, typename T2, typename T3, typename T4, typename T5>
static inline __device__ void rtPrintf( const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5 )
{
  _RT_PRINTF_1();
  _RT_PRINTF_ARG_1( arg1 );
  _RT_PRINTF_ARG_1( arg2 );
  _RT_PRINTF_ARG_1( arg3 );
  _RT_PRINTF_ARG_1( arg4 );
  _RT_PRINTF_ARG_1( arg5 );
  _RT_PRINTF_2();
  _RT_PRINTF_ARG_2( arg1 );
  _RT_PRINTF_ARG_2( arg2 );
  _RT_PRINTF_ARG_2( arg3 );
  _RT_PRINTF_ARG_2( arg4 );
  _RT_PRINTF_ARG_2( arg5 );
}

template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
static inline __device__ void rtPrintf( const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6 )
{
  _RT_PRINTF_1();
  _RT_PRINTF_ARG_1( arg1 );
  _RT_PRINTF_ARG_1( arg2 );
  _RT_PRINTF_ARG_1( arg3 );
  _RT_PRINTF_ARG_1( arg4 );
  _RT_PRINTF_ARG_1( arg5 );
  _RT_PRINTF_ARG_1( arg6 );
  _RT_PRINTF_2();
  _RT_PRINTF_ARG_2( arg1 );
  _RT_PRINTF_ARG_2( arg2 );
  _RT_PRINTF_ARG_2( arg3 );
  _RT_PRINTF_ARG_2( arg4 );
  _RT_PRINTF_ARG_2( arg5 );
  _RT_PRINTF_ARG_2( arg6 );
}

template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
static inline __device__ void rtPrintf( const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7 )
{
  _RT_PRINTF_1();
  _RT_PRINTF_ARG_1( arg1 );
  _RT_PRINTF_ARG_1( arg2 );
  _RT_PRINTF_ARG_1( arg3 );
  _RT_PRINTF_ARG_1( arg4 );
  _RT_PRINTF_ARG_1( arg5 );
  _RT_PRINTF_ARG_1( arg6 );
  _RT_PRINTF_ARG_1( arg7 );
  _RT_PRINTF_2();
  _RT_PRINTF_ARG_2( arg1 );
  _RT_PRINTF_ARG_2( arg2 );
  _RT_PRINTF_ARG_2( arg3 );
  _RT_PRINTF_ARG_2( arg4 );
  _RT_PRINTF_ARG_2( arg5 );
  _RT_PRINTF_ARG_2( arg6 );
  _RT_PRINTF_ARG_2( arg7 );
}

template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
static inline __device__ void rtPrintf( const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8 )
{
  _RT_PRINTF_1();
  _RT_PRINTF_ARG_1( arg1 );
  _RT_PRINTF_ARG_1( arg2 );
  _RT_PRINTF_ARG_1( arg3 );
  _RT_PRINTF_ARG_1( arg4 );
  _RT_PRINTF_ARG_1( arg5 );
  _RT_PRINTF_ARG_1( arg6 );
  _RT_PRINTF_ARG_1( arg7 );
  _RT_PRINTF_ARG_1( arg8 );
  _RT_PRINTF_2();
  _RT_PRINTF_ARG_2( arg1 );
  _RT_PRINTF_ARG_2( arg2 );
  _RT_PRINTF_ARG_2( arg3 );
  _RT_PRINTF_ARG_2( arg4 );
  _RT_PRINTF_ARG_2( arg5 );
  _RT_PRINTF_ARG_2( arg6 );
  _RT_PRINTF_ARG_2( arg7 );
  _RT_PRINTF_ARG_2( arg8 );
}

template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
static inline __device__ void rtPrintf( const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9 )
{
  _RT_PRINTF_1();
  _RT_PRINTF_ARG_1( arg1 );
  _RT_PRINTF_ARG_1( arg2 );
  _RT_PRINTF_ARG_1( arg3 );
  _RT_PRINTF_ARG_1( arg4 );
  _RT_PRINTF_ARG_1( arg5 );
  _RT_PRINTF_ARG_1( arg6 );
  _RT_PRINTF_ARG_1( arg7 );
  _RT_PRINTF_ARG_1( arg8 );
  _RT_PRINTF_ARG_1( arg9 );
  _RT_PRINTF_2();
  _RT_PRINTF_ARG_2( arg1 );
  _RT_PRINTF_ARG_2( arg2 );
  _RT_PRINTF_ARG_2( arg3 );
  _RT_PRINTF_ARG_2( arg4 );
  _RT_PRINTF_ARG_2( arg5 );
  _RT_PRINTF_ARG_2( arg6 );
  _RT_PRINTF_ARG_2( arg7 );
  _RT_PRINTF_ARG_2( arg8 );
  _RT_PRINTF_ARG_2( arg9 );
}

template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10>
static inline __device__ void rtPrintf( const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10 )
{
  _RT_PRINTF_1();
  _RT_PRINTF_ARG_1( arg1 );
  _RT_PRINTF_ARG_1( arg2 );
  _RT_PRINTF_ARG_1( arg3 );
  _RT_PRINTF_ARG_1( arg4 );
  _RT_PRINTF_ARG_1( arg5 );
  _RT_PRINTF_ARG_1( arg6 );
  _RT_PRINTF_ARG_1( arg7 );
  _RT_PRINTF_ARG_1( arg8 );
  _RT_PRINTF_ARG_1( arg9 );
  _RT_PRINTF_ARG_1( arg10 );
  _RT_PRINTF_2();
  _RT_PRINTF_ARG_2( arg1 );
  _RT_PRINTF_ARG_2( arg2 );
  _RT_PRINTF_ARG_2( arg3 );
  _RT_PRINTF_ARG_2( arg4 );
  _RT_PRINTF_ARG_2( arg5 );
  _RT_PRINTF_ARG_2( arg6 );
  _RT_PRINTF_ARG_2( arg7 );
  _RT_PRINTF_ARG_2( arg8 );
  _RT_PRINTF_ARG_2( arg9 );
  _RT_PRINTF_ARG_2( arg10 );
}

template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11>
static inline __device__ void rtPrintf( const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11 )
{
  _RT_PRINTF_1();
  _RT_PRINTF_ARG_1( arg1 );
  _RT_PRINTF_ARG_1( arg2 );
  _RT_PRINTF_ARG_1( arg3 );
  _RT_PRINTF_ARG_1( arg4 );
  _RT_PRINTF_ARG_1( arg5 );
  _RT_PRINTF_ARG_1( arg6 );
  _RT_PRINTF_ARG_1( arg7 );
  _RT_PRINTF_ARG_1( arg8 );
  _RT_PRINTF_ARG_1( arg9 );
  _RT_PRINTF_ARG_1( arg10 );
  _RT_PRINTF_ARG_1( arg11 );
  _RT_PRINTF_2();
  _RT_PRINTF_ARG_2( arg1 );
  _RT_PRINTF_ARG_2( arg2 );
  _RT_PRINTF_ARG_2( arg3 );
  _RT_PRINTF_ARG_2( arg4 );
  _RT_PRINTF_ARG_2( arg5 );
  _RT_PRINTF_ARG_2( arg6 );
  _RT_PRINTF_ARG_2( arg7 );
  _RT_PRINTF_ARG_2( arg8 );
  _RT_PRINTF_ARG_2( arg9 );
  _RT_PRINTF_ARG_2( arg10 );
  _RT_PRINTF_ARG_2( arg11 );
}

template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12>
static inline __device__ void rtPrintf( const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11, T12 arg12 )
{
  _RT_PRINTF_1();
  _RT_PRINTF_ARG_1( arg1 );
  _RT_PRINTF_ARG_1( arg2 );
  _RT_PRINTF_ARG_1( arg3 );
  _RT_PRINTF_ARG_1( arg4 );
  _RT_PRINTF_ARG_1( arg5 );
  _RT_PRINTF_ARG_1( arg6 );
  _RT_PRINTF_ARG_1( arg7 );
  _RT_PRINTF_ARG_1( arg8 );
  _RT_PRINTF_ARG_1( arg9 );
  _RT_PRINTF_ARG_1( arg10 );
  _RT_PRINTF_ARG_1( arg11 );
  _RT_PRINTF_ARG_1( arg12 );
  _RT_PRINTF_2();
  _RT_PRINTF_ARG_2( arg1 );
  _RT_PRINTF_ARG_2( arg2 );
  _RT_PRINTF_ARG_2( arg3 );
  _RT_PRINTF_ARG_2( arg4 );
  _RT_PRINTF_ARG_2( arg5 );
  _RT_PRINTF_ARG_2( arg6 );
  _RT_PRINTF_ARG_2( arg7 );
  _RT_PRINTF_ARG_2( arg8 );
  _RT_PRINTF_ARG_2( arg9 );
  _RT_PRINTF_ARG_2( arg10 );
  _RT_PRINTF_ARG_2( arg11 );
  _RT_PRINTF_ARG_2( arg12 );
}

#undef _RT_PRINTF_1
#undef _RT_PRINTF_2
#undef _RT_PRINTF_ARG_1
#undef _RT_PRINTF_ARG_2


namespace rti_internal_register {
  extern __device__ void* reg_bitness_detector;
  extern __device__ volatile unsigned long long reg_exception_64_detail0;
  extern __device__ volatile unsigned long long reg_exception_64_detail1;
  extern __device__ volatile unsigned long long reg_exception_64_detail2;
  extern __device__ volatile unsigned long long reg_exception_64_detail3;
  extern __device__ volatile unsigned long long reg_exception_64_detail4;
  extern __device__ volatile unsigned long long reg_exception_64_detail5;
  extern __device__ volatile unsigned long long reg_exception_64_detail6;
  extern __device__ volatile unsigned long long reg_exception_64_detail7;
  extern __device__ volatile unsigned long long reg_exception_64_detail8;
  extern __device__ volatile unsigned long long reg_exception_64_detail9;
  extern __device__ volatile unsigned int reg_exception_detail0;
  extern __device__ volatile unsigned int reg_exception_detail1;
  extern __device__ volatile unsigned int reg_exception_detail2;
  extern __device__ volatile unsigned int reg_exception_detail3;
  extern __device__ volatile unsigned int reg_exception_detail4;
  extern __device__ volatile unsigned int reg_exception_detail5;
  extern __device__ volatile unsigned int reg_exception_detail6;
  extern __device__ volatile unsigned int reg_exception_detail7;
  extern __device__ volatile unsigned int reg_exception_detail8;
  extern __device__ volatile unsigned int reg_exception_detail9;
  extern __device__ volatile unsigned int reg_rayIndex_x;
  extern __device__ volatile unsigned int reg_rayIndex_y;
  extern __device__ volatile unsigned int reg_rayIndex_z;
}

static inline __device__ void rtThrow( unsigned int code )
{
  optix::rt_throw( code );
}

static inline __device__ unsigned int rtGetExceptionCode()
{
  return optix::rt_get_exception_code();
}

static inline __device__ void rtPrintExceptionDetails()
{
  const unsigned int code = rtGetExceptionCode();

  if( code == RT_EXCEPTION_STACK_OVERFLOW )
  {
    rtPrintf( "Caught RT_EXCEPTION_STACK_OVERFLOW\n"
              "  launch index : %d, %d, %d\n",
              rti_internal_register::reg_rayIndex_x,
              rti_internal_register::reg_rayIndex_y,
              rti_internal_register::reg_rayIndex_z
              );
  }
  else if( code == RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS )
  {
#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
    const unsigned int dim = rti_internal_register::reg_exception_detail0;

    rtPrintf( "Caught RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS\n"
              "  launch index   : %d, %d, %d\n"
              "  buffer address : 0x%llX\n"
              "  dimensionality : %d\n"
              "  size           : %lldx%lldx%lld\n"
              "  element size   : %d\n"
              "  accessed index : %lld, %lld, %lld\n",
              rti_internal_register::reg_rayIndex_x,
              rti_internal_register::reg_rayIndex_y,
              rti_internal_register::reg_rayIndex_z,
              rti_internal_register::reg_exception_64_detail0,
              rti_internal_register::reg_exception_detail0,
              rti_internal_register::reg_exception_64_detail1,
              dim > 1 ? rti_internal_register::reg_exception_64_detail2 : 1,
              dim > 2 ? rti_internal_register::reg_exception_64_detail3 : 1,
              rti_internal_register::reg_exception_detail1,
              rti_internal_register::reg_exception_64_detail4,
              rti_internal_register::reg_exception_64_detail5,
              rti_internal_register::reg_exception_64_detail6
              );
#else
    const unsigned int dim = rti_internal_register::reg_exception_detail1;

    rtPrintf( "Caught RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS\n"
              "  launch index   : %d, %d, %d\n"
              "  buffer address : 0x%X\n"
              "  dimensionality : %d\n"
              "  size           : %dx%dx%d\n"
              "  element size   : %d\n"
              "  accessed index : %d, %d, %d\n",
              rti_internal_register::reg_rayIndex_x,
              rti_internal_register::reg_rayIndex_y,
              rti_internal_register::reg_rayIndex_z,
              rti_internal_register::reg_exception_detail0,
              rti_internal_register::reg_exception_detail1,
              rti_internal_register::reg_exception_detail2,
              dim > 1 ? rti_internal_register::reg_exception_detail3 : 1,
              dim > 2 ? rti_internal_register::reg_exception_detail4 : 1,
              rti_internal_register::reg_exception_detail5,
              rti_internal_register::reg_exception_detail6,
              rti_internal_register::reg_exception_detail7,
              rti_internal_register::reg_exception_detail8
              );
#endif
  }
  else if( code == RT_EXCEPTION_INDEX_OUT_OF_BOUNDS )
  {
#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
    const unsigned int dim = rti_internal_register::reg_exception_detail0;

    rtPrintf( "Caught RT_EXCEPTION_INDEX_OUT_OF_BOUNDS\n"
              "  launch index   : %d, %d, %d\n"
              "  buffer address : 0x%llX\n"
              "  size           : %lld\n"
              "  accessed index : %lld\n",
              rti_internal_register::reg_rayIndex_x,
              rti_internal_register::reg_rayIndex_y,
              rti_internal_register::reg_rayIndex_z,
              rti_internal_register::reg_exception_64_detail0,
              rti_internal_register::reg_exception_64_detail1,
              rti_internal_register::reg_exception_64_detail2
              );
#else
    const unsigned int dim = rti_internal_register::reg_exception_detail1;

    rtPrintf( "Caught RT_EXCEPTION_INDEX_OUT_OF_BOUNDS\n"
              "  launch index   : %d, %d, %d\n"
              "  buffer address : 0x%X\n"
              "  size           : %d\n"
              "  accessed index : %d\n",
              rti_internal_register::reg_rayIndex_x,
              rti_internal_register::reg_rayIndex_y,
              rti_internal_register::reg_rayIndex_z,
              rti_internal_register::reg_exception_detail0,
              rti_internal_register::reg_exception_detail1,
              rti_internal_register::reg_exception_detail2
              );
#endif
  }
  else if( code == RT_EXCEPTION_INVALID_RAY )
  {
    rtPrintf( "Caught RT_EXCEPTION_INVALID_RAY\n"
              "  launch index  : %d, %d, %d\n"
              "  ray origin    : %f %f %f\n"
              "  ray direction : %f %f %f\n"
              "  ray type      : %d\n"
              "  ray tmin      : %f\n"
              "  ray tmax      : %f\n",
              rti_internal_register::reg_rayIndex_x,
              rti_internal_register::reg_rayIndex_y,
              rti_internal_register::reg_rayIndex_z,
              __int_as_float(rti_internal_register::reg_exception_detail0),
              __int_as_float(rti_internal_register::reg_exception_detail1),
              __int_as_float(rti_internal_register::reg_exception_detail2),
              __int_as_float(rti_internal_register::reg_exception_detail3),
              __int_as_float(rti_internal_register::reg_exception_detail4),
              __int_as_float(rti_internal_register::reg_exception_detail5),
              rti_internal_register::reg_exception_detail6,
              __int_as_float(rti_internal_register::reg_exception_detail7),
              __int_as_float(rti_internal_register::reg_exception_detail8)
              );
  }
  else if( code == RT_EXCEPTION_INTERNAL_ERROR )
  {
    // Should never happen.
    rtPrintf( "Caught RT_EXCEPTION_INTERNAL_ERROR\n"
              "  launch index : %d, %d, %d\n"
              "  error id     : %d\n",
              rti_internal_register::reg_rayIndex_x,
              rti_internal_register::reg_rayIndex_y,
              rti_internal_register::reg_rayIndex_z,
              rti_internal_register::reg_exception_detail0
              );
  }
  else if( code >= RT_EXCEPTION_USER && code <= 0xFFFF )
  {
    rtPrintf( "Caught RT_EXCEPTION_USER+%d\n"
              "  launch index : %d, %d, %d\n",
              code-RT_EXCEPTION_USER,
              rti_internal_register::reg_rayIndex_x,
              rti_internal_register::reg_rayIndex_y,
              rti_internal_register::reg_rayIndex_z
              );
  }
  else
  {
    // Should never happen.
    rtPrintf( "Caught unknown exception\n"
              "  launch index : %d, %d, %d\n",
              rti_internal_register::reg_rayIndex_x,
              rti_internal_register::reg_rayIndex_y,
              rti_internal_register::reg_rayIndex_z
              );
  }
}

#endif /* __optix_optix_cuda__internal_h__ */

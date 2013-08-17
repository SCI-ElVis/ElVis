
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

#ifndef __optix_optix_defines_h__
#define __optix_optix_defines_h__

enum rtSemanticTypes {
  /* Type uint3  */
  _OPTIX_SEMANTIC_TYPE_LaunchIndex          = 0x100,

  /* Type Ray    */
  _OPTIX_SEMANTIC_TYPE_CurrentRay           = 0x200,

  /* Type float  */
  _OPTIX_SEMANTIC_TYPE_IntersectionDistance = 0x300
};

enum RTtransformkind {
  RT_WORLD_TO_OBJECT = 0xf00,
  RT_OBJECT_TO_WORLD
};
enum RTtransformflags {
  RT_INTERNAL_INVERSE_TRANSPOSE = 0x1000
};

namespace rti_internal_typeinfo {
  enum rtiTypeKind {
    _OPTIX_VARIABLE = 0x796152
  };
  struct rti_typeinfo {
    unsigned int kind;
    unsigned int size;
  };
}

#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
#define OPTIX_ASM_PTR          "l"
#define OPTIX_ASM_SIZE_T       "l"
#define OPTIX_ASM_PTR_SIZE_STR "64"
#define OPTIX_BITNESS_SUFFIX   "_64"
namespace optix {
#if defined( _WIN64 )
  typedef unsigned __int64   optix_size_t;
#else
  typedef unsigned long long optix_size_t;
#endif
}
#else
#define OPTIX_ASM_PTR          "r"
#define OPTIX_ASM_SIZE_T       "r"
#define OPTIX_ASM_PTR_SIZE_STR "32"
#define OPTIX_BITNESS_SUFFIX   ""
namespace optix {
  typedef size_t optix_size_t;
}
#endif

#endif /* __optix_optix_defines_h__ */

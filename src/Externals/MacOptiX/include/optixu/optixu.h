
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

#ifndef __optix_optixu_h__
#define __optix_optixu_h__

#include <stddef.h>
#include "../optix.h"

#ifdef __cplusplus
#  define RTU_INLINE inline
#else
#  ifdef _MSC_VER
#    define RTU_INLINE __inline
#  else
#    define RTU_INLINE inline
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

 /*
  * Get the name string of a given type.
  */
  RTresult RTAPI rtuNameForType( RTobjecttype type, char* buffer, RTsize bufferSize );

 /*
  * Return the size of a given RTformat.  RT_FORMAT_USER and RT_FORMAT_UNKNOWN return 0.
  * Returns RT_ERROR_INVALID_VALUE if the format isn't recognized, RT_SUCCESS otherwise.
  */
  RTresult RTAPI rtuGetSizeForRTformat( RTformat format, size_t* size);

 /*
  * Compile a cuda source string.
  * ARGS:
  *
  * source                       source code string
  * preprocessorArguments        list of preprocessor arguments
  * numPreprocessorArguments     number of preprocessor arguments
  * resultSize                   [out] size required to hold compiled result string
  * errorSize                    [out] size required to hold error string
  */
  RTresult RTAPI rtuCUDACompileString( const char* source, const char** preprocessorArguments, unsigned int numPreprocessorArguments, RTsize* resultSize, RTsize* errorSize );

 /*
  * Compile a cuda source file.
  * ARGS:
  *
  * filename                     source code file name
  * preprocessorArguments        list of preprocessor arguments
  * numPreprocessorArguments     number of preprocessor arguments
  * resultSize                   [out] size required to hold compiled result string
  * errorSize                    [out] size required to hold error string
  */
  RTresult RTAPI rtuCUDACompileFile( const char* filename, const char** preprocessorArguments, unsigned int numPreprocessorArguments, RTsize* resultSize, RTsize* errorSize );

 /*
  * Get the result of the most recent call to one of the above compile functions.
  * The 'result' and 'error' parameters must point to memory large enough to hold
  * the respective strings, as returned by the compile function.
  * ARGS:
  *
  * result                      compiled result string
  * error                       error string
  */
  RTresult RTAPI rtuCUDAGetCompileResult( char* result, char* error );

#ifdef __cplusplus
} /* extern "C" */
#endif

 /*
  * Add an entry to the end of the child array.
  * Fills 'index' with the index of the added child, if the pointer is non-NULL.
  */
#ifndef __cplusplus
 RTresult rtuGroupAddChild        ( RTgroup group, RTobject child, unsigned int* index );
 RTresult rtuSelectorAddChild     ( RTselector selector, RTobject child, unsigned int* index );
#else
 RTresult rtuGroupAddChild        ( RTgroup group, RTgroup         child, unsigned int* index );
 RTresult rtuGroupAddChild        ( RTgroup group, RTselector      child, unsigned int* index );
 RTresult rtuGroupAddChild        ( RTgroup group, RTtransform     child, unsigned int* index );
 RTresult rtuGroupAddChild        ( RTgroup group, RTgeometrygroup child, unsigned int* index );
 RTresult rtuSelectorAddChild     ( RTselector selector, RTgroup         child, unsigned int* index );
 RTresult rtuSelectorAddChild     ( RTselector selector, RTselector      child, unsigned int* index );
 RTresult rtuSelectorAddChild     ( RTselector selector, RTtransform     child, unsigned int* index );
 RTresult rtuSelectorAddChild     ( RTselector selector, RTgeometrygroup child, unsigned int* index );
#endif
 RTresult rtuGeometryGroupAddChild( RTgeometrygroup geometrygroup, RTgeometryinstance child, unsigned int* index );

 /*
  * Wrap rtTransformSetChild in order to provide a type-safe version for C++.
  */
#ifndef __cplusplus
 RTresult rtuTransformSetChild    ( RTtransform transform, RTobject        child );
#else
 RTresult rtuTransformSetChild    ( RTtransform transform, RTgroup         child );
 RTresult rtuTransformSetChild    ( RTtransform transform, RTselector      child );
 RTresult rtuTransformSetChild    ( RTtransform transform, RTtransform     child );
 RTresult rtuTransformSetChild    ( RTtransform transform, RTgeometrygroup child );
#endif

 /*
  * Find the given child using a linear search in the child array and remove
  * it. If it's not the last entry in the child array, the last entry in the
  * array will replace the deleted entry, in order to shrink the array size by one.
  */
 RTresult rtuGroupRemoveChild        ( RTgroup group, RTobject child );
 RTresult rtuSelectorRemoveChild     ( RTselector selector, RTobject child );
 RTresult rtuGeometryGroupRemoveChild( RTgeometrygroup geometrygroup, RTgeometryinstance child );

 /*
  * Remove the child at the given index in the child array. If it's not the last
  * entry in the child array, the last entry in the array will replace the deleted
  * entry, in order to shrink the array size by one.
  */
 RTU_INLINE RTresult rtuGroupRemoveChildByIndex        ( RTgroup group, unsigned int index );
 RTU_INLINE RTresult rtuSelectorRemoveChildByIndex     ( RTselector selector, unsigned int index );
 RTU_INLINE RTresult rtuGeometryGroupRemoveChildByIndex( RTgeometrygroup geometrygroup, unsigned int index );

 /*
  * Use a linear search to find the child in the child array, and return its index.
  * Returns RT_SUCCESS if the child was found, RT_INVALID_VALUE otherwise.
  */
 RTU_INLINE RTresult rtuGroupGetChildIndex        ( RTgroup group, RTobject child, unsigned int* index );
 RTU_INLINE RTresult rtuSelectorGetChildIndex     ( RTselector selector, RTobject child, unsigned int* index );
 RTU_INLINE RTresult rtuGeometryGroupGetChildIndex( RTgeometrygroup geometrygroup, RTgeometryinstance child, unsigned int* index );


 /*
  * The following implements the child management helpers declared above.
  */

#define RTU_CHECK_ERROR( func )                 \
  do {                                          \
    RTresult code = func;                       \
    if( code != RT_SUCCESS )                    \
      return code;                              \
  } while(0)

#define RTU_GROUP_ADD_CHILD( _parent, _child, _index )                  \
  unsigned int _count;                                                  \
  RTU_CHECK_ERROR( rtGroupGetChildCount( (_parent), &_count ) );        \
  RTU_CHECK_ERROR( rtGroupSetChildCount( (_parent), _count+1 ) );       \
  RTU_CHECK_ERROR( rtGroupSetChild( (_parent), _count, (_child) ) );    \
  if( _index ) *(_index) = _count;                                      \
  return RT_SUCCESS

#define RTU_SELECTOR_ADD_CHILD( _parent, _child, _index )               \
  unsigned int _count;                                                  \
  RTU_CHECK_ERROR( rtSelectorGetChildCount( (_parent), &_count ) );     \
  RTU_CHECK_ERROR( rtSelectorSetChildCount( (_parent), _count+1 ) );    \
  RTU_CHECK_ERROR( rtSelectorSetChild( (_parent), _count, (_child) ) ); \
  if( _index ) *(_index) = _count;                                      \
  return RT_SUCCESS


#ifndef __cplusplus

 RTU_INLINE RTresult rtuGroupAddChild( RTgroup group, RTobject child, unsigned int* index )
 {
   RTU_GROUP_ADD_CHILD( group, child, index );
 }

 RTU_INLINE RTresult rtuSelectorAddChild( RTselector selector, RTobject child, unsigned int* index )
 {
   RTU_SELECTOR_ADD_CHILD( selector, child, index );
 }

#else /* __cplusplus */

 RTU_INLINE RTresult rtuGroupAddChild( RTgroup group, RTgroup child, unsigned int* index )
 {
   RTU_GROUP_ADD_CHILD( group, child, index );
 }

 RTU_INLINE RTresult rtuGroupAddChild( RTgroup group, RTselector child, unsigned int* index )
 {
   RTU_GROUP_ADD_CHILD( group, child, index );
 }

 RTU_INLINE RTresult rtuGroupAddChild( RTgroup group, RTtransform child, unsigned int* index )
 {
   RTU_GROUP_ADD_CHILD( group, child, index );
 }

 RTU_INLINE RTresult rtuGroupAddChild( RTgroup group, RTgeometrygroup child, unsigned int* index )
 {
   RTU_GROUP_ADD_CHILD( group, child, index );
 }

 RTU_INLINE RTresult rtuSelectorAddChild( RTselector selector, RTgroup child, unsigned int* index )
 {
   RTU_SELECTOR_ADD_CHILD( selector, child, index );
 }

 RTU_INLINE RTresult rtuSelectorAddChild( RTselector selector, RTselector child, unsigned int* index )
 {
   RTU_SELECTOR_ADD_CHILD( selector, child, index );
 }

 RTU_INLINE RTresult rtuSelectorAddChild( RTselector selector, RTtransform child, unsigned int* index )
 {
   RTU_SELECTOR_ADD_CHILD( selector, child, index );
 }

 RTU_INLINE RTresult rtuSelectorAddChild( RTselector selector, RTgeometrygroup child, unsigned int* index )
 {
   RTU_SELECTOR_ADD_CHILD( selector, child, index );
 }

#endif /* __cplusplus */

#undef RTU_GROUP_ADD_CHILD
#undef RTU_SELECTOR_ADD_CHILD

#ifndef __cplusplus

 RTU_INLINE RTresult rtuTransformSetChild( RTtransform transform, RTobject child )
 {
   RTU_CHECK_ERROR( rtTransformSetChild( transform, child ) );
   return RT_SUCCESS;
 }

#else /* __cplusplus */

 RTU_INLINE RTresult rtuTransformSetChild( RTtransform transform, RTgroup child )
 {
   RTU_CHECK_ERROR( rtTransformSetChild( transform, child ) );
   return RT_SUCCESS;
 }

 RTU_INLINE RTresult rtuTransformSetChild( RTtransform transform, RTselector child )
 {
   RTU_CHECK_ERROR( rtTransformSetChild( transform, child ) );
   return RT_SUCCESS;
 }

 RTU_INLINE RTresult rtuTransformSetChild( RTtransform transform, RTtransform child )
 {
   RTU_CHECK_ERROR( rtTransformSetChild( transform, child ) );
   return RT_SUCCESS;
 }

 RTU_INLINE RTresult rtuTransformSetChild( RTtransform transform, RTgeometrygroup child )
 {
   RTU_CHECK_ERROR( rtTransformSetChild( transform, child ) );
   return RT_SUCCESS;
 }

#endif /* __cplusplus */

 RTU_INLINE RTresult rtuGeometryGroupAddChild( RTgeometrygroup geometrygroup, RTgeometryinstance child, unsigned int* index )
 {
   unsigned int count;
   RTU_CHECK_ERROR( rtGeometryGroupGetChildCount( geometrygroup, &count ) );
   RTU_CHECK_ERROR( rtGeometryGroupSetChildCount( geometrygroup, count+1 ) );
   RTU_CHECK_ERROR( rtGeometryGroupSetChild( geometrygroup, count, child ) );
   if( index ) *index = count;
   return RT_SUCCESS;
 }

 RTU_INLINE RTresult rtuGroupRemoveChild( RTgroup group, RTobject child )
 {
   unsigned int index;
   RTU_CHECK_ERROR( rtuGroupGetChildIndex( group, child, &index ) );
   RTU_CHECK_ERROR( rtuGroupRemoveChildByIndex( group, index ) );
   return RT_SUCCESS;
 }

 RTU_INLINE RTresult rtuSelectorRemoveChild( RTselector selector, RTobject child )
 {
   unsigned int index;
   RTU_CHECK_ERROR( rtuSelectorGetChildIndex( selector, child, &index ) );
   RTU_CHECK_ERROR( rtuSelectorRemoveChildByIndex( selector, index ) );
   return RT_SUCCESS;
 }

 RTU_INLINE RTresult rtuGeometryGroupRemoveChild( RTgeometrygroup geometrygroup, RTgeometryinstance child )
 {
   unsigned int index;
   RTU_CHECK_ERROR( rtuGeometryGroupGetChildIndex( geometrygroup, child, &index ) );
   RTU_CHECK_ERROR( rtuGeometryGroupRemoveChildByIndex( geometrygroup, index ) );
   return RT_SUCCESS;
 }

 RTU_INLINE RTresult rtuGroupRemoveChildByIndex( RTgroup group, unsigned int index )
 {
   unsigned int count;
   RTobject temp;
   RTU_CHECK_ERROR( rtGroupGetChildCount( group, &count ) );
   RTU_CHECK_ERROR( rtGroupGetChild( group, count-1, &temp ) );
   RTU_CHECK_ERROR( rtGroupSetChild( group, index, temp ) );
   RTU_CHECK_ERROR( rtGroupSetChildCount( group, count-1 ) );
   return RT_SUCCESS;
 }

 RTU_INLINE RTresult rtuSelectorRemoveChildByIndex( RTselector selector, unsigned int index )
 {
   unsigned int count;
   RTobject temp;
   RTU_CHECK_ERROR( rtSelectorGetChildCount( selector, &count ) );
   RTU_CHECK_ERROR( rtSelectorGetChild( selector, count-1, &temp ) );
   RTU_CHECK_ERROR( rtSelectorSetChild( selector, index, temp ) );
   RTU_CHECK_ERROR( rtSelectorSetChildCount( selector, count-1 ) );
   return RT_SUCCESS;
 }

 RTU_INLINE RTresult rtuGeometryGroupRemoveChildByIndex( RTgeometrygroup geometrygroup, unsigned int index )
 {
   unsigned int count;
   RTgeometryinstance temp;
   RTU_CHECK_ERROR( rtGeometryGroupGetChildCount( geometrygroup, &count ) );
   RTU_CHECK_ERROR( rtGeometryGroupGetChild( geometrygroup, count-1, &temp ) );
   RTU_CHECK_ERROR( rtGeometryGroupSetChild( geometrygroup, index, temp ) );
   RTU_CHECK_ERROR( rtGeometryGroupSetChildCount( geometrygroup, count-1 ) );
   return RT_SUCCESS;
 }

 RTU_INLINE RTresult rtuGroupGetChildIndex(RTgroup group, RTobject child, unsigned int* index)
 {
   unsigned int count;
   RTobject temp;
   RTU_CHECK_ERROR( rtGroupGetChildCount( group, &count ) );
   for( *index=0; *index<count; (*index)++ ) {
     RTU_CHECK_ERROR( rtGroupGetChild( group, *index, &temp ) );
     if( child==temp ) return RT_SUCCESS;
   }
   *index = ~0u;
   return RT_ERROR_INVALID_VALUE;
 }

 RTU_INLINE RTresult rtuSelectorGetChildIndex( RTselector selector, RTobject child, unsigned int* index )
 {
   unsigned int count;
   RTobject temp;
   RTU_CHECK_ERROR( rtSelectorGetChildCount( selector, &count ) );
   for( *index=0; *index<count; (*index)++ ) {
     RTU_CHECK_ERROR( rtSelectorGetChild( selector, *index, &temp ) );
     if( child==temp ) return RT_SUCCESS;
   }
   *index = ~0u;
   return RT_ERROR_INVALID_VALUE;
 }

 RTU_INLINE RTresult rtuGeometryGroupGetChildIndex( RTgeometrygroup geometrygroup, RTgeometryinstance child, unsigned int* index )
 {
   unsigned int count;
   RTgeometryinstance temp;
   RTU_CHECK_ERROR( rtGeometryGroupGetChildCount( geometrygroup, &count ) );
   for( *index=0; *index<count; (*index)++ ) {
     RTU_CHECK_ERROR( rtGeometryGroupGetChild( geometrygroup, *index, &temp ) );
     if( child==temp ) return RT_SUCCESS;
   }
   *index = ~0u;
   return RT_ERROR_INVALID_VALUE;
 }


#ifdef __cplusplus
extern "C" {
#endif

  /**
   * Create clustered triangle mesh for good memory coherence with paging on.
   * Vertex, index and material buffers are created and attached to the mesh.
   * Cluster's bounding box and intersection programs are attached to the mesh.
   * The intersection program has the following attributes: 
   *   rtDeclareVariable(float3, texcoord, attribute texcoord, ); It is always zero
   *   rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
   *   rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); It is equal to geometric_normal
   *
   * Created RTgeometry mesh expects there to be placed into a RTgeometryinstance where
   * the mat_indices specified map into materials attached to the RTgeometryinstance
   *
   * In the event of an error, please query the error string from the RTcontext.
   * 
   *   \param    context        Context
   *   \param    usePTX32InHost64 Use 32bit PTX bounding box and intersection programs in 64bit application. Takes effect only with 64bit host.
   *   \param    mesh           Output geometry
   *   \param    num_verts      Vertex count
   *   \param    verts          Vertices (num_verts*float*3) [ v1_x, v1_y, v1_z, v2.x, ... ]
   *   \param    num_tris       Triangle count
   *   \param    indices        Vertex indices (num_tris*unsigned*3) [ tri1_index1, tr1_index2, ... ]
   *   \param    mat_indices    Indices of materials (num_tris*unsigned) [ tri1_mat_index, tri2_mat_index, ... ]
   */
  RTresult RTAPI rtuCreateClusteredMesh( RTcontext       context,
                                         unsigned int    usePTX32InHost64,
                                         RTgeometry*     mesh,
                                         unsigned int    num_verts,
                                         const float*    verts,
                                         unsigned int    num_tris,
                                         const unsigned* indices,
                                         const unsigned* mat_indices);



  /**
   * Create clustered triangle mesh for good memory coherence with paging on.
   * Buffers for vertices, indices, normals, indices of normals,
   * texture coordinates, indices of texture coordinates and materials are created and attached to the mesh.
   * Cluster's bounding box and intersection programs are attached to the mesh.
   * The intersection program has the following attributes: 
   *   rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
   *   rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
   *   rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
   *
   * Created RTgeometry mesh expects there to be placed into a RTgeometryinstance where
   * the mat_indices specified map into materials attached to the RTgeometryinstance
   *
   * Vertex, normal and texture coordinate buffers can be shared between many geometry objects
   * 
   * In the event of an error, please query the error string from the RTcontext.
   *
   *   \param    context        Context
   *   \param    usePTX32InHost64 Use 32bit PTX bounding box and intersection programs in 64bit application. Takes effect only with 64bit host.
   *   \param    mesh           Output geometry
   *   \param    num_verts      Vertex count
   *   \param    verts          Vertices (num_verts*float*3) [ v1_x, v1_y, v1_z, v2.x, ... ]
   *   \param    num_tris       Triangle count
   *   \param    indices        Vertex indices (num_tris*unsigned*3) [ tri1_index1, tr1_index2, ... ]
   *   \param    mat_indices    Indices of materials (num_tris*unsigned) [ tri1_mat_index, tri2_mat_index, ... ]
   *   \param    norms          Normals (num_norms*float*3) [ v1_x, v1_y, v1_z, v2.x, ... ]
   *   \param    norm_indices   Indices of vertex normals (num_tris*unsigned*3) [ tri1_norm_index1, tri1_norm_index2 ... ]
   *   \param    tex_coords     Texture uv coords (num_tex_coords*float*2) [ t1_u, t1_v, t2_u ... ]
   *   \param    tex_indices    Indices of texture uv (num_tris*unsigned*3) [ tri1_tex_index1, tri1_tex_index2 ... ]
   */
  RTresult RTAPI rtuCreateClusteredMeshExt( RTcontext       context,
                                            unsigned int    usePTX32InHost64,
                                            RTgeometry*     mesh,
                                            unsigned int    num_verts,
                                            const float*    verts,
                                            unsigned int    num_tris,
                                            const unsigned* indices,
                                            const unsigned* mat_indices,
                                            RTbuffer        norms,
                                            const unsigned* norm_indices,
                                            RTbuffer        tex_coords,
                                            const unsigned* tex_indices );

#ifdef __cplusplus
} /* extern "C" */
#endif


#undef RTU_CHECK_ERROR
#undef RTU_INLINE

#endif /* __optix_optixu_h__ */

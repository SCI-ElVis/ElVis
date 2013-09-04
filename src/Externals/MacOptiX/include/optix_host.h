
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

#ifndef __optix_optix_host_h__
#define __optix_optix_host_h__

#ifndef RTAPI
#if defined( _WIN32 )
#define RTAPI __declspec(dllimport)
#else
#define RTAPI
#endif
#endif

#include "internal/optix_declarations.h"


/************************************
 **
 **    Platform-Dependent Types
 **
 ***********************************/

#if defined( _WIN64 )
typedef unsigned __int64    RTsize;
#elif defined( _WIN32 )
typedef unsigned int        RTsize;
#else
typedef long unsigned int   RTsize;
#endif

/************************************
 **
 **    Opaque Object Types
 **
 ***********************************/

/* Opaque types.  Note that the *_api types should never be used directly.  Only the
 * typedef target names will be guaranteed to remain unchanged. */
typedef struct RTacceleration_api       * RTacceleration;
typedef struct RTbuffer_api             * RTbuffer;
typedef struct RTcontext_api            * RTcontext;
typedef struct RTgeometry_api           * RTgeometry;
typedef struct RTgeometryinstance_api   * RTgeometryinstance;
typedef struct RTgeometrygroup_api      * RTgeometrygroup;
typedef struct RTgroup_api              * RTgroup;
typedef struct RTmaterial_api           * RTmaterial;
typedef struct RTprogram_api            * RTprogram;
typedef struct RTselector_api           * RTselector;
typedef struct RTtexturesampler_api     * RTtexturesampler;
typedef struct RTtransform_api          * RTtransform;
typedef struct RTvariable_api           * RTvariable;
typedef void                            * RTobject;

/************************************
 **
 **    Callback Function Types
 **
 ***********************************/

/* Callback signature for use with rtContextSetTimeoutCallback.
 * Return 1 to ask for abort, 0 to continue. */
typedef int (*RTtimeoutcallback)(void);


#ifdef __cplusplus
extern "C" {
#endif

/************************************
 **
 **    Context-free functions
 **
 ***********************************/

  RTresult RTAPI rtGetVersion(unsigned int* version);
  RTresult RTAPI rtDeviceGetDeviceCount(unsigned int* count);
  RTresult RTAPI rtDeviceGetAttribute(int ordinal, RTdeviceattribute attrib, RTsize size, void* p);

/************************************
 **
 **    Object Variable Accessors
 **
 ***********************************/

  /* Sets */
  RTresult RTAPI rtVariableSet1f (RTvariable v, float f1);
  RTresult RTAPI rtVariableSet2f (RTvariable v, float f1, float f2);
  RTresult RTAPI rtVariableSet3f (RTvariable v, float f1, float f2, float f3);
  RTresult RTAPI rtVariableSet4f (RTvariable v, float f1, float f2, float f3, float f4);
  RTresult RTAPI rtVariableSet1fv(RTvariable v, const float* f);
  RTresult RTAPI rtVariableSet2fv(RTvariable v, const float* f);
  RTresult RTAPI rtVariableSet3fv(RTvariable v, const float* f);
  RTresult RTAPI rtVariableSet4fv(RTvariable v, const float* f);

  RTresult RTAPI rtVariableSet1i (RTvariable v, int i1);
  RTresult RTAPI rtVariableSet2i (RTvariable v, int i1, int i2);
  RTresult RTAPI rtVariableSet3i (RTvariable v, int i1, int i2, int i3);
  RTresult RTAPI rtVariableSet4i (RTvariable v, int i1, int i2, int i3, int i4);
  RTresult RTAPI rtVariableSet1iv(RTvariable v, const int* i);
  RTresult RTAPI rtVariableSet2iv(RTvariable v, const int* i);
  RTresult RTAPI rtVariableSet3iv(RTvariable v, const int* i);
  RTresult RTAPI rtVariableSet4iv(RTvariable v, const int* i);

  RTresult RTAPI rtVariableSet1ui (RTvariable v, unsigned int u1);
  RTresult RTAPI rtVariableSet2ui (RTvariable v, unsigned int u1, unsigned int u2);
  RTresult RTAPI rtVariableSet3ui (RTvariable v, unsigned int u1, unsigned int u2, unsigned int u3);
  RTresult RTAPI rtVariableSet4ui (RTvariable v, unsigned int u1, unsigned int u2, unsigned int u3, unsigned int u4);
  RTresult RTAPI rtVariableSet1uiv(RTvariable v, const unsigned int* u);
  RTresult RTAPI rtVariableSet2uiv(RTvariable v, const unsigned int* u);
  RTresult RTAPI rtVariableSet3uiv(RTvariable v, const unsigned int* u);
  RTresult RTAPI rtVariableSet4uiv(RTvariable v, const unsigned int* u);

  RTresult RTAPI rtVariableSetMatrix2x2fv(RTvariable v, int transpose, const float* m);
  RTresult RTAPI rtVariableSetMatrix2x3fv(RTvariable v, int transpose, const float* m);
  RTresult RTAPI rtVariableSetMatrix2x4fv(RTvariable v, int transpose, const float* m);
  RTresult RTAPI rtVariableSetMatrix3x2fv(RTvariable v, int transpose, const float* m);
  RTresult RTAPI rtVariableSetMatrix3x3fv(RTvariable v, int transpose, const float* m);
  RTresult RTAPI rtVariableSetMatrix3x4fv(RTvariable v, int transpose, const float* m);
  RTresult RTAPI rtVariableSetMatrix4x2fv(RTvariable v, int transpose, const float* m);
  RTresult RTAPI rtVariableSetMatrix4x3fv(RTvariable v, int transpose, const float* m);
  RTresult RTAPI rtVariableSetMatrix4x4fv(RTvariable v, int transpose, const float* m);

  RTresult RTAPI rtVariableSetObject  (RTvariable v, RTobject object);
  RTresult RTAPI rtVariableSetUserData(RTvariable v, RTsize size, const void* ptr);

  /* Gets */
  RTresult RTAPI rtVariableGet1f (RTvariable v, float* f1);
  RTresult RTAPI rtVariableGet2f (RTvariable v, float* f1, float* f2);
  RTresult RTAPI rtVariableGet3f (RTvariable v, float* f1, float* f2, float* f3);
  RTresult RTAPI rtVariableGet4f (RTvariable v, float* f1, float* f2, float* f3, float* f4);
  RTresult RTAPI rtVariableGet1fv(RTvariable v, float* f);
  RTresult RTAPI rtVariableGet2fv(RTvariable v, float* f);
  RTresult RTAPI rtVariableGet3fv(RTvariable v, float* f);
  RTresult RTAPI rtVariableGet4fv(RTvariable v, float* f);

  RTresult RTAPI rtVariableGet1i (RTvariable v, int* i1);
  RTresult RTAPI rtVariableGet2i (RTvariable v, int* i1, int* i2);
  RTresult RTAPI rtVariableGet3i (RTvariable v, int* i1, int* i2, int* i3);
  RTresult RTAPI rtVariableGet4i (RTvariable v, int* i1, int* i2, int* i3, int* i4);
  RTresult RTAPI rtVariableGet1iv(RTvariable v, int* i);
  RTresult RTAPI rtVariableGet2iv(RTvariable v, int* i);
  RTresult RTAPI rtVariableGet3iv(RTvariable v, int* i);
  RTresult RTAPI rtVariableGet4iv(RTvariable v, int* i);

  RTresult RTAPI rtVariableGet1ui (RTvariable v, unsigned int* u1);
  RTresult RTAPI rtVariableGet2ui (RTvariable v, unsigned int* u1, unsigned int* u2);
  RTresult RTAPI rtVariableGet3ui (RTvariable v, unsigned int* u1, unsigned int* u2, unsigned int* u3);
  RTresult RTAPI rtVariableGet4ui (RTvariable v, unsigned int* u1, unsigned int* u2, unsigned int* u3, unsigned int* u4);
  RTresult RTAPI rtVariableGet1uiv(RTvariable v, unsigned int* u);
  RTresult RTAPI rtVariableGet2uiv(RTvariable v, unsigned int* u);
  RTresult RTAPI rtVariableGet3uiv(RTvariable v, unsigned int* u);
  RTresult RTAPI rtVariableGet4uiv(RTvariable v, unsigned int* u);

  RTresult RTAPI rtVariableGetMatrix2x2fv(RTvariable v, int transpose, float* m);
  RTresult RTAPI rtVariableGetMatrix2x3fv(RTvariable v, int transpose, float* m);
  RTresult RTAPI rtVariableGetMatrix2x4fv(RTvariable v, int transpose, float* m);
  RTresult RTAPI rtVariableGetMatrix3x2fv(RTvariable v, int transpose, float* m);
  RTresult RTAPI rtVariableGetMatrix3x3fv(RTvariable v, int transpose, float* m);
  RTresult RTAPI rtVariableGetMatrix3x4fv(RTvariable v, int transpose, float* m);
  RTresult RTAPI rtVariableGetMatrix4x2fv(RTvariable v, int transpose, float* m);
  RTresult RTAPI rtVariableGetMatrix4x3fv(RTvariable v, int transpose, float* m);
  RTresult RTAPI rtVariableGetMatrix4x4fv(RTvariable v, int transpose, float* m);

  RTresult RTAPI rtVariableGetObject  (RTvariable v, RTobject* object);
  RTresult RTAPI rtVariableGetUserData(RTvariable v, RTsize size, void* ptr);

  /* Other */
  RTresult RTAPI rtVariableGetName(RTvariable v, const char** name_return);
  RTresult RTAPI rtVariableGetAnnotation(RTvariable v, const char** annotation_return);
  RTresult RTAPI rtVariableGetType(RTvariable v, RTobjecttype* type_return);
  RTresult RTAPI rtVariableGetContext(RTvariable v, RTcontext* context);
  RTresult RTAPI rtVariableGetSize(RTvariable v, RTsize* size);

/************************************
 **
 **    Context object
 **
 ***********************************/

  RTresult RTAPI rtContextCreate  (RTcontext* context);
  RTresult RTAPI rtContextDestroy (RTcontext  context);
  RTresult RTAPI rtContextValidate(RTcontext  context);

  void RTAPI rtContextGetErrorString(RTcontext context, RTresult code, const char** return_string);

  RTresult RTAPI rtContextSetAttribute(RTcontext context, RTcontextattribute attrib, RTsize size, void* p);
  RTresult RTAPI rtContextGetAttribute(RTcontext context, RTcontextattribute attrib, RTsize size, void* p);

  RTresult RTAPI rtContextSetDevices  (RTcontext context, unsigned int count, const int* devices);
  RTresult RTAPI rtContextGetDevices  (RTcontext context, int* devices);
  RTresult RTAPI rtContextGetDeviceCount (RTcontext context, unsigned int* count);

  RTresult RTAPI rtContextSetStackSize(RTcontext context, RTsize  stack_size_bytes);
  RTresult RTAPI rtContextGetStackSize(RTcontext context, RTsize* stack_size_bytes);

  RTresult RTAPI rtContextSetTimeoutCallback (RTcontext context, RTtimeoutcallback callback, double min_polling_seconds);

  RTresult RTAPI rtContextSetEntryPointCount(RTcontext context, unsigned int  num_entry_points);
  RTresult RTAPI rtContextGetEntryPointCount(RTcontext context, unsigned int* num_entry_points);

  RTresult RTAPI rtContextSetRayGenerationProgram(RTcontext context, unsigned int entry_point_index, RTprogram  program);
  RTresult RTAPI rtContextGetRayGenerationProgram(RTcontext context, unsigned int entry_point_index, RTprogram* program);

  RTresult RTAPI rtContextSetExceptionProgram(RTcontext context, unsigned int entry_point_index, RTprogram  program);
  RTresult RTAPI rtContextGetExceptionProgram(RTcontext context, unsigned int entry_point_index, RTprogram* program);

  RTresult RTAPI rtContextSetExceptionEnabled(RTcontext context, RTexception exception, int  enabled );
  RTresult RTAPI rtContextGetExceptionEnabled(RTcontext context, RTexception exception, int* enabled );

  RTresult RTAPI rtContextSetRayTypeCount(RTcontext context, unsigned int  num_ray_types);
  RTresult RTAPI rtContextGetRayTypeCount(RTcontext context, unsigned int* num_ray_types);

  RTresult RTAPI rtContextSetMissProgram(RTcontext context, unsigned int ray_type_index, RTprogram  program);
  RTresult RTAPI rtContextGetMissProgram(RTcontext context, unsigned int ray_type_index, RTprogram* program);

  RTresult RTAPI rtContextCompile (RTcontext context);

  RTresult RTAPI rtContextLaunch1D(RTcontext context, unsigned int entry_point_index, RTsize image_width);
  RTresult RTAPI rtContextLaunch2D(RTcontext context, unsigned int entry_point_index, RTsize image_width, RTsize image_height);
  RTresult RTAPI rtContextLaunch3D(RTcontext context, unsigned int entry_point_index, RTsize image_width, RTsize image_height, RTsize image_depth);

  RTresult RTAPI rtContextGetRunningState(RTcontext context, int* running);

  RTresult RTAPI rtContextSetPrintEnabled(RTcontext context, int  enabled);
  RTresult RTAPI rtContextGetPrintEnabled(RTcontext context, int* enabled);
  RTresult RTAPI rtContextSetPrintBufferSize(RTcontext context, RTsize  buffer_size_bytes);
  RTresult RTAPI rtContextGetPrintBufferSize(RTcontext context, RTsize* buffer_size_bytes);
  RTresult RTAPI rtContextSetPrintLaunchIndex(RTcontext context, int  x, int  y, int  z);
  RTresult RTAPI rtContextGetPrintLaunchIndex(RTcontext context, int* x, int* y, int* z);

  RTresult RTAPI rtContextDeclareVariable (RTcontext context, const char* name, RTvariable* v);
  RTresult RTAPI rtContextQueryVariable   (RTcontext context, const char* name, RTvariable* v);
  RTresult RTAPI rtContextRemoveVariable  (RTcontext context, RTvariable v);
  RTresult RTAPI rtContextGetVariableCount(RTcontext context, unsigned int* count);
  RTresult RTAPI rtContextGetVariable     (RTcontext context, unsigned int index, RTvariable* v);

/************************************
 **
 **    Program object
 **
 ***********************************/

  RTresult RTAPI rtProgramCreateFromPTXString(RTcontext context, const char* ptx, const char* program_name, RTprogram* program);
  RTresult RTAPI rtProgramCreateFromPTXFile  (RTcontext context, const char* filename, const char* program_name, RTprogram* program);
  RTresult RTAPI rtProgramDestroy            (RTprogram program);
  RTresult RTAPI rtProgramValidate           (RTprogram program);
  RTresult RTAPI rtProgramGetContext         (RTprogram program, RTcontext* context);

  RTresult RTAPI rtProgramDeclareVariable (RTprogram program, const char* name, RTvariable* v);
  RTresult RTAPI rtProgramQueryVariable   (RTprogram program, const char* name, RTvariable* v);
  RTresult RTAPI rtProgramRemoveVariable  (RTprogram program, RTvariable v);
  RTresult RTAPI rtProgramGetVariableCount(RTprogram program, unsigned int* count);
  RTresult RTAPI rtProgramGetVariable     (RTprogram program, unsigned int index, RTvariable* v);

/************************************
 **
 **    Group object
 **
 ***********************************/

  RTresult RTAPI rtGroupCreate    (RTcontext context, RTgroup* group);
  RTresult RTAPI rtGroupDestroy   (RTgroup group);
  RTresult RTAPI rtGroupValidate  (RTgroup group);
  RTresult RTAPI rtGroupGetContext(RTgroup group, RTcontext* context);

  RTresult RTAPI rtGroupSetAcceleration(RTgroup group, RTacceleration  acceleration);
  RTresult RTAPI rtGroupGetAcceleration(RTgroup group, RTacceleration* acceleration);

  RTresult RTAPI rtGroupSetChildCount(RTgroup group, unsigned int  count);
  RTresult RTAPI rtGroupGetChildCount(RTgroup group, unsigned int* count);
  RTresult RTAPI rtGroupSetChild     (RTgroup group, unsigned int index, RTobject  child);
  RTresult RTAPI rtGroupGetChild     (RTgroup group, unsigned int index, RTobject* child);
  RTresult RTAPI rtGroupGetChildType (RTgroup group, unsigned int index, RTobjecttype* type);

/************************************
 **
 **    Selector object
 **
 ***********************************/

  RTresult RTAPI rtSelectorCreate    (RTcontext context, RTselector* selector);
  RTresult RTAPI rtSelectorDestroy   (RTselector selector);
  RTresult RTAPI rtSelectorValidate  (RTselector selector);
  RTresult RTAPI rtSelectorGetContext(RTselector selector, RTcontext* context);

  RTresult RTAPI rtSelectorSetVisitProgram(RTselector selector, RTprogram  program);
  RTresult RTAPI rtSelectorGetVisitProgram(RTselector selector, RTprogram* program);

  RTresult RTAPI rtSelectorSetChildCount(RTselector selector, unsigned int  count);
  RTresult RTAPI rtSelectorGetChildCount(RTselector selector, unsigned int* count);
  RTresult RTAPI rtSelectorSetChild     (RTselector selector, unsigned int index, RTobject  child);
  RTresult RTAPI rtSelectorGetChild     (RTselector selector, unsigned int index, RTobject* child);
  RTresult RTAPI rtSelectorGetChildType (RTselector selector, unsigned int index, RTobjecttype* type);

  RTresult RTAPI rtSelectorDeclareVariable (RTselector selector, const char* name, RTvariable* v);
  RTresult RTAPI rtSelectorQueryVariable   (RTselector selector, const char* name, RTvariable* v);
  RTresult RTAPI rtSelectorRemoveVariable  (RTselector selector, RTvariable v);
  RTresult RTAPI rtSelectorGetVariableCount(RTselector selector, unsigned int* count);
  RTresult RTAPI rtSelectorGetVariable     (RTselector selector, unsigned int index, RTvariable* v);

/************************************
 **
 **    Transform object
 **
 ***********************************/

  RTresult RTAPI rtTransformCreate    (RTcontext context, RTtransform* transform);
  RTresult RTAPI rtTransformDestroy   (RTtransform transform);
  RTresult RTAPI rtTransformValidate  (RTtransform transform);
  RTresult RTAPI rtTransformGetContext(RTtransform transform, RTcontext* context);

  RTresult RTAPI rtTransformSetMatrix (RTtransform transform, int transpose, const float* matrix, const float* inverse_matrix);
  RTresult RTAPI rtTransformGetMatrix (RTtransform transform, int transpose, float* matrix, float* inverse_matrix);

  RTresult RTAPI rtTransformSetChild    (RTtransform transform, RTobject  child);
  RTresult RTAPI rtTransformGetChild    (RTtransform transform, RTobject* child);
  RTresult RTAPI rtTransformGetChildType(RTtransform transform, RTobjecttype* type);

/************************************
 **
 **    GeometryGroup object
 **
 ***********************************/

  RTresult RTAPI rtGeometryGroupCreate    (RTcontext context, RTgeometrygroup* geometrygroup);
  RTresult RTAPI rtGeometryGroupDestroy   (RTgeometrygroup geometrygroup);
  RTresult RTAPI rtGeometryGroupValidate  (RTgeometrygroup geometrygroup);
  RTresult RTAPI rtGeometryGroupGetContext(RTgeometrygroup geometrygroup, RTcontext* context);

  RTresult RTAPI rtGeometryGroupSetAcceleration(RTgeometrygroup geometrygroup, RTacceleration  acceleration);
  RTresult RTAPI rtGeometryGroupGetAcceleration(RTgeometrygroup geometrygroup, RTacceleration* acceleration);

  RTresult RTAPI rtGeometryGroupSetChildCount(RTgeometrygroup geometrygroup, unsigned int  count);
  RTresult RTAPI rtGeometryGroupGetChildCount(RTgeometrygroup geometrygroup, unsigned int* count);
  RTresult RTAPI rtGeometryGroupSetChild     (RTgeometrygroup geometrygroup, unsigned int index, RTgeometryinstance  geometryinstance);
  RTresult RTAPI rtGeometryGroupGetChild     (RTgeometrygroup geometrygroup, unsigned int index, RTgeometryinstance* geometryinstance);

/************************************
 **
 **    Acceleration object
 **
 ***********************************/

  RTresult RTAPI rtAccelerationCreate    (RTcontext context, RTacceleration* acceleration);
  RTresult RTAPI rtAccelerationDestroy   (RTacceleration acceleration);
  RTresult RTAPI rtAccelerationValidate  (RTacceleration acceleration);
  RTresult RTAPI rtAccelerationGetContext(RTacceleration acceleration, RTcontext* context);

  RTresult RTAPI rtAccelerationSetBuilder(RTacceleration acceleration, const char* builder);
  RTresult RTAPI rtAccelerationGetBuilder(RTacceleration acceleration, const char** return_string);
  RTresult RTAPI rtAccelerationSetTraverser(RTacceleration acceleration, const char* traverser);
  RTresult RTAPI rtAccelerationGetTraverser(RTacceleration acceleration, const char** return_string);
  RTresult RTAPI rtAccelerationSetProperty(RTacceleration acceleration, const char* name, const char* value);
  RTresult RTAPI rtAccelerationGetProperty(RTacceleration acceleration, const char* name, const char** return_string);

  RTresult RTAPI rtAccelerationGetDataSize(RTacceleration acceleration, RTsize* size);
  RTresult RTAPI rtAccelerationGetData    (RTacceleration acceleration, void* data);
  RTresult RTAPI rtAccelerationSetData    (RTacceleration acceleration, const void* data, RTsize size);

  RTresult RTAPI rtAccelerationMarkDirty(RTacceleration acceleration);
  RTresult RTAPI rtAccelerationIsDirty(RTacceleration acceleration, int* dirty);

/************************************
 **
 **    GeometryInstance object
 **
 ***********************************/

  RTresult RTAPI rtGeometryInstanceCreate    (RTcontext context, RTgeometryinstance* geometryinstance);
  RTresult RTAPI rtGeometryInstanceDestroy   (RTgeometryinstance geometryinstance);
  RTresult RTAPI rtGeometryInstanceValidate  (RTgeometryinstance geometryinstance);
  RTresult RTAPI rtGeometryInstanceGetContext(RTgeometryinstance geometryinstance, RTcontext* context);

  RTresult RTAPI rtGeometryInstanceSetGeometry(RTgeometryinstance geometryinstance, RTgeometry  geometry);
  RTresult RTAPI rtGeometryInstanceGetGeometry(RTgeometryinstance geometryinstance, RTgeometry* geometry);

  RTresult RTAPI rtGeometryInstanceSetMaterialCount(RTgeometryinstance geometryinstance, unsigned int  count);
  RTresult RTAPI rtGeometryInstanceGetMaterialCount(RTgeometryinstance geometryinstance, unsigned int* count);

  RTresult RTAPI rtGeometryInstanceSetMaterial(RTgeometryinstance geometryinstance, unsigned int idx, RTmaterial  material);
  RTresult RTAPI rtGeometryInstanceGetMaterial(RTgeometryinstance geometryinstance, unsigned int idx, RTmaterial* material);

  RTresult RTAPI rtGeometryInstanceDeclareVariable (RTgeometryinstance geometryinstance, const char* name, RTvariable* v);
  RTresult RTAPI rtGeometryInstanceQueryVariable   (RTgeometryinstance geometryinstance, const char* name, RTvariable* v);
  RTresult RTAPI rtGeometryInstanceRemoveVariable  (RTgeometryinstance geometryinstance, RTvariable v);
  RTresult RTAPI rtGeometryInstanceGetVariableCount(RTgeometryinstance geometryinstance, unsigned int* count);
  RTresult RTAPI rtGeometryInstanceGetVariable     (RTgeometryinstance geometryinstance, unsigned int index, RTvariable* v);

/************************************
 **
 **    Geometry object
 **
 ***********************************/

  RTresult RTAPI rtGeometryCreate    (RTcontext context, RTgeometry* geometry);
  RTresult RTAPI rtGeometryDestroy   (RTgeometry geometry);
  RTresult RTAPI rtGeometryValidate  (RTgeometry geometry);
  RTresult RTAPI rtGeometryGetContext(RTgeometry geometry, RTcontext* context);

  RTresult RTAPI rtGeometrySetPrimitiveCount(RTgeometry geometry, unsigned int  num_primitives);
  RTresult RTAPI rtGeometryGetPrimitiveCount(RTgeometry geometry, unsigned int* num_primitives);

  RTresult RTAPI rtGeometrySetBoundingBoxProgram(RTgeometry geometry, RTprogram  program);
  RTresult RTAPI rtGeometryGetBoundingBoxProgram(RTgeometry geometry, RTprogram* program);

  RTresult RTAPI rtGeometrySetIntersectionProgram(RTgeometry geometry, RTprogram  program);
  RTresult RTAPI rtGeometryGetIntersectionProgram(RTgeometry geometry, RTprogram* program);

  RTresult RTAPI rtGeometryMarkDirty(RTgeometry geometry);
  RTresult RTAPI rtGeometryIsDirty(RTgeometry geometry, int* dirty);

  RTresult RTAPI rtGeometryDeclareVariable (RTgeometry geometry, const char* name, RTvariable* v);
  RTresult RTAPI rtGeometryQueryVariable   (RTgeometry geometry, const char* name, RTvariable* v);
  RTresult RTAPI rtGeometryRemoveVariable  (RTgeometry geometry, RTvariable v);
  RTresult RTAPI rtGeometryGetVariableCount(RTgeometry geometry, unsigned int* count);
  RTresult RTAPI rtGeometryGetVariable     (RTgeometry geometry, unsigned int index, RTvariable* v);

/************************************
 **
 **    Material object
 **
 ***********************************/

  RTresult RTAPI rtMaterialCreate    (RTcontext context, RTmaterial* material);
  RTresult RTAPI rtMaterialDestroy   (RTmaterial material);
  RTresult RTAPI rtMaterialValidate  (RTmaterial material);
  RTresult RTAPI rtMaterialGetContext(RTmaterial material, RTcontext* context);

  RTresult RTAPI rtMaterialSetClosestHitProgram(RTmaterial material, unsigned int ray_type_index, RTprogram  program);
  RTresult RTAPI rtMaterialGetClosestHitProgram(RTmaterial material, unsigned int ray_type_index, RTprogram* program);

  RTresult RTAPI rtMaterialSetAnyHitProgram(RTmaterial material, unsigned int ray_type_index, RTprogram  program);
  RTresult RTAPI rtMaterialGetAnyHitProgram(RTmaterial material, unsigned int ray_type_index, RTprogram* program);

  RTresult RTAPI rtMaterialDeclareVariable (RTmaterial material, const char* name, RTvariable* v);
  RTresult RTAPI rtMaterialQueryVariable   (RTmaterial material, const char* name, RTvariable* v);
  RTresult RTAPI rtMaterialRemoveVariable  (RTmaterial material, RTvariable v);
  RTresult RTAPI rtMaterialGetVariableCount(RTmaterial material, unsigned int* count);
  RTresult RTAPI rtMaterialGetVariable     (RTmaterial material, unsigned int index, RTvariable* v);

/************************************
 **
 **    TextureSampler object
 **
 ***********************************/

  RTresult RTAPI rtTextureSamplerCreate    (RTcontext context, RTtexturesampler* texturesampler);
  RTresult RTAPI rtTextureSamplerDestroy   (RTtexturesampler texturesampler);
  RTresult RTAPI rtTextureSamplerValidate  (RTtexturesampler texturesampler);
  RTresult RTAPI rtTextureSamplerGetContext(RTtexturesampler texturesampler, RTcontext* context);

  RTresult RTAPI rtTextureSamplerSetMipLevelCount (RTtexturesampler texturesampler, unsigned int  num_mip_levels);
  RTresult RTAPI rtTextureSamplerGetMipLevelCount (RTtexturesampler texturesampler, unsigned int* num_mip_levels);

  RTresult RTAPI rtTextureSamplerSetArraySize(RTtexturesampler texturesampler, unsigned int  num_textures_in_array);
  RTresult RTAPI rtTextureSamplerGetArraySize(RTtexturesampler texturesampler, unsigned int* num_textures_in_array);

  RTresult RTAPI rtTextureSamplerSetWrapMode(RTtexturesampler texturesampler, unsigned int dimension, RTwrapmode wrapmode);
  RTresult RTAPI rtTextureSamplerGetWrapMode(RTtexturesampler texturesampler, unsigned int dimension, RTwrapmode* wrapmode);

  RTresult RTAPI rtTextureSamplerSetFilteringModes(RTtexturesampler texturesampler, RTfiltermode  minification, RTfiltermode  magnification, RTfiltermode  mipmapping);
  RTresult RTAPI rtTextureSamplerGetFilteringModes(RTtexturesampler texturesampler, RTfiltermode* minification, RTfiltermode* magnification, RTfiltermode* mipmapping);

  RTresult RTAPI rtTextureSamplerSetMaxAnisotropy(RTtexturesampler texturesampler, float value);
  RTresult RTAPI rtTextureSamplerGetMaxAnisotropy(RTtexturesampler texturesampler, float* value);

  RTresult RTAPI rtTextureSamplerSetReadMode(RTtexturesampler texturesampler, RTtexturereadmode  readmode);
  RTresult RTAPI rtTextureSamplerGetReadMode(RTtexturesampler texturesampler, RTtexturereadmode* readmode);

  RTresult RTAPI rtTextureSamplerSetIndexingMode(RTtexturesampler texturesampler, RTtextureindexmode  indexmode);
  RTresult RTAPI rtTextureSamplerGetIndexingMode(RTtexturesampler texturesampler, RTtextureindexmode* indexmode);

  RTresult RTAPI rtTextureSamplerSetBuffer(RTtexturesampler texturesampler, unsigned int texture_array_idx, unsigned int mip_level, RTbuffer  buffer);
  RTresult RTAPI rtTextureSamplerGetBuffer(RTtexturesampler texturesampler, unsigned int texture_array_idx, unsigned int mip_level, RTbuffer* buffer);

  RTresult RTAPI rtTextureSamplerGetId(RTtexturesampler texturesampler, int *texture_id);

/************************************
 **
 **    Buffer object
 **
 ***********************************/

  RTresult RTAPI rtBufferCreate         (RTcontext context, unsigned int bufferdesc, RTbuffer* buffer);
  RTresult RTAPI rtBufferDestroy        (RTbuffer buffer);
  RTresult RTAPI rtBufferValidate       (RTbuffer buffer);
  RTresult RTAPI rtBufferGetContext     (RTbuffer buffer, RTcontext* context);

  RTresult RTAPI rtBufferSetFormat     (RTbuffer buffer, RTformat  format);
  RTresult RTAPI rtBufferGetFormat     (RTbuffer buffer, RTformat* format);
  RTresult RTAPI rtBufferSetElementSize(RTbuffer buffer, RTsize  size_of_element);
  RTresult RTAPI rtBufferGetElementSize(RTbuffer buffer, RTsize* size_of_element);

  RTresult RTAPI rtBufferSetSize1D(RTbuffer buffer, RTsize  width);
  RTresult RTAPI rtBufferGetSize1D(RTbuffer buffer, RTsize* width);
  RTresult RTAPI rtBufferSetSize2D(RTbuffer buffer, RTsize  width, RTsize  height);
  RTresult RTAPI rtBufferGetSize2D(RTbuffer buffer, RTsize* width, RTsize* height);
  RTresult RTAPI rtBufferSetSize3D(RTbuffer buffer, RTsize  width, RTsize  height, RTsize  depth);
  RTresult RTAPI rtBufferGetSize3D(RTbuffer buffer, RTsize* width, RTsize* height, RTsize* depth);
  RTresult RTAPI rtBufferSetSizev (RTbuffer buffer, unsigned int dimensionality, const RTsize* dims);
  RTresult RTAPI rtBufferGetSizev (RTbuffer buffer, unsigned int dimensionality,       RTsize* dims);

  RTresult RTAPI rtBufferGetDimensionality(RTbuffer buffer, unsigned int* dimensionality);

  RTresult RTAPI rtBufferMap  (RTbuffer buffer, void** user_pointer);
  RTresult RTAPI rtBufferUnmap(RTbuffer buffer);

#ifdef __cplusplus
}
#endif

#endif /* __optix_optix_host_h__ */

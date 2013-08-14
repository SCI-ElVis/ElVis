

/****************************************************************************\
 *
 * Traversal API
 *
\****************************************************************************/

/**
 * \defgroup traversal rtuTraversal: traversal API allowing batch raycasting queries utilizing either OptiX or the CPU.
 *
 * The OptiX traversal API is demonstrated in the traversal sample within the
 * OptiX SDK.
 *
 * @{
 */

/**
 * \file optixu_traversal.h
 * \brief A simple API for performing raytracing queries using OptiX or the CPU.
 */

#ifndef _optixu_optux_traversal_h_
#define _optixu_optux_traversal_h_

#include "../optix.h"

#ifdef __cplusplus
extern "C" {
#endif

  /** 
   * \brief Structure encapsulating the result of a single ray query 
   */
  typedef struct {
    int   prim_id;  /**<  Index of the interesected triangle, -1 for miss */
    float t;        /**<  Ray t parameter of hit point                    */
  } RTUtraversalresult;


  /** 
   * \brief The type of ray query to be performed.
   *
   * See OptiX Programming Guide for explanation of any vs. closest hit queries.  Note
   * that in the case of RTU_QUERY_TYPE_ANY_HIT, the prim_id and t intersection values in
   * RTUtraversalresult will correspond to the first successful intersection. These values
   * may not be indicative of the closest intersection, only that there was at least one.
   */
  typedef enum { 
    RTU_QUERY_TYPE_ANY_HIT = 0,  /**< Perform any hit calculation     */
    RTU_QUERY_TYPE_CLOSEST_HIT,  /**< Perform closest hit calculation */
    RTU_QUERY_TYPE_COUNT
  } RTUquerytype;

  /** 
   * \brief The input format of the ray vector. 
   */
  typedef enum  {
    RTU_RAYFORMAT_ORIGIN_DIRECTION_TMIN_TMAX_INTERLEAVED = 0,
    RTU_RAYFORMAT_ORIGIN_DIRECTION_INTERLEAVED,
    RTU_RAYFORMAT_COUNT
  } RTUrayformat;

  /** 
   * \brief The input format of the triangles. 
   *
   * TRIANGLE_SOUP implies future use of rtuTraversalSetTriangles while
   * MESH implies use of rtuTraversalSetMesh.
   */
  typedef enum  {
    RTU_TRIFORMAT_MESH= 0,
    RTU_TRIFORMAT_TRIANGLE_SOUP,
    RTU_TRIFORMAT_COUNT
  } RTUtriformat;

  /** 
   * \brief Initialization options (static across life of traversal object).
   *
   * The rtuTraverse API supports both running on the CPU and GPU.  When
   * RTU_INITOPTION_NONE is specified GPU context creation is attempted.  If
   * that fails (such as when there isn't an NVIDIA GPU part present, the CPU
   * code path is automatically chosen.  Specifying RTU_INITOPTION_GPU_ONLY or
   * RTU_INITOPTION_CPU_ONLY will only use the GPU or CPU modes without
   * automatic transitions from one to the other.
   *
   * RTU_INITOPTION_CULL_BACKFACE will enable back face culling during
   * intersection.
   */
  typedef enum  {
    RTU_INITOPTION_NONE          = 0,
    RTU_INITOPTION_GPU_ONLY      = 1 << 0,
    RTU_INITOPTION_CPU_ONLY      = 1 << 1,
    RTU_INITOPTION_CULL_BACKFACE = 1 << 2
  } RTUinitoptions;

  typedef enum  {
    RTU_OUTPUT_NONE        = 0,
    RTU_OUTPUT_NORMAL      = 1 << 0, /*< float3 [x, y, z]                      */
    RTU_OUTPUT_BARYCENTRIC = 1 << 1, /*< float2 [alpha, beta] (gamma implicit) */
    RTU_OUTPUT_BACKFACING  = 1 << 2  /*< char   [1 | 0]                        */
  } RTUoutput;

  /** 
   * \brief Runtime options (can be set multiple times for a given traversal
   * object). 
   */
  typedef enum  {
    RTU_OPTION_INT_NUM_THREADS=0
  } RTUoption;


  /**
   * Opaque type.  Note that the *_api types should never be used directly.
   * Only the typedef target names will be guaranteed to remain unchanged.
   */
  typedef struct RTUtraversal_api*  RTUtraversal;


  /**
   * Create a traversal state and associate a context with it.  If context is a
   * null pointer a new context will be created internally.  The context should
   * also not be used for any other launch commands from the OptiX host API, nor
   * attached to multiple RTUtraversal objects at one time.
   *
   *   \param[out]  traversal   Return pointer for traverse state handle
   *   \param       query_type  Ray query type 
   *   \param       ray_format  Ray format 
   *   \param       tri_format  Triangle format 
   *   \param       outputs     OR'ed mask of requested RTUoutputs  
   *   \param       options     Bit vector of or'ed RTUinitoptions.
   *   \param       context     RTcontext used for internal object creation
   */
  RTresult RTAPI rtuTraversalCreate( RTUtraversal* traversal,
                                     RTUquerytype  query_type,
                                     RTUrayformat  ray_format,
                                     RTUtriformat  tri_format,
                                     unsigned int  outputs,
                                     unsigned int  options,
                                     RTcontext     context );

  /**
   * Returns the string associated with the error code and any additional
   * information from the last error.  If traversal is non-NULL return_string
   * only remains valid while traversal is live.
   * 
   *   \param       traversal      Traversal state handle.  Can be NULL.
   *   \param       code           Error code from last error
   *   \param[out]  return_string  Pointer to string with error message in it.
   */
  RTresult RTAPI rtuTraversalGetErrorString( RTUtraversal traversal,
                                             RTresult code,
                                             const char** return_string);
  /**
   * Set a runtime option.  Unlike initialization options, these options may be
   * set more than once for a given RTUtraversal instance.
   *
   *   \param    traversal      Traversal state handle
   *   \param    option         The option to be set
   *   \param    value          Value of the option
   */
  RTresult RTAPI rtuTraversalSetOption( RTUtraversal traversal,
                                        RTUoption    option,
                                        void*        value );

  /**
   * Specify triangle mesh to be intersected by the next call to
   * rtuTraversalLaunch.  Only one geometry set may be active at a time.
   * Subsequent calls to rtuTraversalSetTriangles or rtuTraversalSetMesh will
   * override any previously specified geometry.  No internal copies of the mesh
   * data are made.  The user should ensure that the mesh data remains valid
   * until after rtuTraversalTraverse has been called.  Counter-clockwise
   * winding is assumed for normal and backfacing computations.
   * 
   *   \param    traversal      Traversal state handle
   *   \param    num_verts      Vertex count
   *   \param    verts          Vertices [ v1_x, v1_y, v1_z, v2.x, ... ] 
   *   \param    num_tris       Triangle count
   *   \param    indices        Indices [ tri1_index1, tr1_index2, ... ]
   */
  RTresult RTAPI rtuTraversalSetMesh( RTUtraversal    traversal,
                                      unsigned int    num_verts,
                                      const float*    verts,
                                      unsigned int    num_tris,
                                      const unsigned* indices );

  /**
   * Specify triangle soup to be intersected by the next call to
   * rtuTraversalLaunch.  Only one geometry set may be active at a time.
   * Subsequent calls to rtuTraversalSetTriangles or rtuTraversalSetMesh will
   * override any previously specified geometry.  No internal copies of the
   * triangle data are made.  The user should ensure that the triangle data
   * remains valid until after rtuTraversalTraverse has been
   * called. Counter-clockwise winding is assumed for normal and backfacing
   * computations.
   *
   *   \param    traversal      Traversal state handle
   *   \param    num_tris       Triangle count
   *   \param    tris           Triangles [ tri1_v1.x, tri1_v1.y, tr1_v1.z, tri1_v2.x, ... ] 
   */
  RTresult RTAPI rtuTraversalSetTriangles( RTUtraversal traversal,
                                           unsigned int num_tris,
                                           const float* tris );

  /**
   * Specify acceleration data for current geometry.  Input acceleration data
   * should be result of rtuTraversalGetAccelData or rtAccelerationGetData call.
   *
   *   \param    traversal      Traversal state handle
   *   \param    data           Acceleration data
   *   \param    data_size      Size of acceleration data
   */
  RTresult RTAPI rtuTraversalSetAccelData( RTUtraversal traversal,
                                           const void*  data,
                                           RTsize       data_size );

  /**
   * Retrieve acceleration data size for current geometry.  Will force
   * acceleration build if necessary.
   *
   *   \param      traversal    Traversal state handle
   *   \param[out] data_size    Size of acceleration data
   */
  RTresult RTAPI rtuTraversalGetAccelDataSize( RTUtraversal traversal,
                                               RTsize*      data_size );
  
  /**
   * Retrieve acceleration data for current geometry.  Will force acceleration
   * build if necessary.  The data parameter should be preallocated and its
   * length should match return value of rtuTraversalGetAccelDataSize.
   *
   *   \param       traversal   Traversal state handle
   *   \param[out]  data        Acceleration data
   */
  RTresult RTAPI rtuTraversalGetAccelData( RTUtraversal traversal,
                                           void*        data );

  /**
   * Specify set of rays to be cast upon next call to rtuTraversalTraverse.
   * rtuTraversalMapRays obtains a pointer which can be used to copy the ray
   * data into.  Rays should be packed in the format described in
   * rtuTraversalCreate call.  When copying is completed rtuTraversalUnmapRays
   * should be called.  Note that this call invalidates any existing results
   * buffers until rtuTraversalTraverse is called again.
   *
   *   \param    traversal      Traversal state handle
   *   \param    num_rays       Number of rays to be traced
   *   \param    rays           Pointer to ray data
   */
  RTresult RTAPI rtuTraversalMapRays( RTUtraversal traversal,
                                      unsigned int num_rays,
                                      float** rays );

  /**
   * See rtuTraversalMapRays.
   */
  RTresult RTAPI rtuTraversalUnmapRays( RTUtraversal traversal );

  /**
   * Perform any necessary preprocessing (eg, acceleration structure building,
   * optix context compilation).  It is not necessary to call this function as
   * rtuTraversalTraverse will call this internally as necessary.
   *
   *   \param    traversal      Traversal state handle
   */
  RTresult RTAPI rtuTraversalPreprocess( RTUtraversal traversal );

  /**
   * Perform any necessary preprocessing (eg, acceleration structure building
   * and kernel compilation ) and cast current rays against current geometry.
   *
   *   \param    traversal      Traversal state handle
   */
  RTresult RTAPI rtuTraversalTraverse( RTUtraversal traversal );

  /**
   * Retrieve results of last rtuTraversal call.  Results can be copied from the
   * pointer returned by rtuTraversalMapResults and will have length 'num_rays'
   * as prescribed from the previous call to rtuTraversalMapRays.
   * rtuTraversalUnmapResults should be called when finished reading the
   * results.  Returned primitive ID of -1 indicates a ray miss.
   *
   *   \param      traversal    Traversal state handle
   *   \param[out] results      Pointer to results of last traverse 
   */  
  RTresult RTAPI rtuTraversalMapResults( RTUtraversal         traversal,
                                         RTUtraversalresult** results );

  /**
   * See rtuTraversalMapResults
   */
  RTresult RTAPI rtuTraversalUnmapResults( RTUtraversal         traversal );
  
  /**
   * Retrieve user-specified output from last rtuTraversal call. Output can be
   * copied from the pointer returned by rtuTraversalMapOutput and will have
   * length 'num_rays' from as prescribed from the previous call to
   * rtuTraversalSetRays.  For each RTUoutput, a single rtuTraversalMapOutput
   * pointers can be outstanding.  rtuTraversalUnmapOutput should be called when
   * finished reading the output.
   *
   * If requested output type was not turned on with a previous call to
   * rtuTraverseSetOutputs an error will be returned.  See RTUoutput enum for
   * description of output data formats for various outputs.
   *
   *   \param      traversal    Traversal state handle
   *   \param      which        Output type to be specified
   *   \param[out] output       Pointer to output from last traverse 
   */  
  RTresult RTAPI rtuTraversalMapOutput( RTUtraversal traversal,
                                        RTUoutput    which,
                                        void**       output );
  /**
   * See rtuTraversalMapOutput
   */
  RTresult RTAPI rtuTraversalUnmapOutput( RTUtraversal traversal,
                                          RTUoutput    which );
  /**
   * Clean up any internal memory associated with rtuTraversal operations.
   * Includes destruction of result buffers returned via rtuTraversalGetResults.
   * Invalidates traversal object.
   *
   *   \param    traversal      Traversal state handle
   */
  RTresult RTAPI rtuTraversalDestroy( RTUtraversal traversal );

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* _optixu_optux_traversal.h */

/** @} */

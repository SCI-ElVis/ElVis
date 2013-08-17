
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


///
/// \defgroup optixpp OptiXpp: C++ wrapper for the OptiX C API.
///
/// OptiXpp wraps each OptiX C API opaque type in a C++ class.  Most of the OptiXpp
/// class member functions map directly to C API function calls:
///   - VariableObj::getContext -> rtVariableGetContext
///   - ContextObj::createBuffer -> rtBufferCreate
/// 
/// Many classes have convenience functions which encapsulate a related group of 
/// OptiX functions.  For instance
/// \code
///   ContextObj::createBuffer(unsigned int type, RTformat format, RTsize width)
/// \endcode
/// provides the functionality of 
///   - \p rtBufferCreate
///   - \p rtBufferSetFormat
///   - \p rtBufferSetSize1D
///
/// in a single call.
///  
/// Manipulation of these classes is performed via reference counted
/// Handle class.  Rather than working with a ContextObj directly you would use a Context
/// instead, which is simply a typedef for <I>Handle<ContextObj></I>.  The OptiX SDK has
/// many examples of the use of OptiXpp.  In particular, sample5 and sample5pp are a good
/// place to look when learning OptiXpp as they are nearly identical programs, one
/// created with the C API and one with the C++ API.
///
/// @{
///

///
/// \file optixpp_namespace.h
/// \brief A C++ wrapper around the OptiX API.
/// 


#ifndef __optixu_optixpp_namespace_h__
#define __optixu_optixpp_namespace_h__

#include "../optix.h"

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include "../optix_d3d9_interop.h"
#  include "../optix_d3d10_interop.h"
#  include "../optix_d3d11_interop.h"
#endif
#include "../optix_gl_interop.h"
#include "../optix_cuda_interop.h"

#include <string>
#include <vector>
#include <iterator>
#include "optixu_vector_types.h"

//-----------------------------------------------------------------------------
//
// Doxygen group specifications
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// 
// C++ API
//
//-----------------------------------------------------------------------------

namespace optix {
  /// \addtogroup optixpp
  /// @{

  class AccelerationObj;
  class BufferObj;
  class ContextObj;
  class GeometryObj;
  class GeometryGroupObj;
  class GeometryInstanceObj;
  class GroupObj;
  class MaterialObj;
  class ProgramObj;
  class SelectorObj;
  class TextureSamplerObj;
  class TransformObj;
  class VariableObj;

  class APIObj;
  class ScopedObj;


  ///
  ///  \brief The Handle class is a reference counted handle class used to
  ///  manipulate API objects.
  ///
  ///  All interaction with API objects should be done via these handles and the
  ///  associated typedefs rather than direct usage of the objects.
  ///
  template<class T>
  class Handle {
  public:
    /// Default constructor initializes handle to null pointer
    Handle() : ptr(0) {}

    /// Takes a raw pointer to an API object and creates a handle
    Handle(T* ptr) : ptr(ptr) { ref(); }

    /// Takes a raw pointer of arbitrary type and creates a handle 
    template<class U>
    Handle(U* ptr) : ptr(ptr) { ref(); }

    /// Takes a handle of the same type and creates a handle
    Handle(const Handle<T>& copy) : ptr(copy.ptr) { ref(); }
    
    /// Takes a handle of some other type and creates a handle
    template<class U>
    Handle(const Handle<U>& copy) : ptr(copy.ptr) { ref(); }

    /// Assignment of handle with same underlying object type 
    Handle<T>& operator=(const Handle<T>& copy)
    { if(ptr != copy.ptr) { unref(); ptr = copy.ptr; ref(); } return *this; }
    
    /// Assignment of handle with different underlying object type 
    template<class U>
    Handle<T>& operator=( const Handle<U>& copy)
    { if(ptr != copy.ptr) { unref(); ptr = copy.ptr; ref(); } return *this; }

    /// Decrements reference count on the handled object
    ~Handle() { unref(); }

    /// Takes a base optix api opaque type and creates a handle to optixpp wrapper type 
    static Handle<T> take( typename T::api_t p ) { return p? new T(p) : 0; }
    /// Special version that takes an RTobject which must be cast up to the appropriate
    /// OptiX API opaque type.
    static Handle<T> take( RTobject p ) { return p? new T(static_cast<typename T::api_t>(p)) : 0; }

    /// Dereferences the handle
          T* operator->()           { return ptr; }
    const T* operator->() const     { return ptr; }

    /// Retrieve the handled object
          T* get()                  { return ptr; }
    const T* get() const            { return ptr; }

    /// implicit bool cast based on NULLness of wrapped pointer
    operator bool() const  { return ptr != 0; }

    /// Variable access operator.  This operator will query the API object for
    /// a variable with the given name, creating a new variable instance if
    /// necessary. Only valid for ScopedObjs.
    Handle<VariableObj> operator[](const std::string& varname);

    /// \brief Variable access operator.  Identical to operator[](const std::string& varname)
    ///
    /// Explicitly define char* version to avoid ambiguities between builtin
    /// operator[](int, char*) and Handle::operator[]( std::string ).  The
    /// problem lies in that a Handle can be cast to a bool then to an int
    /// which implies that:
    /// \code
    ///    Context context;
    ///    context["var"];
    /// \endcode
    /// can be interpreted as either
    /// \code
    ///    1["var"]; // Strange but legal way to index into a string (same as "var"[1] )
    /// \endcode
    /// or
    /// \code
    ///    context[ std::string("var") ];
    /// \endcode
    Handle<VariableObj> operator[](const char* varname);

    /// Static object creation.  Only valid for contexts.
    static Handle<T> create() { return T::create(); }

    /// Query the machine device count.  Only valid for contexts
    static unsigned int getDeviceCount() { return T::getDeviceCount(); }

  private:
    inline void ref() { if(ptr) ptr->addReference(); }
    inline void unref() { if(ptr && ptr->removeReference() == 0) delete ptr; }
    T* ptr;
  };


  //----------------------------------------------------------------------------

  typedef Handle<AccelerationObj>     Acceleration;     ///< Use this to manipulate RTacceleration objects.
  typedef Handle<BufferObj>           Buffer;           ///< Use this to manipulate RTbuffer objects.
  typedef Handle<ContextObj>          Context;          ///< Use this to manipulate RTcontext objects.
  typedef Handle<GeometryObj>         Geometry;         ///< Use this to manipulate RTgeometry objects.
  typedef Handle<GeometryGroupObj>    GeometryGroup;    ///< Use this to manipulate RTgeometrygroup objects.
  typedef Handle<GeometryInstanceObj> GeometryInstance; ///< Use this to manipulate RTgeometryinstance objects.
  typedef Handle<GroupObj>            Group;            ///< Use this to manipulate RTgroup objects.
  typedef Handle<MaterialObj>         Material;         ///< Use this to manipulate RTmaterial objects.
  typedef Handle<ProgramObj>          Program;          ///< Use this to manipulate RTprogram objects.
  typedef Handle<SelectorObj>         Selector;         ///< Use this to manipulate RTselector objects.
  typedef Handle<TextureSamplerObj>   TextureSampler;   ///< Use this to manipulate RTtexturesampler objects.
  typedef Handle<TransformObj>        Transform;        ///< Use this to manipulate RTtransform objects.
  typedef Handle<VariableObj>         Variable;         ///< Use this to manipulate RTvariable objects.


  //----------------------------------------------------------------------------


  ///
  /// \brief Exception class for error reporting from the OptiXpp API.
  /// 
  /// Encapsulates an error message, often the direct result of a failed OptiX C
  /// API function call and subsequent rtContextGetErrorString call.
  ///
  class Exception: public std::exception {
  public:
    /// Create exception
    Exception( const std::string& message, RTresult error_code = RT_ERROR_UNKNOWN )
      : m_message(message), m_error_code( error_code ) {}

    /// Virtual destructor (needed for virtual function calls inherited from
    /// std::exception).
    virtual ~Exception() throw() {}

    /// Retrieve the error message
    const std::string& getErrorString() const { return m_message; }
  
    /// Retrieve the error code 
    RTresult getErrorCode() const { return m_error_code; }

    /// Helper for creating exceptions from an RTresult code origination from
    /// an OptiX C API function call.
    static Exception makeException( RTresult code, RTcontext context );

    /// From std::exception
    virtual const char* what() const throw() { return getErrorString().c_str(); }
  private:
    std::string m_message;
    RTresult    m_error_code;
  };

  inline Exception Exception::makeException( RTresult code, RTcontext context )
  {
    const char* str;
    rtContextGetErrorString( context, code, &str);
    return Exception( std::string(str), code );
  }


  //----------------------------------------------------------------------------


  ///
  /// \brief Base class for all reference counted wrappers around OptiX C API
  /// opaque types.
  ///  
  /// Wraps:
  ///   - RTcontext
  ///   - RTbuffer
  ///   - RTgeometry
  ///   - RTgeometryinstance
  ///   - RTgeometrygroup
  ///   - RTgroup
  ///   - RTmaterial
  ///   - RTprogram
  ///   - RTselector
  ///   - RTtexturesampler
  ///   - RTtransform
  ///   - RTvariable
  ///
  class APIObj {
  public:
    APIObj() : ref_count(0) {}
    virtual ~APIObj() {}

    /// Increment the reference count for this object
    void addReference()    { ++ref_count; }
    /// Decrement the reference count for this object
    int  removeReference() { return --ref_count; }

    /// Retrieve the context this object is associated with.  See rt[ObjectType]GetContext.
    virtual Context getContext()const=0;

    /// Check the given result code and throw an error with appropriate message
    /// if the code is not RTsuccess
    virtual void checkError(RTresult code)const;
    virtual void checkError(RTresult code, Context context )const;
    
    void checkErrorNoGetContext(RTresult code)const;

    /// For backwards compatability.  Use Exception::makeException instead.
    static Exception makeException( RTresult code, RTcontext context );
  private:
    int ref_count;
  };

  inline Exception APIObj::makeException( RTresult code, RTcontext context )
  {
    return Exception::makeException( code, context );
  }


  //----------------------------------------------------------------------------


  ///
  /// \brief Base class for all wrapper objects which can be destroyed and validated.
  ///
  /// Wraps:
  ///   - RTcontext
  ///   - RTgeometry
  ///   - RTgeometryinstance
  ///   - RTgeometrygroup
  ///   - RTgroup
  ///   - RTmaterial
  ///   - RTprogram
  ///   - RTselector
  ///   - RTtexturesampler
  ///   - RTtransform
  ///
  class DestroyableObj : public APIObj {
  public:
    virtual ~DestroyableObj() {}

    /// call rt[ObjectType]Destroy on the underlying OptiX C object 
    virtual void destroy() = 0;

    /// call rt[ObjectType]Validate on the underlying OptiX C object 
    virtual void validate() = 0;
  };


  
  //----------------------------------------------------------------------------


  ///
  /// \brief Base class for all objects which are OptiX variable containers.
  ///
  /// Wraps:
  ///   - RTcontext
  ///   - RTgeometry
  ///   - RTgeometryinstance
  ///   - RTmaterial
  ///   - RTprogram
  ///
  class ScopedObj : public DestroyableObj {
  public:
    virtual ~ScopedObj() {}

    /// Declare a variable associated with this object.  See rt[ObjectType]DeclareVariable.
    /// Note that this function is wrapped by the convenience function Handle::operator[].
    virtual Variable declareVariable(const std::string& name) = 0;
    /// Query a variable associated with this object by name.  See rt[ObjectType]QueryVariable.
    /// Note that this function is wrapped by the convenience function Handle::operator[].
    virtual Variable queryVariable(const std::string& name) const = 0;
    /// Remove a variable associated with this object
    virtual void removeVariable(Variable v) = 0;
    /// Query the number of variables associated with this object.  Used along
    /// with ScopedObj::getVariable to iterate over variables in an object.
    /// See rt[ObjectType]GetVariableCount
    virtual unsigned int getVariableCount() const = 0;
    /// Query variable by index.  See rt[ObjectType]GetVariable.
    virtual Variable getVariable(unsigned int index) const = 0;
  };

  
  
  //----------------------------------------------------------------------------


  ///
  /// \brief Variable object wraps OptiX C API RTvariable type and its related function set.
  /// 
  /// See OptiX programming guide and API reference for complete description of
  /// the usage and behavior of RTvariable objects.  Creation and querying of
  /// Variables can be performed via the Handle::operator[] function of the scope
  /// object associated with the variable. For example: 
  /// \code 
  ///   my_context["new_variable"]->setFloat( 1.0f );
  /// \endcode
  /// will create a variable named \p new_variable on the object \p my_context if
  /// it does not already exist.  It will then set the value of that variable to
  /// be a float 1.0f.
  ///
  class VariableObj : public APIObj {
  public:

    Context getContext() const;

    /// \name Float setters
    /// Set variable to have a float value. 
    //@{
    /// Set variable value to a scalar float
    void setFloat(float f1);
    /// Set variable value to a float2 
    void setFloat(optix::float2 f);
    /// Set variable value to a float2 
    void setFloat(float f1, float f2);
    /// Set variable value to a float3 
    void setFloat(optix::float3 f);
    /// Set variable value to a float3 
    void setFloat(float f1, float f2, float f3);
    /// Set variable value to a float4 
    void setFloat(optix::float4 f);
    /// Set variable value to a float4 
    void setFloat(float f1, float f2, float f3, float f4);
    /// Set variable value to a scalar float
    void set1fv(const float* f);
    /// Set variable value to a float2 
    void set2fv(const float* f);
    /// Set variable value to a float3 
    void set3fv(const float* f);
    /// Set variable value to a float4 
    void set4fv(const float* f);
    //@}

    /// \name Int setters
    /// Set variable to have an int value. 
    //@{
    void setInt(int i1);
    void setInt(int i1, int i2);
    void setInt(optix::int2 i);
    void setInt(int i1, int i2, int i3);
    void setInt(optix::int3 i);
    void setInt(int i1, int i2, int i3, int i4);
    void setInt(optix::int4 i);
    void set1iv(const int* i);
    void set2iv(const int* i);
    void set3iv(const int* i);
    void set4iv(const int* i);
    //@}

    /// \name Unsigned int  setters
    /// Set variable to have an unsigned int value. 
    //@{
    void setUint(unsigned int u1);
    void setUint(unsigned int u1, unsigned int u2);
    void setUint(unsigned int u1, unsigned int u2, unsigned int u3);
    void setUint(unsigned int u1, unsigned int u2, unsigned int u3, unsigned int u4);
    void setUint(optix::uint2 u);
    void setUint(optix::uint3 u);
    void setUint(optix::uint4 u);
    void set1uiv(const unsigned int* u);
    void set2uiv(const unsigned int* u);
    void set3uiv(const unsigned int* u);
    void set4uiv(const unsigned int* u);
    //@}

    /// \name Matrix setters
    /// Set variable to have a Matrix value 
    //@{
    void setMatrix2x2fv(bool transpose, const float* m);
    void setMatrix2x3fv(bool transpose, const float* m);
    void setMatrix2x4fv(bool transpose, const float* m);
    void setMatrix3x2fv(bool transpose, const float* m);
    void setMatrix3x3fv(bool transpose, const float* m);
    void setMatrix3x4fv(bool transpose, const float* m);
    void setMatrix4x2fv(bool transpose, const float* m);
    void setMatrix4x3fv(bool transpose, const float* m);
    void setMatrix4x4fv(bool transpose, const float* m);
    //@}

    /// \name Numeric value getters 
    /// Query value of a variable with scalar numeric value 
    //@{
    float getFloat() const;
    unsigned int getUint() const;
    int getInt() const;
    //@}
    
#if 0
    // Not implemented yet...

    // The getFloat functions can be overloaded by parameter type.
    void getFloat(float* f);
    void getFloat(float* f1, float* f2);
    void getFloat(optix::float2* f);
    void getFloat(float* f1, float* f2, float* f3);
    void getFloat(optix::float3* f);
    void getFloat(float* f1, float* f2, float* f3, float* f4);
    void getFloat(optix::float4* f);
    // This one will need a different name to distinquish it from 'float getFloat()'.
    optix::float2 getFloat2();
    optix::float3 getFloat3();
    optix::float4 getFloat4();

    void get1fv(float* f);
    void get2fv(float* f);
    void get3fv(float* f);
    void get4fv(float* f);

    get1i (int* i1);
    get2i (int* i1, int* i2);
    get3i (int* i1, int* i2, int* i3);
    get4i (int* i1, int* i2, int* i3, int* i4);
    get1iv(int* i);
    get2iv(int* i);
    get3iv(int* i);
    get4iv(int* i);

    get1ui (unsigned int* u1);
    get2ui (unsigned int* u1, unsigned int* u2);
    get3ui (unsigned int* u1, unsigned int* u2, unsigned int* u3);
    get4ui (unsigned int* u1, unsigned int* u2, unsigned int* u3, unsigned int* u4);
    get1uiv(unsigned int* u);
    get2uiv(unsigned int* u);
    get3uiv(unsigned int* u);
    get4uiv(unsigned int* u);

    getMatrix2x2fv(bool transpose, float* m);
    getMatrix2x3fv(bool transpose, float* m);
    getMatrix2x4fv(bool transpose, float* m);
    getMatrix3x2fv(bool transpose, float* m);
    getMatrix3x3fv(bool transpose, float* m);
    getMatrix3x4fv(bool transpose, float* m);
    getMatrix4x2fv(bool transpose, float* m);
    getMatrix4x3fv(bool transpose, float* m);
    getMatrix4x4fv(bool transpose, float* m);
#endif


    /// \name OptiX API object setters 
    /// Set variable to have an OptiX API object as its value
    //@{
    void setBuffer(Buffer buffer);
    void set(Buffer buffer);
    void setTextureSampler(TextureSampler texturesample);
    void set(TextureSampler texturesample);
    void set(GeometryGroup group);
    void set(Group group);
    void set(Program program);
    void set(Selector selector);
    void set(Transform transform);
    //@}

    /// \name OptiX API object getters 
    /// Reitrieve OptiX API object value from a variable 
    //@{
    Buffer getBuffer() const;
    TextureSampler getTextureSampler() const;
    Program getProgram() const;
    //@}

    /// \name User data variable accessors 
    //@{
    /// Set the variable to a user defined type given the sizeof the user object
    void setUserData(RTsize size, const void* ptr);
    /// Retrieve a user defined type given the sizeof the user object
    void getUserData(RTsize size,       void* ptr) const;
    //@}

    /// Retrieve the name of the variable
    std::string getName() const;
    
    /// Retrieve the annotation associated with the variable 
    std::string getAnnotation() const;

    /// Query the object type of the variable
    RTobjecttype getType() const;

    /// Get the OptiX C API object wrapped by this instance
    RTvariable get();

    /// Get the size of the variable data in bytes (eg, float4 returns 4*sizeof(float) )
    RTsize getSize() const;

  private:
    typedef RTvariable api_t;

    RTvariable m_variable;
    VariableObj(RTvariable variable) : m_variable(variable) {}
    friend class Handle<VariableObj>;

  };

  template<class T>
  Handle<VariableObj> Handle<T>::operator[](const std::string& varname)
  {
    Variable v = ptr->queryVariable( varname );
    if( v.operator->() == 0)
      v = ptr->declareVariable( varname );
    return v;
  }

  template<class T>
  Handle<VariableObj> Handle<T>::operator[](const char* varname) 
  {
    return (*this)[ std::string( varname ) ]; 
  }

  
  //----------------------------------------------------------------------------


  ///
  /// \brief Context object wraps the OptiX C API RTcontext opaque type and its associated function set.
  ///
  class ContextObj : public ScopedObj {
  public:

    /// Call rtDeviceGetDeviceCount and returns number of valid devices
    static unsigned int getDeviceCount();

    /// Call rtDeviceGetAttribute and return the name of the device
    static std::string getDeviceName(int ordinal);
    
    /// Call rtDeviceGetAttribute and return the desired attribute value
    static void getDeviceAttribute(int ordinal, RTdeviceattribute attrib, RTsize size, void* p);

    /// Creates a Context object.  See rtContextCreate
    static Context create();

    /// Destroy Context and all of its associated objects.  See rtContextDestroy.
    void destroy();

    /// See rtContextValidate
    void validate();

    /// Retrieve the Context object associated with this APIObject.  In this case,
    /// simply returns itself.
    Context getContext() const;

    /// @{
    /// See APIObj::checkError
    void checkError(RTresult code)const;

    /// See rtContextGetErrroString
    std::string getErrorString( RTresult code ) const;
    /// @}

    /// @{
    /// See rtAccelerationCreate
    Acceleration createAcceleration(const char* builder, const char* traverser);

    /// Create a buffer with given RTbuffertype.  See rtBufferCreate.
    Buffer createBuffer(unsigned int type);
    /// Create a buffer with given RTbuffertype and RTformat.  See rtBufferCreate, rtBufferSetFormat
    Buffer createBuffer(unsigned int type, RTformat format);
    /// Create a buffer with given RTbuffertype, RTformat and dimension.  See rtBufferCreate,
    /// rtBufferSetFormat and rtBufferSetSize1D.
    Buffer createBuffer(unsigned int type, RTformat format, RTsize width);
    /// Create a buffer with given RTbuffertype, RTformat and dimension.  See rtBufferCreate,
    /// rtBufferSetFormat and rtBufferSetSize2D.
    Buffer createBuffer(unsigned int type, RTformat format, RTsize width, RTsize height);
    /// Create a buffer with given RTbuffertype, RTformat and dimension.  See rtBufferCreate,
    /// rtBufferSetFormat and rtBufferSetSize3D.
    Buffer createBuffer(unsigned int type, RTformat format, RTsize width, RTsize height, RTsize depth);

    /// Create a buffer for CUDA with given RTbuffertype.  See rtBufferCreate.
    Buffer createBufferForCUDA(unsigned int type);
    /// Create a buffer for CUDA with given RTbuffertype and RTformat.  See rtBufferCreate, rtBufferSetFormat
    Buffer createBufferForCUDA(unsigned int type, RTformat format);
    /// Create a buffer for CUDA with given RTbuffertype, RTformat and dimension.  See rtBufferCreate,
    /// rtBufferSetFormat and rtBufferSetSize1D.
    Buffer createBufferForCUDA(unsigned int type, RTformat format, RTsize width);
    /// Create a buffer for CUDA with given RTbuffertype, RTformat and dimension.  See rtBufferCreate,
    /// rtBufferSetFormat and rtBufferSetSize2D.
    Buffer createBufferForCUDA(unsigned int type, RTformat format, RTsize width, RTsize height);
    /// Create a buffer for CUDA with given RTbuffertype, RTformat and dimension.  See rtBufferCreate,
    /// rtBufferSetFormat and rtBufferSetSize3D.
    Buffer createBufferForCUDA(unsigned int type, RTformat format, RTsize width, RTsize height, RTsize depth);

    /// Create buffer from GL buffer object.  See rtBufferCreateFromGLBO
    Buffer createBufferFromGLBO(unsigned int type, unsigned int vbo);

    /// Create TextureSampler from GL image.  See rtTextureSamplerCreateFromGLImage
    TextureSampler createTextureSamplerFromGLImage(unsigned int id, RTgltarget target);

#ifdef _WIN32
    /// Create buffer from D3D9 buffer object.  Windows only.  See rtBufferCreateFromD3D9Resource.
    Buffer createBufferFromD3D9Resource(unsigned int type, IDirect3DResource9 *pResource);
    /// Create buffer from D3D10 buffer object.  Windows only.  See rtBufferCreateFromD3D10Resource.
    Buffer createBufferFromD3D10Resource(unsigned int type, ID3D10Resource *pResource);
    /// Create buffer from D3D11 buffer object.  Windows only.  See rtBufferCreateFromD3D11Resource.
    Buffer createBufferFromD3D11Resource(unsigned int type, ID3D11Resource *pResource);

    /// Create TextureSampler from D3D9 image.  Windows only.  See rtTextureSamplerCreateFromD3D9Resource.
    TextureSampler createTextureSamplerFromD3D9Resource(IDirect3DResource9 *pResource);
    /// Create TextureSampler from D3D10 image.  Windows only.  See rtTextureSamplerCreateFromD3D10Resource.
    TextureSampler createTextureSamplerFromD3D10Resource(ID3D10Resource *pResource);
    /// Create TextureSampler from D3D11 image.  Windows only.  See rtTextureSamplerCreateFromD3D11Resource.
    TextureSampler createTextureSamplerFromD3D11Resource(ID3D11Resource *pResource);
#endif

    /// See rtGeometryCreate
    Geometry createGeometry();
    /// See rtGeometryInstanceCreate
    GeometryInstance createGeometryInstance();
    /// Create a geometry instance with a Geometry object and a set of associated materials.  See
    /// rtGeometryInstanceCreate, rtGeometryInstanceSetMaterialCount, and rtGeometryInstanceSetMaterial
    template<class Iterator>
    GeometryInstance createGeometryInstance( Geometry geometry, Iterator matlbegin, Iterator matlend );

    /// See rtGroupCreate
    Group createGroup();
    /// Create a Group with a set of child nodes.  See rtGroupCreate, rtGroupSetChildCount and
    /// rtGroupSetChild
    template<class Iterator>
    Group createGroup( Iterator childbegin, Iterator childend );

    /// See rtGeometryGroupCreate
    GeometryGroup createGeometryGroup();
    /// Create a GeometryGroup with a set of child nodes.  See rtGeometryGroupCreate,
    /// rtGeometryGroupSetChildCount and rtGeometryGroupSetChild
    template<class Iterator>
    GeometryGroup createGeometryGroup( Iterator childbegin, Iterator childend );

    /// See rtTransformCreate
    Transform createTransform();

    /// See rtMaterialCreate
    Material createMaterial();

    /// See rtProgramCreateFromPTXFile
    Program createProgramFromPTXFile  ( const std::string& ptx, const std::string& program_name );
    /// See rtProgramCreateFromPTXString
    Program createProgramFromPTXString( const std::string& ptx, const std::string& program_name );

    /// See rtSelectorCreate
    Selector createSelector();

    /// See rtTextureSamplerCreate
    TextureSampler createTextureSampler();
    /// @}

    /// @{
    /// See rtContextSetDevices
    template<class Iterator>
    void setDevices(Iterator begin, Iterator end);

#ifdef _WIN32
    /// Set the D3D device assocaiated with this context (Windows only).  See rtContextSetD3D9Device.
    void setD3D9Device(IDirect3DDevice9* device);
    /// Set the D3D device assocaiated with this context (Windows only).  See rtContextSetD3D10Device.
    void setD3D10Device(ID3D10Device* device);
    /// Set the D3D device assocaiated with this context (Windows only).  See rtContextSetD3D11Device.
    void setD3D11Device(ID3D11Device* device);
#endif

    /// See rtContextGetDevices.  This returns the list of currently enabled devices.
    std::vector<int> getEnabledDevices() const;

    /// See rtContextGetDeviceCount.  As opposed to getDeviceCount, this returns only the
    /// number of enabled devices.
    unsigned int getEnabledDeviceCount() const;
    /// @}

    /// @{
    /// See rtContextGetAttribute
    int getMaxTextureCount() const;

    /// See rtContextGetAttribute
    int getCPUNumThreads() const;

    /// See rtContextGetAttribute
    RTsize getUsedHostMemory() const;

    /// See rtContextGetAttribute
    int getGPUPagingActive() const;

    /// See rtContextGetAttribute
    int getGPUPagingForcedOff() const;

    /// See rtContextGetAttribute
    RTsize getAvailableDeviceMemory(int ordinal) const;
    /// @}

    /// @{
    /// See rtContextSetAttribute
    void setCPUNumThreads(int cpu_num_threads);

    /// See rtContextSetAttribute
    void setGPUPagingForcedOff(int gpu_paging_forced_off);
    /// @}

    /// @{    
    /// See rtContextSetStackSize
    void setStackSize(RTsize  stack_size_bytes);
    /// See rtContextGetStackSize
    RTsize getStackSize() const;

    /// See rtContextSetTimeoutCallback
    /// RTtimeoutcallback is defined as typedef int (*RTtimeoutcallback)(void).
    void setTimeoutCallback(RTtimeoutcallback callback, double min_polling_seconds);

    /// See rtContextSetEntryPointCount
    void setEntryPointCount(unsigned int  num_entry_points);
    /// See rtContextgetEntryPointCount
    unsigned int getEntryPointCount() const;

    /// See rtContextSetRayTypeCount
    void setRayTypeCount(unsigned int  num_ray_types);
    /// See rtContextGetRayTypeCount
    unsigned int getRayTypeCount() const;
    /// @}

    /// @{
    /// See rtContextSetRayGenerationProgram
    void setRayGenerationProgram(unsigned int entry_point_index, Program  program);
    /// See rtContextGetRayGenerationProgram
    Program getRayGenerationProgram(unsigned int entry_point_index) const;

    /// See rtContextSetExceptionProgram
    void setExceptionProgram(unsigned int entry_point_index, Program  program);
    /// See rtContextGetExceptionProgram
    Program getExceptionProgram(unsigned int entry_point_index) const;

    /// See rtContextSetExceptionEnabled
    void setExceptionEnabled( RTexception exception, bool enabled );
    /// See rtContextGetExceptionEnabled
    bool getExceptionEnabled( RTexception exception ) const;

    /// See rtContextSetMissProgram
    void setMissProgram(unsigned int ray_type_index, Program  program);
    /// See rtContextGetMissProgram
    Program getMissProgram(unsigned int ray_type_index) const;
    /// @}

    /// See rtContextCompile
    void compile();

    /// @{
    /// See rtContextLaunch1D
    void launch(unsigned int entry_point_index, RTsize image_width);
    /// See rtContextLaunch2D
    void launch(unsigned int entry_point_index, RTsize image_width, RTsize image_height);
    /// See rtContextLaunch3D
    void launch(unsigned int entry_point_index, RTsize image_width, RTsize image_height, RTsize image_depth);
    /// @}

    /// See rtContextGetRunningState
    int getRunningState() const;

    /// @{
    /// See rtContextSetPrintEnabled
    void setPrintEnabled(bool enabled);
    /// See rtContextGetPrintEnabled
    bool getPrintEnabled() const;
    /// See rtContextSetPrintBufferSize
    void setPrintBufferSize(RTsize buffer_size_bytes);
    /// See rtContextGetPrintBufferSize
    RTsize getPrintBufferSize() const;
    /// See rtContextSetPrintLaunchIndex.
    void setPrintLaunchIndex(int x, int y=-1, int z=-1);
    /// See rtContextGetPrintLaunchIndex
    optix::int3 getPrintLaunchIndex() const;
    /// @}

    /// @{
    Variable declareVariable (const std::string& name);
    Variable queryVariable   (const std::string& name) const;
    void     removeVariable  (Variable v);
    unsigned int getVariableCount() const;
    Variable getVariable     (unsigned int index) const;
    /// @}

    /// Return the OptiX C API RTcontext object
    RTcontext get();
  private:
    typedef RTcontext api_t;

    virtual ~ContextObj() {}
    RTcontext m_context;
    ContextObj(RTcontext context) : m_context(context) {}
    friend class Handle<ContextObj>;
  };


  //----------------------------------------------------------------------------
  

  ///
  /// \brief Program object wraps the OptiX C API RTprogram opaque type and its associated function set.
  ///
  class ProgramObj : public ScopedObj {
  public:
    void destroy();
    void validate();

    Context getContext() const;

    Variable declareVariable (const std::string& name);
    Variable queryVariable   (const std::string& name) const;
    void     removeVariable  (Variable v);
    unsigned int getVariableCount() const;
    Variable getVariable     (unsigned int index) const;

    RTprogram get();
  private:
    typedef RTprogram api_t;
    virtual ~ProgramObj() {}
    RTprogram m_program;
    ProgramObj(RTprogram program) : m_program(program) {}
    friend class Handle<ProgramObj>;
  };


  //----------------------------------------------------------------------------


  ///
  /// \brief Group wraps the OptiX C API RTgroup opaque type and its associated function set.
  ///
  class GroupObj : public DestroyableObj {
  public:
    void destroy();
    void validate();

    Context getContext() const;

    /// @{
    /// Set the Acceleration structure for this group.  See rtGroupSetAcceleration.
    void setAcceleration(Acceleration acceleration);
    /// Query the Acceleration structure for this group.  See rtGroupGetAcceleration.
    Acceleration getAcceleration() const;
    /// @}

    /// @{
    /// Set the number of children for this group.  See rtGroupSetChildCount.
    void setChildCount(unsigned int  count);
    /// Query the number of children for this group.  See rtGroupGetChildCount.
    unsigned int getChildCount() const;

    /// Set an indexed child within this group.  See rtGroupSetChild.
    template< typename T > void setChild(unsigned int index, T child);
    /// Query an indexed child within this group.  See rtGroupGetChild.
    template< typename T > T getChild(unsigned int index) const;
    /// @}

    /// Get the underlying OptiX C API RTgroup opaque pointer.
    RTgroup get();

  private:
    typedef RTgroup api_t;
    virtual ~GroupObj() {}
    RTgroup m_group;
    GroupObj(RTgroup group) : m_group(group) {}
    friend class Handle<GroupObj>;
  };


  //----------------------------------------------------------------------------


  ///
  /// \brief GeometryGroup wraps the OptiX C API RTgeometrygroup opaque type and its associated function set.
  ///
  class GeometryGroupObj : public DestroyableObj {
  public:
    void destroy();
    void validate();
    Context getContext() const;

    /// @{
    /// Set the Acceleration structure for this group.  See rtGeometryGroupSetAcceleration.
    void setAcceleration(Acceleration acceleration);
    /// Query the Acceleration structure for this group.  See rtGeometryGroupGetAcceleration.
    Acceleration getAcceleration() const;
    /// @}

    /// @{
    /// Set the number of children for this group.  See rtGeometryGroupSetChildCount.
    void setChildCount(unsigned int  count);
    /// Query the number of children for this group.  See rtGeometryGroupGetChildCount.
    unsigned int getChildCount() const;

    /// Set an indexed GeometryInstance child of this group.  See rtGeometryGroupSetChild.
    void setChild(unsigned int index, GeometryInstance geometryinstance);
    /// Query an indexed GeometryInstance within this group.  See rtGeometryGroupGetChild.
    GeometryInstance getChild(unsigned int index) const;
    /// @}

    /// Get the underlying OptiX C API RTgeometrygroup opaque pointer.
    RTgeometrygroup get();

  private:
    typedef RTgeometrygroup api_t;
    virtual ~GeometryGroupObj() {}
    RTgeometrygroup m_geometrygroup;
    GeometryGroupObj(RTgeometrygroup geometrygroup) : m_geometrygroup(geometrygroup) {}
    friend class Handle<GeometryGroupObj>;
  };


  //----------------------------------------------------------------------------


  ///
  /// \brief Transform wraps the OptiX C API RTtransform opaque type and its associated function set.
  ///
  class TransformObj  : public DestroyableObj {
  public:
    void destroy();
    void validate();
    Context getContext() const;

    /// @{
    /// Set the child node of this transform.  See rtTransformSetChild.
    template< typename T > void setChild(T child);
    /// Set the child node of this transform.  See rtTransformGetChild.
    template< typename T > T getChild() const;
    /// @}

    /// @{
    /// Set the transform matrix for this node.  See rtTransformSetMatrix.
    void setMatrix(bool transpose, const float* matrix, const float* inverse_matrix);
    /// Get the transform matrix for this node.  See rtTransformGetMatrix.
    void getMatrix(bool transpose, float* matrix, float* inverse_matrix) const;
    /// @}

    /// Get the underlying OptiX C API RTtransform opaque pointer.
    RTtransform get();

  private:
    typedef RTtransform api_t;
    virtual ~TransformObj() {}
    RTtransform m_transform;
    TransformObj(RTtransform transform) : m_transform(transform) {}
    friend class Handle<TransformObj>;
  };


  //----------------------------------------------------------------------------


  ///
  /// \brief Selector wraps the OptiX C API RTselector opaque type and its associated function set.
  ///
  class SelectorObj : public DestroyableObj {
  public:
    void destroy();
    void validate();
    Context getContext() const;

    /// @{
    /// Set the visitor program for this selector.  See rtSelectorSetVisitProgram
    void setVisitProgram(Program  program);
    /// Get the visitor program for this selector.  See rtSelectorGetVisitProgram
    Program getVisitProgram() const;
    /// @}

    /// @{
    /// Set the number of children for this group.  See rtSelectorSetChildCount.
    void setChildCount(unsigned int  count);
    /// Query the number of children for this group.  See rtSelectorGetChildCount.
    unsigned int getChildCount() const;

    /// Set an indexed child child of this group.  See rtSelectorSetChild.
    template< typename T > void setChild(unsigned int index, T child);
    /// Query an indexed child within this group.  See rtSelectorGetChild.
    template< typename T > T getChild(unsigned int index) const;
    /// @}

    /// @{
    Variable declareVariable (const std::string& name);
    Variable queryVariable   (const std::string& name) const;
    void     removeVariable  (Variable v);
    unsigned int getVariableCount() const;
    Variable getVariable     (unsigned int index) const;
    /// @}

    /// Get the underlying OptiX C API RTselector opaque pointer.
    RTselector get();

  private:
    typedef RTselector api_t;
    virtual ~SelectorObj() {}
    RTselector m_selector;
    SelectorObj(RTselector selector) : m_selector(selector) {}
    friend class Handle<SelectorObj>;
  };

  
  //----------------------------------------------------------------------------


  ///
  /// \brief Acceleration wraps the OptiX C API RTacceleration opaque type and its associated function set.
  ///
  class AccelerationObj : public DestroyableObj {
  public:
    void destroy();
    void validate();
    Context getContext() const;

    /// @{
    /// Mark the acceleration as needing a rebuild.  See rtAccelerationMarkDirty.
    void markDirty();
    /// Query if the acceleration needs a rebuild.  See rtAccelerationIsDirty.
    bool isDirty() const;
    /// @}

    /// @{
    /// Set properties specifying Acceleration builder/traverser behavior.
    /// See rtAccelerationSetProperty.
    void        setProperty( const std::string& name, const std::string& value );
    /// Query properties specifying Acceleration builder/traverser behavior.
    /// See rtAccelerationGetProperty.
    std::string getProperty( const std::string& name ) const;

    /// Specify the acceleration structure builder.  See rtAccelerationSetBuilder.
    void        setBuilder(const std::string& builder);
    /// Query the acceleration structure builder.  See rtAccelerationGetBuilder.
    std::string getBuilder() const;
    /// Specify the acceleration structure traverser.  See rtAccelerationSetTraverser.
    void        setTraverser(const std::string& traverser);
    /// Query the acceleration structure traverser.  See rtAccelerationGetTraverser.
    std::string getTraverser() const;
    /// @}

    /// @{
    /// Query the size of the marshalled acceleration data.  See rtAccelerationGetDataSize.
    RTsize getDataSize() const;
    /// Get the marshalled acceleration data.  See rtAccelerationGetData.
    void   getData( void* data ) const;
    /// Specify the acceleration structure via marshalled acceleration data.  See rtAccelerationSetData.
    void   setData( const void* data, RTsize size );
    /// @}

    /// Get the underlying OptiX C API RTacceleration opaque pointer.
    RTacceleration get();

  private:
    typedef RTacceleration api_t;
    virtual ~AccelerationObj() {}
    RTacceleration m_acceleration;
    AccelerationObj(RTacceleration acceleration) : m_acceleration(acceleration) {}
    friend class Handle<AccelerationObj>;
  };


  //----------------------------------------------------------------------------


  ///
  /// \brief GeometryInstance wraps the OptiX C API RTgeometryinstance acceleration
  /// opaque type and its associated function set.
  ///
  class GeometryInstanceObj : public ScopedObj {
  public:
    void destroy();
    void validate();
    Context getContext() const;

    /// @{
    /// Set the geometry object associated with this instance.  See rtGeometryInstanceSetGeometry.
    void setGeometry(Geometry  geometry);
    /// Get the geometry object associated with this instance.  See rtGeometryInstanceGetGeometry.
    Geometry getGeometry() const;

    /// Set the number of materials associated with this instance.  See rtGeometryInstanceSetMaterialCount.
    void setMaterialCount(unsigned int  count);
    /// Query the number of materials associated with this instance.  See rtGeometryInstanceGetMaterialCount.
    unsigned int getMaterialCount() const;

    /// Set the material at given index.  See rtGeometryInstanceSetMaterial.
    void setMaterial(unsigned int idx, Material  material);
    /// Get the material at given index.  See rtGeometryInstanceGetMaterial.
    Material getMaterial(unsigned int idx) const;

    /// Adds the provided material and returns the index to newly added material; increases material count by one.
    unsigned int addMaterial(Material material);
    /// @}

    /// @{
    Variable declareVariable (const std::string& name);
    Variable queryVariable   (const std::string& name) const;
    void     removeVariable  (Variable v);
    unsigned int getVariableCount() const;
    Variable getVariable     (unsigned int index) const;
    /// @}

    /// Get the underlying OptiX C API RTgeometryinstance opaque pointer.
    RTgeometryinstance get();

  private:
    typedef RTgeometryinstance api_t;
    virtual ~GeometryInstanceObj() {}
    RTgeometryinstance m_geometryinstance;
    GeometryInstanceObj(RTgeometryinstance geometryinstance) : m_geometryinstance(geometryinstance) {}
    friend class Handle<GeometryInstanceObj>;
  };


  //----------------------------------------------------------------------------


  ///
  /// \brief Geometry wraps the OptiX C API RTgeometry opaque type and its associated function set.
  /// 
  class GeometryObj : public ScopedObj {
  public:
    void destroy();
    void validate();
    Context getContext() const;

    /// @{
    /// Mark this geometry as dirty, causing rebuild of parent groups acceleration.  See rtGeometryMarkDirty.
    void markDirty();
    /// Query whether this geometry has been marked dirty.  See rtGeometryIsDirty.
    bool isDirty() const;
    /// @}

    /// @{
    /// Set the number of primitives in this geometry objects (eg, number of triangles in mesh).
    /// See rtGeometrySetPrimitiveCount
    void setPrimitiveCount(unsigned int  num_primitives);
    /// Query the number of primitives in this geometry objects (eg, number of triangles in mesh).
    /// See rtGeometryGetPrimitiveCount
    unsigned int getPrimitiveCount() const;
    /// @}

    /// @{
    /// Set the bounding box program for this geometry.  See rtGeometrySetBoundingBoxProgram.
    void setBoundingBoxProgram(Program  program);
    /// Get the bounding box program for this geometry.  See rtGeometryGetBoundingBoxProgram.
    Program getBoundingBoxProgram() const;

    /// Set the intersection program for this geometry.  See rtGeometrySetIntersectionProgram.
    void setIntersectionProgram(Program  program);
    /// Get the intersection program for this geometry.  See rtGeometryGetIntersectionProgram.
    Program getIntersectionProgram() const;
    /// @}

    /// @{
    Variable declareVariable (const std::string& name);
    Variable queryVariable   (const std::string& name) const;
    void     removeVariable  (Variable v);
    unsigned int getVariableCount() const;
    Variable getVariable     (unsigned int index) const;
    /// @}

    /// Get the underlying OptiX C API RTgeometry opaque pointer.
    RTgeometry get();

  private:
    typedef RTgeometry api_t;
    virtual ~GeometryObj() {}
    RTgeometry m_geometry;
    GeometryObj(RTgeometry geometry) : m_geometry(geometry) {}
    friend class Handle<GeometryObj>;
  };


  //----------------------------------------------------------------------------


  ///
  /// \brief Material wraps the OptiX C API RTmaterial opaque type and its associated function set.
  ///
  class MaterialObj : public ScopedObj {
  public:
    void destroy();
    void validate();
    Context getContext() const;

    /// @{
    /// Set closest hit program for this material at the given \a ray_type index.  See rtMaterialSetClosestHitProgram.
    void setClosestHitProgram(unsigned int ray_type_index, Program  program);
    /// Get closest hit program for this material at the given \a ray_type index.  See rtMaterialGetClosestHitProgram.
    Program getClosestHitProgram(unsigned int ray_type_index) const;

    /// Set any hit program for this material at the given \a ray_type index.  See rtMaterialSetAnyHitProgram.
    void setAnyHitProgram(unsigned int ray_type_index, Program  program);
    /// Get any hit program for this material at the given \a ray_type index.  See rtMaterialGetAnyHitProgram.
    Program getAnyHitProgram(unsigned int ray_type_index) const;
    /// @}

    /// @{
    Variable declareVariable (const std::string& name);
    Variable queryVariable   (const std::string& name) const;
    void     removeVariable  (Variable v);
    unsigned int getVariableCount() const;
    Variable getVariable     (unsigned int index) const;
    /// @}

    /// Get the underlying OptiX C API RTmaterial opaque pointer.
    RTmaterial get();
  private:
    typedef RTmaterial api_t;
    virtual ~MaterialObj() {}
    RTmaterial m_material;
    MaterialObj(RTmaterial material) : m_material(material) {}
    friend class Handle<MaterialObj>;
  };


  //----------------------------------------------------------------------------


  ///
  /// \brief TextureSampler wraps the OptiX C API RTtexturesampler opaque type and its associated function set.
  ///
  class TextureSamplerObj : public DestroyableObj {
  public:
    void destroy();
    void validate();
    Context getContext() const;

    /// @{
    /// Set the number of mip levels for this sampler.  See rtTextureSamplerSetMipLevelCount.
    void setMipLevelCount (unsigned int  num_mip_levels);
    /// Query the number of mip levels for this sampler.  See rtTextureSamplerGetMipLevelCount.
    unsigned int getMipLevelCount () const;

    /// Set the texture array size for this sampler.  See rtTextureSamplerSetArraySize
    void setArraySize(unsigned int  num_textures_in_array);
    /// Query the texture array size for this sampler.  See rtTextureSamplerGetArraySize
    unsigned int getArraySize() const;

    /// Set the texture wrap mode for this sampler.  See rtTextureSamplerSetWrapMode
    void setWrapMode(unsigned int dim, RTwrapmode wrapmode);
    /// Query the texture wrap mode for this sampler.  See rtTextureSamplerGetWrapMode
    RTwrapmode getWrapMode(unsigned int dim) const;

    /// Set filtering modes for this sampler.  See rtTextureSamplerSetFilteringModes.
    void setFilteringModes(RTfiltermode  minification, RTfiltermode  magnification, RTfiltermode  mipmapping);
    /// Query filtering modes for this sampler.  See rtTextureSamplerGetFilteringModes.
    void getFilteringModes(RTfiltermode& minification, RTfiltermode& magnification, RTfiltermode& mipmapping) const;

    /// Set maximum anisotropy for this sampler.  See rtTextureSamplerSetMaxAnisotropy.
    void setMaxAnisotropy(float value);
    /// Query maximum anisotropy for this sampler.  See rtTextureSamplerGetMaxAnisotropy.
    float getMaxAnisotropy() const;

    /// Set texture read mode for this sampler.  See rtTextureSamplerSetReadMode.
    void setReadMode(RTtexturereadmode  readmode);
    /// Query texture read mode for this sampler.  See rtTextureSamplerGetReadMode.
    RTtexturereadmode getReadMode() const;

    /// Set texture indexing mode for this sampler.  See rtTextureSamplerSetIndexingMode.
    void setIndexingMode(RTtextureindexmode  indexmode);
    /// Query texture indexing mode for this sampler.  See rtTextureSamplerGetIndexingMode.
    RTtextureindexmode getIndexingMode() const;
    /// @}

    /// @{
    /// Returns the device-side ID of this sampler.
    int getId() const;
    /// @}

    /// @{
    /// Set the underlying buffer used for texture storage.  rtTextureSamplerSetBuffer.
    void setBuffer(unsigned int texture_array_idx, unsigned int mip_level, Buffer buffer);
    /// Get the underlying buffer used for texture storage.  rtTextureSamplerGetBuffer.
    Buffer getBuffer(unsigned int texture_array_idx, unsigned int mip_level) const;
    /// @}

    /// Get the underlying OptiX C API RTtexturesampler opaque pointer.
    RTtexturesampler get();

    /// @{
    /// Declare the texture's buffer as mutable and inaccessible by OptiX.  See rtTextureSamplerGLRegister.
    void registerGLTexture();
    /// Unregister the texture's buffer, re-enabling OptiX operations.  See rtTextureSamplerGLUnregister.
    void unregisterGLTexture();
    /// @}

#ifdef _WIN32

    /// @{
    /// Declare the texture's buffer as mutable and inaccessible by OptiX.  See rtTextureSamplerD3D9Register.
    void registerD3D9Texture();
    /// Declare the texture's buffer as mutable and inaccessible by OptiX.  See rtTextureSamplerD3D10Register.
    void registerD3D10Texture();
    /// Declare the texture's buffer as mutable and inaccessible by OptiX.  See rtTextureSamplerD3D11Register.
    void registerD3D11Texture();

    /// Unregister the texture's buffer, re-enabling OptiX operations.  See rtTextureSamplerD3D9Unregister.
    void unregisterD3D9Texture();
    /// Unregister the texture's buffer, re-enabling OptiX operations.  See rtTextureSamplerD3D10Unregister.
    void unregisterD3D10Texture();
    /// Unregister the texture's buffer, re-enabling OptiX operations.  See rtTextureSamplerD3D11Unregister.
    void unregisterD3D11Texture();
    /// @}

#endif

  private:
    typedef RTtexturesampler api_t;
    virtual ~TextureSamplerObj() {}
    RTtexturesampler m_texturesampler;
    TextureSamplerObj(RTtexturesampler texturesampler) : m_texturesampler(texturesampler) {}
    friend class Handle<TextureSamplerObj>;
  };


  //----------------------------------------------------------------------------


  ///
  /// \brief Buffer wraps the OptiX C API RTbuffer opaque type and its associated function set.
  ///
  class BufferObj : public DestroyableObj {
  public:
    void destroy();
    void validate();
    Context getContext() const;

    /// @{
    /// Set the data format for the buffer.  See rtBufferSetFormat.
    void setFormat    (RTformat format);
    /// Query the data format for the buffer.  See rtBufferGetFormat.
    RTformat getFormat() const;

    /// Set the data element size for user format buffers.  See rtBufferSetElementSize.
    void setElementSize  (RTsize size_of_element);
    /// Query the data element size for user format buffers.  See rtBufferGetElementSize.
    RTsize getElementSize() const;

    /// Get the pointer to buffer memory on a specific device. See rtBufferGetDevicePointer
    void getDevicePointer( unsigned int optix_device_number, CUdeviceptr *device_pointer );

    /// Set the pointer to buffer memory on a specific device. See rtBufferSetDevicePointer
    void setDevicePointer( unsigned int optix_device_number, CUdeviceptr device_pointer );

    /// Mark the buffer dirty
    void markDirty();

    /// Set buffer dimensionality to one and buffer width to specified width.  See rtBufferSetSize1D.
    void setSize(RTsize  width);
    /// Query 1D buffer dimension.  See rtBufferGetSize1D.
    void getSize(RTsize& width) const;
    /// Set buffer dimensionality to two and buffer dimensions to specified width,height.  See rtBufferSetSize2D.
    void setSize(RTsize  width, RTsize  height);
    /// Query 2D buffer dimension.  See rtBufferGetSize2D.
    void getSize(RTsize& width, RTsize& height) const;
    /// Set buffer dimensionality to three and buffer dimensions to specified width,height,depth.
    /// See rtBufferSetSize3D.
    void setSize(RTsize  width, RTsize  height, RTsize  depth);
    /// Query 3D buffer dimension.  See rtBufferGetSize3D.
    void getSize(RTsize& width, RTsize& height, RTsize& depth) const;

    /// Set buffer dimensionality and dimensions to specified values. See rtBufferSetSizev.
    void setSize(unsigned int dimensionality, const RTsize* dims);
    /// Query dimensions of buffer.  See rtBufferGetSizev.
    void getSize(unsigned int dimensionality,       RTsize* dims) const;

    /// Query dimensionality of buffer.  See rtBufferGetDimensionality.
    unsigned int getDimensionality() const;
    /// @}

    /// @{
    /// Queries the OpenGL Buffer Object ID associated with this buffer.  See rtBufferGetGLBOId.
    unsigned int getGLBOId() const;

    /// Declare the buffer as mutable and inaccessible by OptiX.  See rtTextureSamplerGLRegister.
    void registerGLBuffer();
    /// Unregister the buffer, re-enabling OptiX operations.  See rtTextureSamplerGLUnregister.
    void unregisterGLBuffer();
    /// @}

#ifdef _WIN32

    /// @{
    /// Declare the texture's buffer as mutable and inaccessible by OptiX.  See rtBufferD3D9Register.
    void registerD3D9Buffer();
    /// Declare the texture's buffer as mutable and inaccessible by OptiX.  See rtBufferD3D10Register.
    void registerD3D10Buffer();
    /// Declare the texture's buffer as mutable and inaccessible by OptiX.  See rtBufferD3D11Register.
    void registerD3D11Buffer();

    /// Unregister the buffer, re-enabling OptiX operations.  See rtTextureSamplerD3D9Unregister.
    void unregisterD3D9Buffer();
    /// Unregister the buffer, re-enabling OptiX operations.  See rtTextureSamplerD3D10Unregister.
    void unregisterD3D10Buffer();
    /// Unregister the buffer, re-enabling OptiX operations.  See rtTextureSamplerD3D11Unregister.
    void unregisterD3D11Buffer();

    /// Queries the D3D9 resource associated with this buffer.  See rtBufferGetD3D9Resource.
    IDirect3DResource9* getD3D9Resource();
    /// Queries the D3D10 resource associated with this buffer.  See rtBufferGetD3D10Resource.
    ID3D10Resource* getD3D10Resource();
    /// Queries the D3D11 resource associated with this buffer.  See rtBufferGetD3D11Resource.
    ID3D11Resource* getD3D11Resource();
    /// @}

#endif

    /// @{
    /// Maps a buffer object for host access.  See rtBufferMap.
    void* map();
    /// Unmaps a buffer object.  See rtBufferUnmap.
    void unmap();
    /// @}

    /// Get the underlying OptiX C API RTbuffer opaque pointer.
    RTbuffer get();

  private:
    typedef RTbuffer api_t;
    virtual ~BufferObj() {}
    RTbuffer m_buffer;
    BufferObj(RTbuffer buffer) : m_buffer(buffer) {}
    friend class Handle<BufferObj>;
  };


  //----------------------------------------------------------------------------


  inline void APIObj::checkError( RTresult code ) const
  {
    if( code != RT_SUCCESS) {
      RTcontext c = this->getContext()->get();
      throw Exception::makeException( code, c );
    }
  }

  inline void APIObj::checkError( RTresult code, Context context ) const
  {
    if( code != RT_SUCCESS) {
      RTcontext c = context->get();
      throw Exception::makeException( code, c );
    }
  }

  inline void APIObj::checkErrorNoGetContext( RTresult code ) const
  {
    if( code != RT_SUCCESS) {
      throw Exception::makeException( code, 0u );
    }
  }

  inline Context ContextObj::getContext() const
  {
    return Context::take( m_context );
  }

  inline void ContextObj::checkError(RTresult code) const
  {
    if( code != RT_SUCCESS && code != RT_TIMEOUT_CALLBACK )
      throw Exception::makeException( code, m_context );
  }

  inline unsigned int ContextObj::getDeviceCount()
  {
    unsigned int count;
    if( RTresult code = rtDeviceGetDeviceCount(&count) )
      throw Exception::makeException( code, 0 );

    return count;
  }

  inline std::string ContextObj::getDeviceName(int ordinal)
  {
    const RTsize max_string_size = 256;
    char name[max_string_size];
    if( RTresult code = rtDeviceGetAttribute(ordinal, RT_DEVICE_ATTRIBUTE_NAME,
                                             max_string_size, name) )
      throw Exception::makeException( code, 0 );
    return std::string(name);
  }

  inline void ContextObj::getDeviceAttribute(int ordinal, RTdeviceattribute attrib, RTsize size, void* p)
  {
    if( RTresult code = rtDeviceGetAttribute(ordinal, attrib, size, p) )
      throw Exception::makeException( code, 0 );
  }

  inline Context ContextObj::create()
  {
    RTcontext c;
    if( RTresult code = rtContextCreate(&c) )
      throw Exception::makeException( code, 0 );

    return Context::take(c);
  }

  inline void ContextObj::destroy()
  {
    checkErrorNoGetContext( rtContextDestroy( m_context ) );
    m_context = 0;
  }

  inline void ContextObj::validate()
  {
    checkError( rtContextValidate( m_context ) );
  }

  inline Acceleration ContextObj::createAcceleration(const char* builder, const char* traverser)
  {
    RTacceleration acceleration;
    checkError( rtAccelerationCreate( m_context, &acceleration ) );
    checkError( rtAccelerationSetBuilder( acceleration, builder ) );
    checkError( rtAccelerationSetTraverser( acceleration, traverser ) );
    return Acceleration::take(acceleration);
  }


  inline Buffer ContextObj::createBuffer(unsigned int type)
  {
    RTbuffer buffer;
    checkError( rtBufferCreate( m_context, type, &buffer ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBuffer(unsigned int type, RTformat format)
  {
    RTbuffer buffer;
    checkError( rtBufferCreate( m_context, type, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBuffer(unsigned int type, RTformat format, RTsize width)
  {
    RTbuffer buffer;
    checkError( rtBufferCreate( m_context, type, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetSize1D( buffer, width ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBuffer(unsigned int type, RTformat format, RTsize width, RTsize height)
  {
    RTbuffer buffer;
    checkError( rtBufferCreate( m_context, type, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetSize2D( buffer, width, height ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBuffer(unsigned int type, RTformat format, RTsize width, RTsize height, RTsize depth)
  {
    RTbuffer buffer;
    checkError( rtBufferCreate( m_context, type, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetSize3D( buffer, width, height, depth ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBufferForCUDA(unsigned int type)
  {
    RTbuffer buffer;
    checkError( rtBufferCreateForCUDA( m_context, type, &buffer ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBufferForCUDA(unsigned int type, RTformat format)
  {
    RTbuffer buffer;
    checkError( rtBufferCreateForCUDA( m_context, type, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBufferForCUDA(unsigned int type, RTformat format, RTsize width)
  {
    RTbuffer buffer;
    checkError( rtBufferCreateForCUDA( m_context, type, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetSize1D( buffer, width ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBufferForCUDA(unsigned int type, RTformat format, RTsize width, RTsize height)
  {
    RTbuffer buffer;
    checkError( rtBufferCreateForCUDA( m_context, type, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetSize2D( buffer, width, height ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBufferForCUDA(unsigned int type, RTformat format, RTsize width, RTsize height, RTsize depth)
  {
    RTbuffer buffer;
    checkError( rtBufferCreateForCUDA( m_context, type, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetSize3D( buffer, width, height, depth ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBufferFromGLBO(unsigned int type, unsigned int vbo)
  {
    RTbuffer buffer;
    checkError( rtBufferCreateFromGLBO( m_context, type, vbo, &buffer ) );
    return Buffer::take(buffer);
  }

#ifdef _WIN32

  inline Buffer ContextObj::createBufferFromD3D9Resource(unsigned int type, IDirect3DResource9 *pResource)
  {
    RTbuffer buffer;
    checkError( rtBufferCreateFromD3D9Resource( m_context, type, pResource, &buffer ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBufferFromD3D10Resource(unsigned int type, ID3D10Resource *pResource)
  {
    RTbuffer buffer;
    checkError( rtBufferCreateFromD3D10Resource( m_context, type, pResource, &buffer ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBufferFromD3D11Resource(unsigned int type, ID3D11Resource *pResource)
  {
    RTbuffer buffer;
    checkError( rtBufferCreateFromD3D11Resource( m_context, type, pResource, &buffer ) );
    return Buffer::take(buffer);
  }

  inline TextureSampler ContextObj::createTextureSamplerFromD3D9Resource(IDirect3DResource9 *pResource)
  {
    RTtexturesampler textureSampler;
    checkError( rtTextureSamplerCreateFromD3D9Resource(m_context, pResource, &textureSampler));
    return TextureSampler::take(textureSampler);
  }

  inline TextureSampler ContextObj::createTextureSamplerFromD3D10Resource(ID3D10Resource *pResource)
  {
    RTtexturesampler textureSampler;
    checkError( rtTextureSamplerCreateFromD3D10Resource(m_context, pResource, &textureSampler));
    return TextureSampler::take(textureSampler);
  }

  inline TextureSampler ContextObj::createTextureSamplerFromD3D11Resource(ID3D11Resource *pResource)
  {
    RTtexturesampler textureSampler;
    checkError( rtTextureSamplerCreateFromD3D11Resource(m_context, pResource, &textureSampler));
    return TextureSampler::take(textureSampler);
  }

  inline void ContextObj::setD3D9Device(IDirect3DDevice9* device)
  {
    checkError( rtContextSetD3D9Device( m_context, device ) );
  }

  inline void ContextObj::setD3D10Device(ID3D10Device* device)
  {
    checkError( rtContextSetD3D10Device( m_context, device ) );
  }

  inline void ContextObj::setD3D11Device(ID3D11Device* device)
  {
    checkError( rtContextSetD3D11Device( m_context, device ) );
  }

#endif

  inline TextureSampler ContextObj::createTextureSamplerFromGLImage(unsigned int id, RTgltarget target)
  {
    RTtexturesampler textureSampler;
    checkError( rtTextureSamplerCreateFromGLImage(m_context, id, target, &textureSampler));
    return TextureSampler::take(textureSampler);
  }

  inline Geometry ContextObj::createGeometry()
  {
    RTgeometry geometry;
    checkError( rtGeometryCreate( m_context, &geometry ) );
    return Geometry::take(geometry);
  }

  inline GeometryInstance ContextObj::createGeometryInstance()
  {
    RTgeometryinstance geometryinstance;
    checkError( rtGeometryInstanceCreate( m_context, &geometryinstance ) );
    return GeometryInstance::take(geometryinstance);
  }

  template<class Iterator>
    GeometryInstance ContextObj::createGeometryInstance( Geometry geometry, Iterator matlbegin, Iterator matlend)
  {
    GeometryInstance result = createGeometryInstance();
    result->setGeometry( geometry );
    unsigned int count = 0;
    for( Iterator iter = matlbegin; iter != matlend; ++iter )
      ++count;
    result->setMaterialCount( count );
    unsigned int index = 0;
    for(Iterator iter = matlbegin; iter != matlend; ++iter, ++index )
      result->setMaterial( index, *iter );
    return result;
  }

  inline Group ContextObj::createGroup()
  {
    RTgroup group;
    checkError( rtGroupCreate( m_context, &group ) );
    return Group::take(group);
  }

  template<class Iterator>
    inline Group ContextObj::createGroup( Iterator childbegin, Iterator childend )
  {
    Group result = createGroup();
    unsigned int count = 0;
    for(Iterator iter = childbegin; iter != childend; ++iter )
      ++count;
    result->setChildCount( count );
    unsigned int index = 0;
    for(Iterator iter = childbegin; iter != childend; ++iter, ++index )
      result->setChild( index, *iter );
    return result;
  }

  inline GeometryGroup ContextObj::createGeometryGroup()
  {
    RTgeometrygroup gg;
    checkError( rtGeometryGroupCreate( m_context, &gg ) );
    return GeometryGroup::take( gg );
  }

  template<class Iterator>
  inline GeometryGroup ContextObj::createGeometryGroup( Iterator childbegin, Iterator childend )
  {
    GeometryGroup result = createGeometryGroup();
    unsigned int count = 0;
    for(Iterator iter = childbegin; iter != childend; ++iter )
      ++count;
    result->setChildCount( count );
    unsigned int index = 0;
    for(Iterator iter = childbegin; iter != childend; ++iter, ++index )
      result->setChild( index, *iter );
    return result;
  }

  inline Transform ContextObj::createTransform()
  {
    RTtransform t;
    checkError( rtTransformCreate( m_context, &t ) );
    return Transform::take( t );
  }

  inline Material ContextObj::createMaterial()
  {
    RTmaterial material;
    checkError( rtMaterialCreate( m_context, &material ) );
    return Material::take(material);
  }

  inline Program ContextObj::createProgramFromPTXFile( const std::string& filename, const std::string& program_name )
  {
    RTprogram program;
    checkError( rtProgramCreateFromPTXFile( m_context, filename.c_str(), program_name.c_str(), &program ) );
    return Program::take(program);
  }

  inline Program ContextObj::createProgramFromPTXString( const std::string& ptx, const std::string& program_name )
  {
    RTprogram program;
    checkError( rtProgramCreateFromPTXString( m_context, ptx.c_str(), program_name.c_str(), &program ) );
    return Program::take(program);
  }

  inline Selector ContextObj::createSelector()
  {
    RTselector selector;
    checkError( rtSelectorCreate( m_context, &selector ) );
    return Selector::take(selector);
  }

  inline TextureSampler ContextObj::createTextureSampler()
  {
    RTtexturesampler texturesampler;
    checkError( rtTextureSamplerCreate( m_context, &texturesampler ) );
    return TextureSampler::take(texturesampler);
  }

  inline std::string ContextObj::getErrorString( RTresult code ) const
  {
    const char* str;
    rtContextGetErrorString( m_context, code, &str);
    return std::string(str);
  }

  template<class Iterator> inline
    void ContextObj::setDevices(Iterator begin, Iterator end)
  {
    std::vector<int> devices;
    std::copy( begin, end, std::insert_iterator<std::vector<int> >( devices, devices.begin() ) );
    checkError( rtContextSetDevices( m_context, static_cast<unsigned int>(devices.size()), &devices[0]) );
  }

  inline std::vector<int> ContextObj::getEnabledDevices() const
  {
    // Initialize with the number of enabled devices
    std::vector<int> devices(getEnabledDeviceCount());
    checkError( rtContextGetDevices( m_context, &devices[0] ) );
    return devices;
  }

  inline unsigned int ContextObj::getEnabledDeviceCount() const
  {
    unsigned int num;
    checkError( rtContextGetDeviceCount( m_context, &num ) );
    return num;
  }
  
  inline int ContextObj::getMaxTextureCount() const
  {
    int tex_count;
    checkError( rtContextGetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_MAX_TEXTURE_COUNT, sizeof(tex_count), &tex_count) );
    return tex_count;
  }

  inline int ContextObj::getCPUNumThreads() const
  {
    int cpu_num_threads;
    checkError( rtContextGetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS, sizeof(cpu_num_threads), &cpu_num_threads) );
    return cpu_num_threads;
  }

  inline RTsize ContextObj::getUsedHostMemory() const
  {
    RTsize used_mem;
    checkError( rtContextGetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_USED_HOST_MEMORY, sizeof(used_mem), &used_mem) );
    return used_mem;
  }

  inline int ContextObj::getGPUPagingActive() const
  {
    int gpu_paging_active;
    checkError( rtContextGetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_GPU_PAGING_ACTIVE, sizeof(gpu_paging_active), &gpu_paging_active) );
    return gpu_paging_active;
  }

  inline int ContextObj::getGPUPagingForcedOff() const
  {
    int gpu_paging_forced_off;
    checkError( rtContextGetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_GPU_PAGING_FORCED_OFF, sizeof(gpu_paging_forced_off), &gpu_paging_forced_off) );
    return gpu_paging_forced_off;
  }

  inline RTsize ContextObj::getAvailableDeviceMemory(int ordinal) const
  {
    RTsize free_mem;
    checkError( rtContextGetAttribute( m_context,
                                       static_cast<RTcontextattribute>(RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY + ordinal),
                                       sizeof(free_mem), &free_mem) );
    return free_mem;
  }

  inline void ContextObj::setCPUNumThreads(int cpu_num_threads)
  {
    checkError( rtContextSetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS, sizeof(cpu_num_threads), &cpu_num_threads) );
  }

  inline void ContextObj::setGPUPagingForcedOff(int gpu_paging_forced_off)
  {
    checkError( rtContextSetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_GPU_PAGING_FORCED_OFF, sizeof(gpu_paging_forced_off), &gpu_paging_forced_off) );
  }

  inline void ContextObj::setStackSize(RTsize  stack_size_bytes)
  {
    checkError(rtContextSetStackSize(m_context, stack_size_bytes) );
  }

  inline RTsize ContextObj::getStackSize() const
  {
    RTsize result;
    checkError( rtContextGetStackSize( m_context, &result ) );
    return result;
  }

  inline void ContextObj::setTimeoutCallback(RTtimeoutcallback callback, double min_polling_seconds)
  {
    checkError( rtContextSetTimeoutCallback( m_context, callback, min_polling_seconds ) );
  }

  inline void ContextObj::setEntryPointCount(unsigned int  num_entry_points)
  {
    checkError( rtContextSetEntryPointCount( m_context, num_entry_points ) );
  }

  inline unsigned int ContextObj::getEntryPointCount() const
  {
    unsigned int result;
    checkError( rtContextGetEntryPointCount( m_context, &result ) );
    return result;
  }


  inline void ContextObj::setRayGenerationProgram(unsigned int entry_point_index, Program  program)
  {
    checkError( rtContextSetRayGenerationProgram( m_context, entry_point_index, program->get() ) );
  }

  inline Program ContextObj::getRayGenerationProgram(unsigned int entry_point_index) const
  {
    RTprogram result;
    checkError( rtContextGetRayGenerationProgram( m_context, entry_point_index, &result ) );
    return Program::take( result );
  }


  inline void ContextObj::setExceptionProgram(unsigned int entry_point_index, Program  program)
  {
    checkError( rtContextSetExceptionProgram( m_context, entry_point_index, program->get() ) );
  }

  inline Program ContextObj::getExceptionProgram(unsigned int entry_point_index) const
  {
    RTprogram result;
    checkError( rtContextGetExceptionProgram( m_context, entry_point_index, &result ) );
    return Program::take( result );
  }


  inline void ContextObj::setExceptionEnabled( RTexception exception, bool enabled )
  {
    checkError( rtContextSetExceptionEnabled( m_context, exception, enabled ) );
  }

  inline bool ContextObj::getExceptionEnabled( RTexception exception ) const
  {
    int enabled;
    checkError( rtContextGetExceptionEnabled( m_context, exception, &enabled ) );
    return enabled != 0;
  }


  inline void ContextObj::setRayTypeCount(unsigned int  num_ray_types)
  {
    checkError( rtContextSetRayTypeCount( m_context, num_ray_types ) );
  }

  inline unsigned int ContextObj::getRayTypeCount() const
  {
    unsigned int result;
    checkError( rtContextGetRayTypeCount( m_context, &result ) );
    return result;
  }

  inline void ContextObj::setMissProgram(unsigned int ray_type_index, Program  program)
  {
    checkError( rtContextSetMissProgram( m_context, ray_type_index, program->get() ) );
  }

  inline Program ContextObj::getMissProgram(unsigned int ray_type_index) const
  {
    RTprogram result;
    checkError( rtContextGetMissProgram( m_context, ray_type_index, &result ) );
    return Program::take( result );
  }

  inline void ContextObj::compile()
  {
    checkError( rtContextCompile( m_context ) );
  }

  inline void ContextObj::launch(unsigned int entry_point_index, RTsize image_width)
  {
    checkError( rtContextLaunch1D( m_context, entry_point_index, image_width ) );
  }

  inline void ContextObj::launch(unsigned int entry_point_index, RTsize image_width, RTsize image_height)
  {
    checkError( rtContextLaunch2D( m_context, entry_point_index, image_width, image_height ) );
  }

  inline void ContextObj::launch(unsigned int entry_point_index, RTsize image_width, RTsize image_height, RTsize image_depth)
  {
    checkError( rtContextLaunch3D( m_context, entry_point_index, image_width, image_height, image_depth ) );
  }


  inline int ContextObj::getRunningState() const
  {
    int result;
    checkError( rtContextGetRunningState( m_context, &result ) );
    return result;
  }

  inline void ContextObj::setPrintEnabled(bool enabled)
  {
    checkError( rtContextSetPrintEnabled( m_context, enabled ) );
  }

  inline bool ContextObj::getPrintEnabled() const
  {
    int enabled;
    checkError( rtContextGetPrintEnabled( m_context, &enabled ) );
    return enabled != 0;
  }

  inline void ContextObj::setPrintBufferSize(RTsize buffer_size_bytes)
  {
    checkError( rtContextSetPrintBufferSize( m_context, buffer_size_bytes ) );
  }

  inline RTsize ContextObj::getPrintBufferSize() const
  {
    RTsize result;
    checkError( rtContextGetPrintBufferSize( m_context, &result ) );
    return result;
  }

  inline void ContextObj::setPrintLaunchIndex(int x, int y, int z)
  {
    checkError( rtContextSetPrintLaunchIndex( m_context, x, y, z ) );
  }

  inline optix::int3 ContextObj::getPrintLaunchIndex() const
  {
    optix::int3 result;
    checkError( rtContextGetPrintLaunchIndex( m_context, &result.x, &result.y, &result.z ) );
    return result;
  }

  inline Variable ContextObj::declareVariable(const std::string& name)
  {
    RTvariable v;
    checkError( rtContextDeclareVariable( m_context, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline Variable ContextObj::queryVariable(const std::string& name) const
  {
    RTvariable v;
    checkError( rtContextQueryVariable( m_context, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline void ContextObj::removeVariable(Variable v)
  {
    checkError( rtContextRemoveVariable( m_context, v->get() ) );
  }

  inline unsigned int ContextObj::getVariableCount() const
  {
    unsigned int result;
    checkError( rtContextGetVariableCount( m_context, &result ) );
    return result;
  }

  inline Variable ContextObj::getVariable(unsigned int index) const
  {
    RTvariable v;
    checkError( rtContextGetVariable( m_context, index, &v ) );
    return Variable::take( v );
  }


  inline RTcontext ContextObj::get()
  {
    return m_context;
  }

  inline void ProgramObj::destroy()
  {
    Context context = getContext();
    checkError( rtProgramDestroy( m_program ), context );
    m_program = 0;
  }

  inline void ProgramObj::validate()
  {
    checkError( rtProgramValidate( m_program ) );
  }

  inline Context ProgramObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtProgramGetContext( m_program, &c ) );
    return Context::take( c );
  }

  inline Variable ProgramObj::declareVariable(const std::string& name)
  {
    RTvariable v;
    checkError( rtProgramDeclareVariable( m_program, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline Variable ProgramObj::queryVariable(const std::string& name) const
  {
    RTvariable v;
    checkError( rtProgramQueryVariable( m_program, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline void ProgramObj::removeVariable(Variable v)
  {
    checkError( rtProgramRemoveVariable( m_program, v->get() ) );
  }

  inline unsigned int ProgramObj::getVariableCount() const
  {
    unsigned int result;
    checkError( rtProgramGetVariableCount( m_program, &result ) );
    return result;
  }

  inline Variable ProgramObj::getVariable(unsigned int index) const
  {
    RTvariable v;
    checkError( rtProgramGetVariable( m_program, index, &v ) );
    return Variable::take(v);
  }

  inline RTprogram ProgramObj::get()
  {
    return m_program;
  }

  inline void GroupObj::destroy()
  {
    Context context = getContext();
    checkError( rtGroupDestroy( m_group ), context );
    m_group = 0;
  }

  inline void GroupObj::validate()
  {
    checkError( rtGroupValidate( m_group ) );
  }

  inline Context GroupObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtGroupGetContext( m_group, &c) );
    return Context::take(c);
  }

  inline void SelectorObj::destroy()
  {
    Context context = getContext();
    checkError( rtSelectorDestroy( m_selector ), context );
    m_selector = 0;
  }

  inline void SelectorObj::validate()
  {
    checkError( rtSelectorValidate( m_selector ) );
  }

  inline Context SelectorObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtSelectorGetContext( m_selector, &c ) );
    return Context::take( c );
  }

  inline void SelectorObj::setVisitProgram(Program program)
  {
    checkError( rtSelectorSetVisitProgram( m_selector, program->get() ) );
  }

  inline Program SelectorObj::getVisitProgram() const
  {
    RTprogram result;
    checkError( rtSelectorGetVisitProgram( m_selector, &result ) );
    return Program::take( result );
  }

  inline void SelectorObj::setChildCount(unsigned int count)
  {
    checkError( rtSelectorSetChildCount( m_selector, count) );
  }

  inline unsigned int SelectorObj::getChildCount() const
  {
    unsigned int result;
    checkError( rtSelectorGetChildCount( m_selector, &result ) );
    return result;
  }

  template< typename T >
  inline void SelectorObj::setChild(unsigned int index, T child)
  {
    checkError( rtSelectorSetChild( m_selector, index, child->get() ) );
  }

  template< typename T >
  inline T SelectorObj::getChild(unsigned int index) const
  {
    RTobject result;
    checkError( rtSelectorGetChild( m_selector, index, &result ) );
    return T::take( result );
  }

  inline Variable SelectorObj::declareVariable(const std::string& name)
  {
    RTvariable v;
    checkError( rtSelectorDeclareVariable( m_selector, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline Variable SelectorObj::queryVariable(const std::string& name) const
  {
    RTvariable v;
    checkError( rtSelectorQueryVariable( m_selector, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline void SelectorObj::removeVariable(Variable v)
  {
    checkError( rtSelectorRemoveVariable( m_selector, v->get() ) );
  }

  inline unsigned int SelectorObj::getVariableCount() const
  {
    unsigned int result;
    checkError( rtSelectorGetVariableCount( m_selector, &result ) );
    return result;
  }

  inline Variable SelectorObj::getVariable(unsigned int index) const
  {
    RTvariable v;
    checkError( rtSelectorGetVariable( m_selector, index, &v ) );
    return Variable::take( v );
  }

  inline RTselector SelectorObj::get()
  {
    return m_selector;
  }

  inline void GroupObj::setAcceleration(Acceleration acceleration)
  {
    checkError( rtGroupSetAcceleration( m_group, acceleration->get() ) );
  }

  inline Acceleration GroupObj::getAcceleration() const
  {
    RTacceleration result;
    checkError( rtGroupGetAcceleration( m_group, &result ) );
    return Acceleration::take( result );
  }

  inline void GroupObj::setChildCount(unsigned int  count)
  {
    checkError( rtGroupSetChildCount( m_group, count ) );
  }

  inline unsigned int GroupObj::getChildCount() const
  {
    unsigned int result;
    checkError( rtGroupGetChildCount( m_group, &result ) );
    return result;
  }

  template< typename T >
  inline void GroupObj::setChild(unsigned int index, T child)
  {
    checkError( rtGroupSetChild( m_group, index, child->get() ) );
  }

  template< typename T >
  inline T GroupObj::getChild(unsigned int index) const
  {
    RTobject result;
    checkError( rtGroupGetChild( m_group, index, &result) );
    return T::take( result );
  }

  inline RTgroup GroupObj::get()
  {
    return m_group;
  }

  inline void GeometryGroupObj::destroy()
  {
    Context context = getContext();
    checkError( rtGeometryGroupDestroy( m_geometrygroup ), context );
    m_geometrygroup = 0;
  }

  inline void GeometryGroupObj::validate()
  {
    checkError( rtGeometryGroupValidate( m_geometrygroup ) );
  }

  inline Context GeometryGroupObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtGeometryGroupGetContext( m_geometrygroup, &c) );
    return Context::take(c);
  }

  inline void GeometryGroupObj::setAcceleration(Acceleration acceleration)
  {
    checkError( rtGeometryGroupSetAcceleration( m_geometrygroup, acceleration->get() ) );
  }

  inline Acceleration GeometryGroupObj::getAcceleration() const
  {
    RTacceleration result;
    checkError( rtGeometryGroupGetAcceleration( m_geometrygroup, &result ) );
    return Acceleration::take( result );
  }

  inline void GeometryGroupObj::setChildCount(unsigned int  count)
  {
    checkError( rtGeometryGroupSetChildCount( m_geometrygroup, count ) );
  }

  inline unsigned int GeometryGroupObj::getChildCount() const
  {
    unsigned int result;
    checkError( rtGeometryGroupGetChildCount( m_geometrygroup, &result ) );
    return result;
  }

  inline void GeometryGroupObj::setChild(unsigned int index, GeometryInstance child)
  {
    checkError( rtGeometryGroupSetChild( m_geometrygroup, index, child->get() ) );
  }

  inline GeometryInstance GeometryGroupObj::getChild(unsigned int index) const
  {
    RTgeometryinstance result;
    checkError( rtGeometryGroupGetChild( m_geometrygroup, index, &result) );
    return GeometryInstance::take( result );
  }

  inline RTgeometrygroup GeometryGroupObj::get()
  {
    return m_geometrygroup;
  }

  inline void TransformObj::destroy()
  {
    Context context = getContext();
    checkError( rtTransformDestroy( m_transform ), context );
    m_transform = 0;
  }

  inline void TransformObj::validate()
  {
    checkError( rtTransformValidate( m_transform ) );
  }

  inline Context TransformObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtTransformGetContext( m_transform, &c) );
    return Context::take(c);
  }

  template< typename T >
  inline void TransformObj::setChild(T child)
  {
    checkError( rtTransformSetChild( m_transform, child->get() ) );
  }

  template< typename T >
  inline T TransformObj::getChild() const
  {
    RTobject result;
    checkError( rtTransformGetChild( m_transform, &result) );
    return T::take( result );
  }

  inline void TransformObj::setMatrix(bool transpose, const float* matrix, const float* inverse_matrix)
  {
    rtTransformSetMatrix( m_transform, transpose, matrix, inverse_matrix );
  }

  inline void TransformObj::getMatrix(bool transpose, float* matrix, float* inverse_matrix) const
  {
    rtTransformGetMatrix( m_transform, transpose, matrix, inverse_matrix );
  }

  inline RTtransform TransformObj::get()
  {
    return m_transform;
  }

  inline void AccelerationObj::destroy()
  {
    Context context = getContext();
    checkError( rtAccelerationDestroy(m_acceleration), context );
    m_acceleration = 0;
  }

  inline void AccelerationObj::validate()
  {
    checkError( rtAccelerationValidate(m_acceleration) );
  }

  inline Context AccelerationObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtAccelerationGetContext(m_acceleration, &c ) );
    return Context::take( c );
  }

  inline void AccelerationObj::markDirty()
  {
    checkError( rtAccelerationMarkDirty(m_acceleration) );
  }

  inline bool AccelerationObj::isDirty() const
  {
    int dirty;
    checkError( rtAccelerationIsDirty(m_acceleration,&dirty) );
    return dirty != 0;
  }

  inline void AccelerationObj::setProperty( const std::string& name, const std::string& value )
  {
    checkError( rtAccelerationSetProperty(m_acceleration, name.c_str(), value.c_str() ) );
  }

  inline std::string AccelerationObj::getProperty( const std::string& name ) const
  {
    const char* s;
    checkError( rtAccelerationGetProperty(m_acceleration, name.c_str(), &s ) );
    return std::string( s );
  }

  inline void AccelerationObj::setBuilder(const std::string& builder)
  {
    checkError( rtAccelerationSetBuilder(m_acceleration, builder.c_str() ) );
  }

  inline std::string AccelerationObj::getBuilder() const
  {
    const char* s;
    checkError( rtAccelerationGetBuilder(m_acceleration, &s ) );
    return std::string( s );
  }

  inline void AccelerationObj::setTraverser(const std::string& traverser)
  {
    checkError( rtAccelerationSetTraverser(m_acceleration, traverser.c_str() ) );
  }

  inline std::string AccelerationObj::getTraverser() const
  {
    const char* s;
    checkError( rtAccelerationGetTraverser(m_acceleration, &s ) );
    return std::string( s );
  }

  inline RTsize AccelerationObj::getDataSize() const
  {
    RTsize sz;
    checkError( rtAccelerationGetDataSize(m_acceleration, &sz) );
    return sz;
  }

  inline void AccelerationObj::getData( void* data ) const
  {
    checkError( rtAccelerationGetData(m_acceleration,data) );
  }

  inline void AccelerationObj::setData( const void* data, RTsize size )
  {
    checkError( rtAccelerationSetData(m_acceleration,data,size) );
  }

  inline RTacceleration AccelerationObj::get()
  {
    return m_acceleration;
  }

  inline void GeometryInstanceObj::destroy()
  {
    Context context = getContext();
    checkError( rtGeometryInstanceDestroy( m_geometryinstance ), context );
    m_geometryinstance = 0;
  }

  inline void GeometryInstanceObj::validate()
  {
    checkError( rtGeometryInstanceValidate( m_geometryinstance ) );
  }

  inline Context GeometryInstanceObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtGeometryInstanceGetContext( m_geometryinstance, &c ) );
    return Context::take( c );
  }

  inline void GeometryInstanceObj::setGeometry(Geometry geometry)
  {
    checkError( rtGeometryInstanceSetGeometry( m_geometryinstance, geometry->get() ) );
  }

  inline Geometry GeometryInstanceObj::getGeometry() const
  {
    RTgeometry result;
    checkError( rtGeometryInstanceGetGeometry( m_geometryinstance, &result ) );
    return Geometry::take( result );
  }

  inline void GeometryInstanceObj::setMaterialCount(unsigned int  count)
  {
    checkError( rtGeometryInstanceSetMaterialCount( m_geometryinstance, count ) );
  }

  inline unsigned int GeometryInstanceObj::getMaterialCount() const
  {
    unsigned int result;
    checkError( rtGeometryInstanceGetMaterialCount( m_geometryinstance, &result ) );
    return result;
  }

  inline void GeometryInstanceObj::setMaterial(unsigned int idx, Material  material)
  {
    checkError( rtGeometryInstanceSetMaterial( m_geometryinstance, idx, material->get()) );
  }

  inline Material GeometryInstanceObj::getMaterial(unsigned int idx) const
  {
    RTmaterial result;
    checkError( rtGeometryInstanceGetMaterial( m_geometryinstance, idx, &result ) );
    return Material::take( result );
  }

  // Adds the material and returns the index to the added material.
  inline unsigned int GeometryInstanceObj::addMaterial(Material material)
  {
    unsigned int old_count = getMaterialCount();
    setMaterialCount(old_count+1);
    setMaterial(old_count, material);
    return old_count;
  }

  inline Variable GeometryInstanceObj::declareVariable(const std::string& name)
  {
    RTvariable v;
    checkError( rtGeometryInstanceDeclareVariable( m_geometryinstance, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline Variable GeometryInstanceObj::queryVariable(const std::string& name) const
  {
    RTvariable v;
    checkError( rtGeometryInstanceQueryVariable( m_geometryinstance, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline void GeometryInstanceObj::removeVariable(Variable v)
  {
    checkError( rtGeometryInstanceRemoveVariable( m_geometryinstance, v->get() ) );
  }

  inline unsigned int GeometryInstanceObj::getVariableCount() const
  {
    unsigned int result;
    checkError( rtGeometryInstanceGetVariableCount( m_geometryinstance, &result ) );
    return result;
  }

  inline Variable GeometryInstanceObj::getVariable(unsigned int index) const
  {
    RTvariable v;
    checkError( rtGeometryInstanceGetVariable( m_geometryinstance, index, &v ) );
    return Variable::take( v );
  }

  inline RTgeometryinstance GeometryInstanceObj::get()
  {
    return m_geometryinstance;
  }

  inline void GeometryObj::destroy()
  {
    Context context = getContext();
    checkError( rtGeometryDestroy( m_geometry ), context );
    m_geometry = 0;
  }

  inline void GeometryObj::validate()
  {
    checkError( rtGeometryValidate( m_geometry ) );
  }

  inline Context GeometryObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtGeometryGetContext( m_geometry, &c ) );
    return Context::take( c );
  }

  inline void GeometryObj::setPrimitiveCount(unsigned int  num_primitives)
  {
    checkError( rtGeometrySetPrimitiveCount( m_geometry, num_primitives ) );
  }

  inline unsigned int GeometryObj::getPrimitiveCount() const
  {
    unsigned int result;
    checkError( rtGeometryGetPrimitiveCount( m_geometry, &result ) );
    return result;
  }

  inline void GeometryObj::setBoundingBoxProgram(Program  program)
  {
    checkError( rtGeometrySetBoundingBoxProgram( m_geometry, program->get() ) );
  }

  inline Program GeometryObj::getBoundingBoxProgram() const
  {
    RTprogram result;
    checkError( rtGeometryGetBoundingBoxProgram( m_geometry, &result ) );
    return Program::take( result );
  }

  inline void GeometryObj::setIntersectionProgram(Program  program)
  {
    checkError( rtGeometrySetIntersectionProgram( m_geometry, program->get() ) );
  }

  inline Program GeometryObj::getIntersectionProgram() const
  {
    RTprogram result;
    checkError( rtGeometryGetIntersectionProgram( m_geometry, &result ) );
    return Program::take( result );
  }

  inline Variable GeometryObj::declareVariable(const std::string& name)
  {
    RTvariable v;
    checkError( rtGeometryDeclareVariable( m_geometry, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline Variable GeometryObj::queryVariable(const std::string& name) const
  {
    RTvariable v;
    checkError( rtGeometryQueryVariable( m_geometry, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline void GeometryObj::removeVariable(Variable v)
  {
    checkError( rtGeometryRemoveVariable( m_geometry, v->get() ) );
  }

  inline unsigned int GeometryObj::getVariableCount() const
  {
    unsigned int result;
    checkError( rtGeometryGetVariableCount( m_geometry, &result ) );
    return result;
  }

  inline Variable GeometryObj::getVariable(unsigned int index) const
  {
    RTvariable v;
    checkError( rtGeometryGetVariable( m_geometry, index, &v ) );
    return Variable::take( v );
  }

  inline void GeometryObj::markDirty()
  {
    checkError( rtGeometryMarkDirty(m_geometry) );
  }

  inline bool GeometryObj::isDirty() const
  {
    int dirty;
    checkError( rtGeometryIsDirty(m_geometry,&dirty) );
    return dirty != 0;
  }

  inline RTgeometry GeometryObj::get()
  {
    return m_geometry;
  }

  inline void MaterialObj::destroy()
  {
    Context context = getContext();
    checkError( rtMaterialDestroy( m_material ), context );
    m_material = 0;
  }

  inline void MaterialObj::validate()
  {
    checkError( rtMaterialValidate( m_material ) );
  }

  inline Context MaterialObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtMaterialGetContext( m_material, &c ) );
    return Context::take( c );
  }

  inline void MaterialObj::setClosestHitProgram(unsigned int ray_type_index, Program  program)
  {
    checkError( rtMaterialSetClosestHitProgram( m_material, ray_type_index, program->get() ) );
  }

  inline Program MaterialObj::getClosestHitProgram(unsigned int ray_type_index) const
  {
    RTprogram result;
    checkError( rtMaterialGetClosestHitProgram( m_material, ray_type_index, &result ) );
    return Program::take( result );
  }

  inline void MaterialObj::setAnyHitProgram(unsigned int ray_type_index, Program  program)
  {
    checkError( rtMaterialSetAnyHitProgram( m_material, ray_type_index, program->get() ) );
  }

  inline Program MaterialObj::getAnyHitProgram(unsigned int ray_type_index) const
  {
    RTprogram result;
    checkError( rtMaterialGetAnyHitProgram( m_material, ray_type_index, &result ) );
    return Program::take( result );
  }

  inline Variable MaterialObj::declareVariable(const std::string& name)
  {
    RTvariable v;
    checkError( rtMaterialDeclareVariable( m_material, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline Variable MaterialObj::queryVariable(const std::string& name) const
  {
    RTvariable v;
    checkError( rtMaterialQueryVariable( m_material, name.c_str(), &v) );
    return Variable::take( v );
  }

  inline void MaterialObj::removeVariable(Variable v)
  {
    checkError( rtMaterialRemoveVariable( m_material, v->get() ) );
  }

  inline unsigned int MaterialObj::getVariableCount() const
  {
    unsigned int result;
    checkError( rtMaterialGetVariableCount( m_material, &result ) );
    return result;
  }

  inline Variable MaterialObj::getVariable(unsigned int index) const
  {
    RTvariable v;
    checkError( rtMaterialGetVariable( m_material, index, &v) );
    return Variable::take( v );
  }

  inline RTmaterial MaterialObj::get()
  {
    return m_material;
  }

  inline void TextureSamplerObj::destroy()
  {
    Context context = getContext();
    checkError( rtTextureSamplerDestroy( m_texturesampler ), context );
    m_texturesampler = 0;
  }

  inline void TextureSamplerObj::validate()
  {
    checkError( rtTextureSamplerValidate( m_texturesampler ) );
  }

  inline Context TextureSamplerObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtTextureSamplerGetContext( m_texturesampler, &c ) );
    return Context::take( c );
  }

  inline void TextureSamplerObj::setMipLevelCount(unsigned int  num_mip_levels)
  {
    checkError( rtTextureSamplerSetMipLevelCount(m_texturesampler, num_mip_levels ) );
  }

  inline unsigned int TextureSamplerObj::getMipLevelCount() const
  {
    unsigned int result;
    checkError( rtTextureSamplerGetMipLevelCount( m_texturesampler, &result ) );
    return result;
  }

  inline void TextureSamplerObj::setArraySize(unsigned int  num_textures_in_array)
  {
    checkError( rtTextureSamplerSetArraySize( m_texturesampler, num_textures_in_array ) );
  }

  inline unsigned int TextureSamplerObj::getArraySize() const
  {
    unsigned int result;
    checkError( rtTextureSamplerGetArraySize( m_texturesampler, &result ) );
    return result;
  }

  inline void TextureSamplerObj::setWrapMode(unsigned int dim, RTwrapmode wrapmode)
  {
    checkError( rtTextureSamplerSetWrapMode( m_texturesampler, dim, wrapmode ) );
  }

  inline RTwrapmode TextureSamplerObj::getWrapMode(unsigned int dim) const
  {
    RTwrapmode wrapmode;
    checkError( rtTextureSamplerGetWrapMode( m_texturesampler, dim, &wrapmode ) );
    return wrapmode;
  }

  inline void TextureSamplerObj::setFilteringModes(RTfiltermode  minification, RTfiltermode  magnification, RTfiltermode  mipmapping)
  {
    checkError( rtTextureSamplerSetFilteringModes( m_texturesampler, minification, magnification, mipmapping ) );
  }

  inline void TextureSamplerObj::getFilteringModes(RTfiltermode& minification, RTfiltermode& magnification, RTfiltermode& mipmapping) const
  {
    checkError( rtTextureSamplerGetFilteringModes( m_texturesampler, &minification, &magnification, &mipmapping ) );
  }

  inline void TextureSamplerObj::setMaxAnisotropy(float value)
  {
    checkError( rtTextureSamplerSetMaxAnisotropy(m_texturesampler, value ) );
  }

  inline float TextureSamplerObj::getMaxAnisotropy() const
  {
    float result;
    checkError( rtTextureSamplerGetMaxAnisotropy( m_texturesampler, &result) );
    return result;
  }

  inline int TextureSamplerObj::getId() const
  {
    int result;
    checkError( rtTextureSamplerGetId( m_texturesampler, &result) );
    return result;
  }

  inline void TextureSamplerObj::setReadMode(RTtexturereadmode  readmode)
  {
    checkError( rtTextureSamplerSetReadMode( m_texturesampler, readmode ) );
  }

  inline RTtexturereadmode TextureSamplerObj::getReadMode() const
  {
    RTtexturereadmode result;
    checkError( rtTextureSamplerGetReadMode( m_texturesampler, &result) );
    return result;
  }

  inline void TextureSamplerObj::setIndexingMode(RTtextureindexmode  indexmode)
  {
    checkError( rtTextureSamplerSetIndexingMode( m_texturesampler, indexmode ) );
  }

  inline RTtextureindexmode TextureSamplerObj::getIndexingMode() const
  {
    RTtextureindexmode result;
    checkError( rtTextureSamplerGetIndexingMode( m_texturesampler, &result ) );
    return result;
  }

  inline void TextureSamplerObj::setBuffer(unsigned int texture_array_idx, unsigned int mip_level, Buffer buffer)
  {
    checkError( rtTextureSamplerSetBuffer( m_texturesampler, texture_array_idx, mip_level, buffer->get() ) );
  }

  inline Buffer TextureSamplerObj::getBuffer(unsigned int texture_array_idx, unsigned int mip_level) const
  {
    RTbuffer result;
    checkError( rtTextureSamplerGetBuffer(m_texturesampler, texture_array_idx, mip_level, &result ) );
    return Buffer::take(result);
  }

  inline RTtexturesampler TextureSamplerObj::get()
  {
    return m_texturesampler;
  }

  inline void TextureSamplerObj::registerGLTexture()
  {
    checkError( rtTextureSamplerGLRegister( m_texturesampler ) );
  }

  inline void TextureSamplerObj::unregisterGLTexture()
  {
    checkError( rtTextureSamplerGLUnregister( m_texturesampler ) );
  }

#ifdef _WIN32

  inline void TextureSamplerObj::registerD3D9Texture()
  {
    checkError( rtTextureSamplerD3D9Register( m_texturesampler ) );
  }

  inline void TextureSamplerObj::registerD3D10Texture()
  {
    checkError( rtTextureSamplerD3D10Register( m_texturesampler ) );
  }

  inline void TextureSamplerObj::registerD3D11Texture()
  {
    checkError( rtTextureSamplerD3D11Register( m_texturesampler ) );
  }

  inline void TextureSamplerObj::unregisterD3D9Texture()
  {
    checkError( rtTextureSamplerD3D9Unregister( m_texturesampler ) );
  }

  inline void TextureSamplerObj::unregisterD3D10Texture()
  {
    checkError( rtTextureSamplerD3D10Unregister( m_texturesampler ) );
  }

  inline void TextureSamplerObj::unregisterD3D11Texture()
  {
    checkError( rtTextureSamplerD3D11Unregister( m_texturesampler ) );
  }

#endif

  inline void BufferObj::destroy()
  {
    Context context = getContext();
    checkError( rtBufferDestroy( m_buffer ), context );
    m_buffer = 0;
  }

  inline void BufferObj::validate()
  {
    checkError( rtBufferValidate( m_buffer ) );
  }

  inline Context BufferObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtBufferGetContext( m_buffer, &c ) );
    return Context::take( c );
  }

  inline void BufferObj::setFormat(RTformat format)
  {
    checkError( rtBufferSetFormat( m_buffer, format ) );
  }

  inline RTformat BufferObj::getFormat() const
  {
    RTformat result;
    checkError( rtBufferGetFormat( m_buffer, &result ) );
    return result;
  }

  inline void BufferObj::setElementSize(RTsize size_of_element)
  {
    checkError( rtBufferSetElementSize ( m_buffer, size_of_element ) );
  }

  inline RTsize BufferObj::getElementSize() const
  {
    RTsize result;
    checkError( rtBufferGetElementSize ( m_buffer, &result) );
    return result;
  }

  inline void BufferObj::getDevicePointer(unsigned int optix_device_number, CUdeviceptr *device_pointer)
  {
    checkError( rtBufferGetDevicePointer( m_buffer, optix_device_number, (void**)device_pointer ) );
  }

  inline void BufferObj::setDevicePointer(unsigned int optix_device_number, CUdeviceptr device_pointer)
  {
    checkError( rtBufferSetDevicePointer( m_buffer, optix_device_number, device_pointer ) );
  }

  inline void BufferObj::markDirty()
  {
    checkError( rtBufferMarkDirty( m_buffer ) );
  }

  inline void BufferObj::setSize(RTsize width)
  {
    checkError( rtBufferSetSize1D( m_buffer, width ) );
  }

  inline void BufferObj::getSize(RTsize& width) const
  {
    checkError( rtBufferGetSize1D( m_buffer, &width ) );
  }

  inline void BufferObj::setSize(RTsize width, RTsize height)
  {
    checkError( rtBufferSetSize2D( m_buffer, width, height ) );
  }

  inline void BufferObj::getSize(RTsize& width, RTsize& height) const
  {
    checkError( rtBufferGetSize2D( m_buffer, &width, &height ) );
  }

  inline void BufferObj::setSize(RTsize width, RTsize height, RTsize depth)
  {
    checkError( rtBufferSetSize3D( m_buffer, width, height, depth ) );
  }

  inline void BufferObj::getSize(RTsize& width, RTsize& height, RTsize& depth) const
  {
    checkError( rtBufferGetSize3D( m_buffer, &width, &height, &depth ) );
  }

  inline void BufferObj::setSize(unsigned int dimensionality, const RTsize* dims)
  {
    checkError( rtBufferSetSizev( m_buffer, dimensionality, dims ) );
  }

  inline void BufferObj::getSize(unsigned int dimensionality, RTsize* dims) const
  {
    checkError( rtBufferGetSizev( m_buffer, dimensionality, dims ) );
  }

  inline unsigned int BufferObj::getDimensionality() const
  {
    unsigned int result;
    checkError( rtBufferGetDimensionality( m_buffer, &result ) );
    return result;
  }

  inline unsigned int BufferObj::getGLBOId() const
  {
    unsigned int result;
    checkError( rtBufferGetGLBOId( m_buffer, &result ) );
    return result;
  }

  inline void BufferObj::registerGLBuffer()
  {
    checkError( rtBufferGLRegister( m_buffer ) );
  }

  inline void BufferObj::unregisterGLBuffer()
  {
    checkError( rtBufferGLUnregister( m_buffer ) );
  }

#ifdef _WIN32

  inline void BufferObj::registerD3D9Buffer()
  {
    checkError( rtBufferD3D9Register( m_buffer ) );
  }

  inline void BufferObj::registerD3D10Buffer()
  {
    checkError( rtBufferD3D10Register( m_buffer ) );
  }

  inline void BufferObj::registerD3D11Buffer()
  {
    checkError( rtBufferD3D11Register( m_buffer ) );
  }

  inline void BufferObj::unregisterD3D9Buffer()
  {
    checkError( rtBufferD3D9Unregister( m_buffer ) );
  }

  inline void BufferObj::unregisterD3D10Buffer()
  {
    checkError( rtBufferD3D10Unregister( m_buffer ) );
  }

  inline void BufferObj::unregisterD3D11Buffer()
  {
    checkError( rtBufferD3D11Unregister( m_buffer ) );
  }

  inline IDirect3DResource9* BufferObj::getD3D9Resource()
  {
    IDirect3DResource9* result = NULL;
    checkError( rtBufferGetD3D9Resource( m_buffer, &result ) );
    return result;
  }
  
  inline ID3D10Resource* BufferObj::getD3D10Resource()
  {
    ID3D10Resource* result = NULL;
    checkError( rtBufferGetD3D10Resource( m_buffer, &result ) );
    return result;
  }
  
  inline ID3D11Resource* BufferObj::getD3D11Resource()
  {
    ID3D11Resource* result = NULL;
    checkError( rtBufferGetD3D11Resource( m_buffer, &result ) );
    return result;
  }

#endif

  inline void* BufferObj::map()
  {
    void* result;
    checkError( rtBufferMap( m_buffer, &result ) );
    return result;
  }

  inline void BufferObj::unmap()
  {
    checkError( rtBufferUnmap( m_buffer ) );
  }


  inline RTbuffer BufferObj::get()
  {
    return m_buffer;
  }

  inline Context VariableObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtVariableGetContext( m_variable, &c ) );
    return Context::take( c );
  }

  inline void VariableObj::setUint(unsigned int u1)
  {
    checkError( rtVariableSet1ui( m_variable, u1 ) );
  }

  inline void VariableObj::setUint(unsigned int u1, unsigned int u2)
  {
    checkError( rtVariableSet2ui( m_variable, u1, u2 ) );
  }

  inline void VariableObj::setUint(unsigned int u1, unsigned int u2, unsigned int u3)
  {
    checkError( rtVariableSet3ui( m_variable, u1, u2, u3 ) );
  }

  inline void VariableObj::setUint(unsigned int u1, unsigned int u2, unsigned int u3, unsigned int u4)
  {
    checkError( rtVariableSet4ui( m_variable, u1, u2, u3, u4 ) );
  }

  inline void VariableObj::setUint(optix::uint2 u)
  {
    checkError( rtVariableSet2uiv( m_variable, &u.x ) );
  }

  inline void VariableObj::setUint(optix::uint3 u)
  {
    checkError( rtVariableSet3uiv( m_variable, &u.x ) );
  }

  inline void VariableObj::setUint(optix::uint4 u)
  {
    checkError( rtVariableSet4uiv( m_variable, &u.x ) );
  }

  inline void VariableObj::set1uiv(const unsigned int* u)
  {
    checkError( rtVariableSet1uiv( m_variable, u ) );
  }

  inline void VariableObj::set2uiv(const unsigned int* u)
  {
    checkError( rtVariableSet2uiv( m_variable, u ) );
  }

  inline void VariableObj::set3uiv(const unsigned int* u)
  {
    checkError( rtVariableSet3uiv( m_variable, u ) );
  }

  inline void VariableObj::set4uiv(const unsigned int* u)
  {
    checkError( rtVariableSet4uiv( m_variable, u ) );
  }

  inline void VariableObj::setMatrix2x2fv(bool transpose, const float* m)
  {
    checkError( rtVariableSetMatrix2x2fv( m_variable, (int)transpose, m ) );
  }

  inline void VariableObj::setMatrix2x3fv(bool transpose, const float* m)
  {
    checkError( rtVariableSetMatrix2x3fv( m_variable, (int)transpose, m ) );
  }

  inline void VariableObj::setMatrix2x4fv(bool transpose, const float* m)
  {
    checkError( rtVariableSetMatrix2x4fv( m_variable, (int)transpose, m ) );
  }

  inline void VariableObj::setMatrix3x2fv(bool transpose, const float* m)
  {
    checkError( rtVariableSetMatrix3x2fv( m_variable, (int)transpose, m ) );
  }

  inline void VariableObj::setMatrix3x3fv(bool transpose, const float* m)
  {
    checkError( rtVariableSetMatrix3x3fv( m_variable, (int)transpose, m ) );
  }

  inline void VariableObj::setMatrix3x4fv(bool transpose, const float* m)
  {
    checkError( rtVariableSetMatrix3x4fv( m_variable, (int)transpose, m ) );
  }

  inline void VariableObj::setMatrix4x2fv(bool transpose, const float* m)
  {
    checkError( rtVariableSetMatrix4x2fv( m_variable, (int)transpose, m ) );
  }

  inline void VariableObj::setMatrix4x3fv(bool transpose, const float* m)
  {
    checkError( rtVariableSetMatrix4x3fv( m_variable, (int)transpose, m ) );
  }

  inline void VariableObj::setMatrix4x4fv(bool transpose, const float* m)
  {
    checkError( rtVariableSetMatrix4x4fv( m_variable, (int)transpose, m ) );
  }

  inline void VariableObj::setFloat(float f1)
  {
    checkError( rtVariableSet1f( m_variable, f1 ) );
  }

  inline void VariableObj::setFloat(optix::float2 f)
  {
    checkError( rtVariableSet2fv( m_variable, &f.x ) );
  }

  inline void VariableObj::setFloat(float f1, float f2)
  {
    checkError( rtVariableSet2f( m_variable, f1, f2 ) );
  }

  inline void VariableObj::setFloat(optix::float3 f)
  {
    checkError( rtVariableSet3fv( m_variable, &f.x ) );
  }

  inline void VariableObj::setFloat(float f1, float f2, float f3)
  {
    checkError( rtVariableSet3f( m_variable, f1, f2, f3 ) );
  }

  inline void VariableObj::setFloat(optix::float4 f)
  {
    checkError( rtVariableSet4fv( m_variable, &f.x ) );
  }

  inline void VariableObj::setFloat(float f1, float f2, float f3, float f4)
  {
    checkError( rtVariableSet4f( m_variable, f1, f2, f3, f4 ) );
  }

  inline void VariableObj::set1fv(const float* f)
  {
    checkError( rtVariableSet1fv( m_variable, f ) );
  }

  inline void VariableObj::set2fv(const float* f)
  {
    checkError( rtVariableSet2fv( m_variable, f ) );
  }

  inline void VariableObj::set3fv(const float* f)
  {
    checkError( rtVariableSet3fv( m_variable, f ) );
  }

  inline void VariableObj::set4fv(const float* f)
  {
    checkError( rtVariableSet4fv( m_variable, f ) );
  }

  ///////
  inline void VariableObj::setInt(int i1)
  {
    checkError( rtVariableSet1i( m_variable, i1 ) );
  }

  inline void VariableObj::setInt(optix::int2 i)
  {
    checkError( rtVariableSet2iv( m_variable, &i.x ) );
  }

  inline void VariableObj::setInt(int i1, int i2)
  {
    checkError( rtVariableSet2i( m_variable, i1, i2 ) );
  }

  inline void VariableObj::setInt(optix::int3 i)
  {
    checkError( rtVariableSet3iv( m_variable, &i.x ) );
  }

  inline void VariableObj::setInt(int i1, int i2, int i3)
  {
    checkError( rtVariableSet3i( m_variable, i1, i2, i3 ) );
  }

  inline void VariableObj::setInt(optix::int4 i)
  {
    checkError( rtVariableSet4iv( m_variable, &i.x ) );
  }

  inline void VariableObj::setInt(int i1, int i2, int i3, int i4)
  {
    checkError( rtVariableSet4i( m_variable, i1, i2, i3, i4 ) );
  }

  inline void VariableObj::set1iv( const int* i )
  {
    checkError( rtVariableSet1iv( m_variable, i ) );
  }

  inline void VariableObj::set2iv( const int* i )
  {
    checkError( rtVariableSet2iv( m_variable, i ) );
  }

  inline void VariableObj::set3iv( const int* i )
  {
    checkError( rtVariableSet3iv( m_variable, i ) );
  }

  inline void VariableObj::set4iv( const int* i )
  {
    checkError( rtVariableSet4iv( m_variable, i ) );
  }

  inline float VariableObj::getFloat() const
  {
    float f;
    checkError( rtVariableGet1f( m_variable, &f ) );
    return f;
  }

  inline unsigned int VariableObj::getUint() const
  {
    unsigned int i;
    checkError( rtVariableGet1ui( m_variable, &i ) );
    return i;
  }

  inline int VariableObj::getInt() const
  {
    int i;
    checkError( rtVariableGet1i( m_variable, &i ) );
    return i;
  }

  inline void VariableObj::setBuffer(Buffer buffer)
  {
    checkError( rtVariableSetObject( m_variable, buffer->get() ) );
  }

  inline void VariableObj::set(Buffer buffer)
  {
    checkError( rtVariableSetObject( m_variable, buffer->get() ) );
  }

  inline void VariableObj::setUserData(RTsize size, const void* ptr)
  {
    checkError( rtVariableSetUserData( m_variable, size, ptr ) );
  }

  inline void VariableObj::getUserData(RTsize size,       void* ptr) const
  {
    checkError( rtVariableGetUserData( m_variable, size, ptr ) );
  }

  inline void VariableObj::setTextureSampler(TextureSampler texturesampler)
  {
    checkError( rtVariableSetObject( m_variable, texturesampler->get() ) );
  }

  inline void VariableObj::set(TextureSampler texturesampler)
  {
    checkError( rtVariableSetObject( m_variable, texturesampler->get() ) );
  }

  inline void VariableObj::set(GeometryGroup group)
  {
    checkError( rtVariableSetObject( m_variable, group->get() ) );
  }

  inline void VariableObj::set(Group group)
  {
    checkError( rtVariableSetObject( m_variable, group->get() ) );
  }

  inline void VariableObj::set(Program program)
  {
    checkError( rtVariableSetObject( m_variable, program->get() ) );
  }
  
  inline void VariableObj::set(Selector sel)
  {
    checkError( rtVariableSetObject( m_variable, sel->get() ) );
  }

  inline void VariableObj::set(Transform tran)
  {
    checkError( rtVariableSetObject( m_variable, tran->get() ) );
  }

  inline Buffer VariableObj::getBuffer() const
  {
    RTobject temp;
    checkError( rtVariableGetObject( m_variable, &temp ) );
    RTbuffer buffer = reinterpret_cast<RTbuffer>(temp);
    return Buffer::take(buffer);
  }

  inline std::string VariableObj::getName() const
  {
    const char* name;
    checkError( rtVariableGetName( m_variable, &name ) );
    return std::string(name);
  }

  inline std::string VariableObj::getAnnotation() const
  {
    const char* annotation;
    checkError( rtVariableGetAnnotation( m_variable, &annotation ) );
    return std::string(annotation);
  }

  inline RTobjecttype VariableObj::getType() const
  {
    RTobjecttype type;
    checkError( rtVariableGetType( m_variable, &type ) );
    return type;
  }

  inline RTvariable VariableObj::get()
  {
    return m_variable;
  }

  inline RTsize VariableObj::getSize() const
  {
    RTsize size;
    checkError( rtVariableGetSize( m_variable, &size ) );
    return size;
  }

  inline optix::TextureSampler VariableObj::getTextureSampler() const
  {
    RTobject temp;
    checkError( rtVariableGetObject( m_variable, &temp ) );
    RTtexturesampler sampler = reinterpret_cast<RTtexturesampler>(temp);
    return TextureSampler::take(sampler);
  }

  inline optix::Program VariableObj::getProgram() const
  {
    RTobject temp;
    checkError( rtVariableGetObject( m_variable, &temp ) );
    RTprogram program = reinterpret_cast<RTprogram>(temp);
    return Program::take(program);
  }

  /// @}
}

#endif /* __optixu_optixpp_namespace_h__ */

/// @}


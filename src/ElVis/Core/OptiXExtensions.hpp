///////////////////////////////////////////////////////////////////////////////
//
// The MIT License
//
// Copyright (c) 2006 Scientific Computing and Imaging Institute,
// University of Utah (USA)
//
// License for the specific language governing rights and limitations under
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ELVIS_OPTIX_EXTENSIONS_HPP
#define ELVIS_OPTIX_EXTENSIONS_HPP

#include <optixu/optixpp.h>
#include <optix_math.h>

#include <ElVis/Core/Point.hpp>

namespace ElVis
{
    template<typename OptixObjectType, typename T>
    void SetFloat(OptixObjectType obj, const T& value)
    {
        obj->setUserData(sizeof(T), &value);
    }

    template<typename OptixObjectType, typename T>
    void SetFloat(OptixObjectType obj, const T& x, const T& y, const T& z)
    {
        T buf[] = {x, y, z};
        obj->setUserData(sizeof(buf), buf);
    }

    template<typename OptixObjectType>
    void SetFloat(OptixObjectType obj, const WorldPoint& p)
    {
        SetFloat(obj, p.x(), p.y(), p.z());
    }

    template<typename OptixObjectType, typename T>
    void SetFloat(OptixObjectType obj, const T& x, const T& y, const T& z, const T& w)
    {
        T buf[] = {x, y, z, w};
        obj->setUserData(sizeof(buf), buf);
    }

    template<typename T>
    struct FormatMapping
    {
        static void SetElementSize(optixu::Buffer buffer)
        {
            buffer->setElementSize(sizeof(T));
        }
        static const RTformat value = RT_FORMAT_USER;
    };

    template<>
    struct FormatMapping<float>
    {
        static void SetElementSize(optixu::Buffer buffer)
        {
        }

        static const RTformat value = RT_FORMAT_FLOAT;
    };

    template<>
    struct FormatMapping<float2>
    {
        static void SetElementSize(optixu::Buffer buffer)
        {
        }
        static const RTformat value = RT_FORMAT_FLOAT2;
    };

    template<>
    struct FormatMapping<float3>
    {
        static void SetElementSize(optixu::Buffer buffer)
        {
        }
        static const RTformat value = RT_FORMAT_FLOAT3;
    };

    template<>
    struct FormatMapping<float4>
    {
        static void SetElementSize(optixu::Buffer buffer)
        {
        }
        static const RTformat value = RT_FORMAT_FLOAT4;
    };

    template<>
    struct FormatMapping<int>
    {
        static void SetElementSize(optixu::Buffer buffer)
        {
        }
        static const RTformat value = RT_FORMAT_INT;
    };

    template<>
    struct FormatMapping<unsigned int>
    {
        static void SetElementSize(optixu::Buffer buffer)
        {
        }
        static const RTformat value = RT_FORMAT_UNSIGNED_INT;
    };

    template<>
    struct FormatMapping<uint3>
    {
        static void SetElementSize(optixu::Buffer buffer)
        {
        }
        static const RTformat value = RT_FORMAT_UNSIGNED_INT3;
    };

    template<>
    struct FormatMapping<uint4>
    {
        static void SetElementSize(optixu::Buffer buffer)
        {
        }
        static const RTformat value = RT_FORMAT_UNSIGNED_INT4;
    };

    template<>
    struct FormatMapping<unsigned char>
    {
        static void SetElementSize(optixu::Buffer buffer)
        {
        }
        static const RTformat value = RT_FORMAT_UNSIGNED_BYTE;
    };

    template<>
    struct FormatMapping<uchar4>
    {
        static void SetElementSize(optixu::Buffer buffer)
        {
        }
        static const RTformat value = RT_FORMAT_UNSIGNED_BYTE4;
    };

    template<>
    struct FormatMapping<double>
    {
        static void SetElementSize(optixu::Buffer buffer)
        {
            buffer->setElementSize(sizeof(double));
        }
        static const RTformat value = RT_FORMAT_USER;
    };
    template<>
    struct FormatMapping<double2>
    {
        static void SetElementSize(optixu::Buffer buffer)
        {
            buffer->setElementSize(sizeof(double2));
        }
        static const RTformat value = RT_FORMAT_USER;
    };

    template<>
    struct FormatMapping<double3>
    {
        static void SetElementSize(optixu::Buffer buffer)
        {
            buffer->setElementSize(sizeof(double3));
        }
        static const RTformat value = RT_FORMAT_USER;
    };
    template<>
    struct FormatMapping<double4>
    {
        static void SetElementSize(optixu::Buffer buffer)
        {
            buffer->setElementSize(sizeof(double4));
        }
        static const RTformat value = RT_FORMAT_USER;
    };

  //,
  //,
  //,
  //RT_FORMAT_BYTE,
  //RT_FORMAT_BYTE2,
  //RT_FORMAT_BYTE3,
  //RT_FORMAT_BYTE4,
  //,
  //RT_FORMAT_UNSIGNED_BYTE2,
  //RT_FORMAT_UNSIGNED_BYTE3,
  //,
  //RT_FORMAT_SHORT,
  //RT_FORMAT_SHORT2,
  //RT_FORMAT_SHORT3,
  //RT_FORMAT_SHORT4,
  //RT_FORMAT_UNSIGNED_SHORT,
  //RT_FORMAT_UNSIGNED_SHORT2,
  //RT_FORMAT_UNSIGNED_SHORT3,
  //RT_FORMAT_UNSIGNED_SHORT4,
  //,
  //RT_FORMAT_INT2,
  //RT_FORMAT_INT3,
  //RT_FORMAT_INT4,
  //,
  //RT_FORMAT_UNSIGNED_INT2,
  //,
  //,

//    class RayGeneratorProgram;
//    void AddRayGenerationProgram(optixu::Context context, RayGeneratorProgram& rayGenProgram);
}

#endif //ELVIS_OPTIX_EXTENSIONS_HPP

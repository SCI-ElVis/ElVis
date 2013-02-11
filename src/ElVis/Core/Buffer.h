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

#ifndef ELVIS_BUFFER_H
#define ELVIS_BUFFER_H

#include <optixu/optixpp.h>
#include <ElVis/Core/ElVisDeclspec.h>

namespace ElVis
{
    /// \brief An interface for creating buffers for OptiX that use the currently
    ///        defined value for ElVisFloat.
    class FloatingPointBuffer
    {
        public:
            /// \brief Creates an OptiX buffer with the given name and dimension.
            ///
            /// OptiX buffers are semi-global variables that are visible to a set
            /// of OptiX programs.  These buffers are referenced by name in OptiX
            /// code.  That name is must be provided to this function to created
            /// the association between the C++ representation of the buffer and
            /// the OptiX representation.
            ///
            /// Data types in OptiX can often have different dimensions, such as
            /// float, float2, float3, etc, in which the dimension (1, 2, 3) represents
            /// how many floating point values are associated with each item in the
            /// buffer.  So if you need an OptiX array of float3, then dataDimension should
            /// be set to 3.
            ELVIS_EXPORT FloatingPointBuffer(const std::string& name, unsigned int dataDimension);

            /// \brief Creates a 2D OptiX buffer.
            ///
            /// This object exists on the CPU.  In order to get data to the GPU,
            /// it must be created through the OptiX interface.  This function
            /// creates the buffer in OptiX as a 2D buffer.
            ELVIS_EXPORT void Create(optixu::Context context, unsigned int type, 
                unsigned int width);

            /// \brief Creates a 3D OptiX buffer.
            ///
            /// This object exists on the CPU.  In order to get data to the GPU,
            /// it must be created through the OptiX interface.  This function
            /// creates the buffer in OptiX as a 3D buffer.
            ELVIS_EXPORT void Create(optixu::Context context, unsigned int type, 
                unsigned int width, unsigned int height);

            /// \brief Provides access to the underlying optixu::Buffer object.
            const optixu::Buffer& operator->() const { return m_buffer; }

            /// \brief Provides access to the underlying optixu::Buffer object.
            const optixu::Buffer& operator*() const { return m_buffer; }

            /// \brief Provides access to the underlying optixu::Buffer object.
            optixu::Buffer& operator->() { return m_buffer; }

            /// \brief Provides access to the underlying optixu::Buffer object.
            optixu::Buffer& operator*() { return m_buffer; }

            const std::string& Name() const { return m_name; }

        private:
            FloatingPointBuffer(const FloatingPointBuffer& rhs);
            FloatingPointBuffer& operator=(const FloatingPointBuffer& rhs);

            optixu::Buffer m_buffer;
            std::string m_name;
            unsigned int m_dataDimension;
    };

}
#endif //ELVIS_BUFFER_H

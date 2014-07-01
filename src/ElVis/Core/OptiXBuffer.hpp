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

#ifndef ELVIS_OPTIX_BUFFER_HPP
#define ELVIS_OPTIX_BUFFER_HPP

#include <ElVis/Core/OpenGL.h>
#include <ElVis/Core/Cuda.h>

#include <optixu/optixpp.h>
#include <ElVis/Core/OptiXExtensions.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <boost/bind.hpp>

#include <iostream>

namespace ElVis
{

    /// \brief An simplified interface for creating OptiX buffers.
    /// @param T the type to store in the buffer.
    /// @param BufferType one of   RT_BUFFER_INPUT, RT_BUFFER_OUTPUT, RT_BUFFER_INPUT_OUTPUT
    template<typename T, RTbuffertype BufferType = RT_BUFFER_INPUT_OUTPUT>
    class OptiXBuffer
    {
        public:
            explicit OptiXBuffer(const std::string& name) :
                m_optixBuffer(),
                m_name(name),
                m_width(1),
                m_height(1),
                m_context()
            {
            }

            ~OptiXBuffer()
            {
            }

            void SetDimensions(size_t w)
            {
                if( w == 0 )
                {
                    w = 1;
                }

                SetDimensions(w, 1);
            }

            void SetDimensions(size_t w, size_t h)
            {
                ReleaseResourcesIfNecessary();
                m_width = w;
                m_height = h;
                ClaimResourcesIfNecessary();
            }

            void SetContext(optixu::Context c)
            {
                m_context = c;
            }

            boost::shared_array<T> Map()
            {
                return boost::shared_array<T>(static_cast<T*>(m_optixBuffer->map()),
                                                            boost::bind(&OptiXBuffer::Unmap, this, _1));
            }

            boost::shared_array<T> map()
            {
                return Map();
            }

            bool Initialized() const
            {
                return m_optixBuffer;
            }

            T operator()(size_t x, size_t y)
            {
                T result;

                if( x < m_width && y < m_height )
                {
                    boost::shared_array<T> buf = Map();
                    result = buf[y*m_width + x];
                }

                return result;
            }

        private:
            void Unmap(T* ptr)
            {
                m_optixBuffer->unmap();
            }

            OptiXBuffer& operator=(const OptiXBuffer& rhs);
            OptiXBuffer(const OptiXBuffer& rhs);

            void ReleaseResourcesIfNecessary()
            {
              if( !m_context ) return;
              // Does this get rid of references in the context?
              if( m_optixBuffer )
              {
                  m_optixBuffer->destroy();
                  m_optixBuffer = optixu::Buffer();
              }
            }

            void ClaimResourcesIfNecessary()
            {
              if( !m_context ) return;
              // Setup the memory in OptiX.
              m_optixBuffer = m_context->createBuffer(BufferType);
              if( m_height == 1 )
              {
                  m_optixBuffer->setSize(m_width);
              }
              else
              {
                  m_optixBuffer->setSize(m_width, m_height);
              }
              m_optixBuffer->setFormat(FormatMapping<T>::value);
              FormatMapping<T>::SetElementSize(m_optixBuffer);
              m_context[m_name.c_str()]->set(m_optixBuffer);
            }

            optixu::Buffer m_optixBuffer;

            std::string m_name;
            size_t m_width;
            size_t m_height;
            optixu::Context m_context;
    };
}

#endif

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

#ifndef ELVIS_INTEROP_BUFFER_HPP
#define ELVIS_INTEROP_BUFFER_HPP

#include <ElVis/Core/OpenGL.h>
#include <ElVis/Core/Cuda.h>

#include <optixu/optixpp.h>
#include <ElVis/Core/OptiXExtensions.hpp>

#include <iostream>

namespace ElVis
{

    /// \brief Creates a buffer that can be used by both Cuda and Optix.
    ///
    /// Cuda is unable to access memory allocated for OptiX, and vice-versa.
    /// In order to share memory an OpenGL buffer object must first be created
    /// and shared between these technologies.  This class handes the allocation
    /// and sharing of resources.
    template<typename T>
    class InteropBuffer
    {
        public:
            explicit InteropBuffer(const std::string& name) :
                m_optixBuffer(),
                m_name(name),
                m_width(1),
                m_height(1),
                m_context()
            {
            }

            ~InteropBuffer()
            {
            }

            void SetDimensions(unsigned int w)
            {
                if( w == 0 )
                {
                    w = 1;
                }

                SetDimensions(w, 1);
            }

            void SetDimensions(unsigned int w, unsigned int h)
            {
                ReleaseResourcesIfNecessary();
                m_width = w;
                m_height = h;
                ClaimResourcesIfNecessary();
            }

            void setSize(unsigned int w, unsigned int h)
            {
                SetDimensions(w, h);
            }

            void SetContextInfo(optixu::Context c)
            {
                m_context = c;
            }

            T* MapOptiXPointer()
            {
                return static_cast<T*>(m_optixBuffer->map());
            }
            
            void UnmapOptiXPointer()
            {
                m_optixBuffer->unmap();
            }

            bool Initialized() 
            {
              if( m_optixBuffer ) return true;
              return false;
            }

            T operator()(unsigned int x, unsigned int y)
            {
                T result;

                if( x < m_width && y < m_height )
                {
                    T* buf = MapOptiXPointer();
                    result = buf[y*m_width + x];
                    UnmapOptiXPointer();
                }

                return result;
            }

        private:
            InteropBuffer& operator=(const InteropBuffer& rhs);
            InteropBuffer(const InteropBuffer& rhs);

            void ReleaseResourcesIfNecessary()
            {
              // Does this get rid of references in the context?
              if( m_optixBuffer )
              {
                  m_optixBuffer->destroy();
                  m_optixBuffer = optixu::Buffer();
              }
            }

            void ClaimResourcesIfNecessary()
            {

              // Setup the memory in OptiX.
              m_optixBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
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
            unsigned int m_width;
            unsigned int m_height;
            optixu::Context m_context;
    };
}

#endif

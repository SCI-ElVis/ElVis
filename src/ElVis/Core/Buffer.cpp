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

#include <ElVis/Core/Buffer.h>
#include <ElVis/Core/Float.h>

namespace ElVis
{
    FloatingPointBuffer::FloatingPointBuffer(const std::string& name, unsigned int dataDimension) :
        m_buffer(),
        m_name(name),
        m_dataDimension(dataDimension)
    {
    }   

    void FloatingPointBuffer::Create(optixu::Context context, unsigned int type, 
        unsigned int width, unsigned int height)
    {
        m_buffer = context->createBuffer(type, RT_FORMAT_USER, width, height);
        m_buffer->setElementSize(sizeof(ElVisFloat) * m_dataDimension);
    }

    void FloatingPointBuffer::Create(optixu::Context context, unsigned int type, 
        unsigned int width)
    {
        m_buffer = context->createBuffer(type, RT_FORMAT_USER, width);
        m_buffer->setElementSize(sizeof(ElVisFloat) * m_dataDimension);
    }
}

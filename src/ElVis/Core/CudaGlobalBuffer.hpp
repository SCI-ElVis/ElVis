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


#ifndef ELVIS_CUDA_GLOBAL_BUFFER_HPP
#define ELVIS_CUDA_GLOBAL_BUFFER_HPP

#include <ElVis/Core/Cuda.h>
#include <iostream>

namespace ElVis
{
    template<typename T>
    class CudaGlobalBuffer
    {
        public:
            CudaGlobalBuffer(const std::string& name, unsigned int size, CUmodule module) :
                m_name(name),
                m_size(size),
                m_mapBuffer(0),
                m_module(module),
                m_deviceMemory(0),
                m_deviceGlobalBuffer(0)
            {
            }

            void* map()
            {
                m_mapBuffer = new char[m_size*sizeof(T)];
                return (void*)m_mapBuffer;
            }

            void unmap()
            {
                if( m_deviceMemory )
                {
                    // Delete previously allocated memory.
                    cuMemFree(m_deviceMemory);
                    m_deviceMemory = 0;
                }

                // Copy data to device memory.
                CUresult r = cuMemAlloc(&m_deviceMemory, m_size*sizeof(T));
                r = cuMemcpyHtoD(m_deviceMemory, m_mapBuffer, m_size*sizeof(T));
                delete [] m_mapBuffer;
                m_mapBuffer = 0;

                // Locate the global buffer.  This will cause an error if there are no variables in the module
                // with the given name.
                size_t size = 0;
                checkedCudaCall(cuModuleGetGlobal(&m_deviceGlobalBuffer, &size, m_module, m_name.c_str()));

                // Copy the address of the device memory we allocated earlier to the global variable.
                checkedCudaCall(cuMemcpyHtoD(m_deviceGlobalBuffer, &m_deviceMemory, sizeof(CUdeviceptr)));
            }

        private:
            CudaGlobalBuffer(const CudaGlobalBuffer& rhs);
            CudaGlobalBuffer& operator=(const CudaGlobalBuffer& rhs);

            std::string m_name;
            unsigned int m_size;

            // temporary storage to copy data into host memory before being copied 
            // to device memory.
            char* m_mapBuffer;

            CUmodule m_module;

            CUdeviceptr m_deviceMemory;

            CUdeviceptr m_deviceGlobalBuffer;
    };
}

#endif

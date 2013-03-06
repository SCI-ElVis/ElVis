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

#ifndef ELVIS_CUDA_GLOBAL_VARIABLE_HPP
#define ELVIS_CUDA_GLOBAL_VARIABLE_HPP

#include <string>
#include <ElVis/Core/Cuda.h>

namespace ElVis
{
    template<typename T>
    class CudaGlobalVariable
    {
        public:
            CudaGlobalVariable(const std::string& name, CUmodule module) :
                m_devicePointer(0),
                m_name(name),
                m_module(module)
            {
            }

            explicit CudaGlobalVariable(const std::string& name) :
                m_devicePointer(0),
                m_name(name),
                m_module(0)
            {
            }

            ~CudaGlobalVariable()
            {
                cuMemFree(m_devicePointer);
                m_devicePointer = 0;
            }

            void WriteToDevice(const T& data)
            {
                if( !m_devicePointer)
                {
                    size_t size = 0;
                    checkedCudaCall(cuModuleGetGlobal(&m_devicePointer, &size, m_module, m_name.c_str()));

                    if( size != sizeof(T) )
                    {
                        std::cout << "Size of global variable " << m_name << " does not match size of datatype." << std::endl;
                    }    
                }

                checkedCudaCall(cuMemcpyHtoD(m_devicePointer, &data, sizeof(T)));
            }

        private:
            CudaGlobalVariable(const CudaGlobalVariable&);
            CudaGlobalVariable& operator=(const CudaGlobalVariable& rhs);

            CUdeviceptr m_devicePointer;
            std::string m_name;
            CUmodule m_module;
    };
};


#endif //ELVIS_CUDA_GLOBAL_VARIABLE_HPP

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

#ifndef ELVIS_CORE_HOST_TRANSFER_FUNCTION_H
#define ELVIS_CORE_HOST_TRANSFER_FUNCTION_H

#include <ElVis/Core/TransferFunction.h>
#include <ElVis/Core/ColorMap.h>
#include <ElVis/Core/OptiXBuffer.hpp>
#include <boost/signals2.hpp>
#include <optixu/optixpp.h>

namespace ElVis
{
    struct Breakpoint
    {
        ElVis::Color Col;
        ElVisFloat Density;
        ElVisFloat Scalar;
    };
    
    class HostTransferFunction
    {
        public:
            ELVIS_EXPORT HostTransferFunction();
            ELVIS_EXPORT TransferFunction GetOptixObject();

            ELVIS_EXPORT void SetBreakpoint(double s, const Color& c);
            ELVIS_EXPORT void SetBreakpoint(double s, const Color& c, const ElVisFloat& density);

            ELVIS_EXPORT bool IsValid() const
            {
                return m_breakpoints.size() >= 2;
            }

            ELVIS_EXPORT const std::map<double, Breakpoint>& GetBreakpoints() const { return m_breakpoints; }

            ELVIS_EXPORT void Clear();

            ELVIS_EXPORT void CopyToOptix(optixu::Context context, OptiXBuffer<ElVisFloat>& buffer, OptiXBuffer<ElVisFloat>& values, TransferFunctionChannel channel);

            boost::signals2::signal<void (void)> OnTransferFunctionChanged;

            bool& Dirty() { return m_dirty; }

        protected:

        private:
            HostTransferFunction(const HostTransferFunction&);
            HostTransferFunction& operator=(const HostTransferFunction&);

            void UpdateBreakpoints(std::map<double, double>& container);

            void SynchronizeDeviceIfNeeded();
            void SynchronizeOptiXIfNeeded();
            void FreeDeviceMemory();
            void AllocateDeviceMemory();
            void CopyToDeviceMemory();

            std::map<double, Breakpoint> m_breakpoints;

            TransferFunction m_localDeviceTransferFunction;
            bool m_dirty;
    };
}

#endif

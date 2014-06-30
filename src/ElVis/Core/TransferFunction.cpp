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

#include <ElVis/Core/TransferFunction.h>
#include <ElVis/Core/HostTransferFunction.h>

#include <ElVis/Core/Cuda.h>
#include <boost/typeof/typeof.hpp>

namespace ElVis
{
    HostTransferFunction::HostTransferFunction() :
        m_breakpoints(),
        m_localDeviceTransferFunction(),
        m_dirty(true)
    {
        m_localDeviceTransferFunction.Initialize();
    }
        

    void HostTransferFunction::FreeDeviceMemory()
    {

    }

    void HostTransferFunction::AllocateDeviceMemory()
    {

    }

    void HostTransferFunction::CopyToOptix(optixu::Context context, OptiXBuffer<ElVisFloat>& breakpoints, OptiXBuffer<ElVisFloat>& values, TransferFunctionChannel channel)
    {
        breakpoints.SetContext(context);
        breakpoints.SetDimensions(m_breakpoints.size());
        values.SetContext(context);
        values.SetDimensions(m_breakpoints.size());
        BOOST_AUTO(breakpointData, breakpoints.Map());
        BOOST_AUTO(valueData,values.Map());

        int index = 0;
        for(std::map<double, Breakpoint>::iterator iter = m_breakpoints.begin(); iter != m_breakpoints.end(); ++iter)
        {
            breakpointData[index] = (*iter).first;

            if( channel == eDensity ) valueData[index] = (*iter).second.Density;
            if( channel == eRed ) valueData[index] = (*iter).second.Col.Red();
            if( channel == eGreen ) valueData[index] = (*iter).second.Col.Green();
            if( channel == eBlue ) valueData[index] = (*iter).second.Col.Blue();

            ++index;
        }
    }


    void HostTransferFunction::CopyToDeviceMemory()
    {

    }

    void HostTransferFunction::SynchronizeDeviceIfNeeded()
    {
        if (!m_dirty) return;

        FreeDeviceMemory();
        AllocateDeviceMemory();
        CopyToDeviceMemory();

        // Reset after OptiX updates.
        m_dirty = true;
    }

    void HostTransferFunction::SynchronizeOptiXIfNeeded()
    {
        if( !m_dirty) return;
    }

    void HostTransferFunction::SetBreakpoint(double s, const Color& c)
    {
        SetBreakpoint(s, c, c.Alpha());
    }

    void HostTransferFunction::SetBreakpoint(double s, const Color& c, const ElVisFloat& density)
    {
        if( m_breakpoints.find(s) == m_breakpoints.end() )
        {
            m_breakpoints[s].Col = c;
            m_breakpoints[s].Density = density;
            m_breakpoints[s].Scalar = static_cast<ElVisFloat>(s);
            m_dirty = true;
            OnTransferFunctionChanged();
        }
    }

    void HostTransferFunction::Clear()
    {
        if( m_breakpoints.size() > 0 )
        {
            m_breakpoints = std::map<double, Breakpoint>();
            m_dirty = true;
            OnTransferFunctionChanged();
        }
    }

}

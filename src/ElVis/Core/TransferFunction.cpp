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

namespace ElVis
{
    HostTransferFunction::HostTransferFunction() :
        m_breakpoints(),
        m_deviceDensityBreakpoints(0),
        m_deviceRedBreakpoints(0),
        m_deviceGreenBreakpoints(0),
        m_deviceBlueBreakpoints(0),
        m_deviceDensityValues(0),
        m_deviceRedValues(0),
        m_deviceGreenValues(0),
        m_deviceBlueValues(0),
        m_deviceObject(0),
        m_localDeviceTransferFunction(),
        m_dirty(true)
    {
        m_localDeviceTransferFunction.Initialize();
    }
        
    CUdeviceptr HostTransferFunction::GetDeviceObject()
    {
        SynchronizeDeviceIfNeeded();
        return m_deviceObject;
    }

    void HostTransferFunction::InitializeArrayIfNeeded(CUdeviceptr& devicePtr, ElVisFloat* data, int size)
    {
//        if( devicePtr != 0 ) return;

//        checkedCudaCall(cuMemAlloc(&devicePtr, size*sizeof(ElVisFloat)));
//        checkedCudaCall(cuMemcpyHtoD(devicePtr, data, size*sizeof(ElVisFloat)));
    }

    void HostTransferFunction::FreeDeviceMemory()
    {
//        // Free existing memory.
//        if( m_deviceDensityBreakpoints != 0 )
//        {
//            checkedCudaCall(cuMemFree(m_deviceDensityBreakpoints));
//            m_deviceDensityBreakpoints = 0;
//            //delete [] m_localDeviceTransferFunction.DensityBreakpoints();
//            m_localDeviceTransferFunction.DensityBreakpoints() = 0;
//        }

//        if( m_deviceRedBreakpoints != 0 )
//        {
//            checkedCudaCall(cuMemFree(m_deviceRedBreakpoints));
//            m_deviceRedBreakpoints = 0;
//            //delete [] m_localDeviceTransferFunction.RedBreakpoints();
//            m_localDeviceTransferFunction.RedBreakpoints() = 0;
//        }

//        if( m_deviceGreenBreakpoints != 0 )
//        {
//            checkedCudaCall(cuMemFree(m_deviceGreenBreakpoints));
//            m_deviceGreenBreakpoints = 0;
//            //delete [] m_localDeviceTransferFunction.GreenBreakpoints();
//            m_localDeviceTransferFunction.GreenBreakpoints() = 0;
//        }

//        if( m_deviceBlueBreakpoints != 0 )
//        {
//            checkedCudaCall(cuMemFree(m_deviceBlueBreakpoints));
//            m_deviceBlueBreakpoints = 0;
//            //delete [] m_localDeviceTransferFunction.BlueBreakpoints();
//            m_localDeviceTransferFunction.BlueBreakpoints() = 0;
//        }

//        if( m_deviceDensityValues != 0 )
//        {
//            checkedCudaCall(cuMemFree(m_deviceDensityValues));
//            m_deviceDensityValues = 0;
//            //delete [] m_localDeviceTransferFunction.DensityValues();
//            m_localDeviceTransferFunction.DensityValues() = 0;
//        }

//        if( m_deviceRedValues != 0 )
//        {
//            checkedCudaCall(cuMemFree(m_deviceRedValues));
//            m_deviceRedValues = 0;
//            //delete [] m_localDeviceTransferFunction.RedValues();
//            m_localDeviceTransferFunction.RedValues() = 0;
//        }

//        if( m_deviceGreenValues != 0 )
//        {
//            checkedCudaCall(cuMemFree(m_deviceGreenValues));
//            m_deviceGreenValues = 0;
//            //delete [] m_localDeviceTransferFunction.GreenValues();
//            m_localDeviceTransferFunction.GreenValues() = 0;
//        }

//        if( m_deviceBlueValues != 0 )
//        {
//            checkedCudaCall(cuMemFree(m_deviceBlueValues));
//            m_deviceBlueValues = 0;
//            //delete [] m_localDeviceTransferFunction.BlueValues();
//            m_localDeviceTransferFunction.BlueValues() = 0;
//        }

//        m_localDeviceTransferFunction.NumDensityBreakpoints() = 0;
//        m_localDeviceTransferFunction.NumRedBreakpoints() = 0;
//        m_localDeviceTransferFunction.NumGreenBreakpoints() = 0;
//        m_localDeviceTransferFunction.NumBlueBreakpoints() = 0;

//        if( m_deviceObject != 0 )
//        {
//            checkedCudaCall(cuMemFree(m_deviceObject));
//            m_deviceObject = 0;
//        }
    }

    void HostTransferFunction::AllocateDeviceMemory()
    {
//        checkedCudaCall(cuMemAlloc(&m_deviceDensityBreakpoints, m_breakpoints.size()*sizeof(ElVisFloat)));
//        checkedCudaCall(cuMemAlloc(&m_deviceRedBreakpoints, m_breakpoints.size()*sizeof(ElVisFloat)));
//        checkedCudaCall(cuMemAlloc(&m_deviceGreenBreakpoints, m_breakpoints.size()*sizeof(ElVisFloat)));
//        checkedCudaCall(cuMemAlloc(&m_deviceBlueBreakpoints, m_breakpoints.size()*sizeof(ElVisFloat)));

//        checkedCudaCall(cuMemAlloc(&m_deviceDensityValues, m_breakpoints.size()*sizeof(ElVisFloat)));
//        checkedCudaCall(cuMemAlloc(&m_deviceRedValues, m_breakpoints.size()*sizeof(ElVisFloat)));
//        checkedCudaCall(cuMemAlloc(&m_deviceGreenValues, m_breakpoints.size()*sizeof(ElVisFloat)));
//        checkedCudaCall(cuMemAlloc(&m_deviceBlueValues, m_breakpoints.size()*sizeof(ElVisFloat)));

//        checkedCudaCall(cuMemAlloc(&m_deviceObject, sizeof(TransferFunction)));
    }

    void HostTransferFunction::CopyToOptix(optixu::Context context, FloatingPointBuffer& breakpoints, FloatingPointBuffer& values, TransferFunctionChannel channel)
    {
        breakpoints.Create(context,RT_BUFFER_INPUT, m_breakpoints.size());
        values.Create(context, RT_BUFFER_INPUT, m_breakpoints.size());
        ElVisFloat* breakpointData = static_cast<ElVisFloat*>(breakpoints->map());
        ElVisFloat* valueData = static_cast<ElVisFloat*>(values->map());

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

        breakpoints->unmap();
        values->unmap();
        context[breakpoints.Name().c_str()]->set(*breakpoints);
        context[values.Name().c_str()]->set(*values);
    }


    void HostTransferFunction::CopyToDeviceMemory()
    {
//        ElVisFloat* DensityBreakpoints = new ElVisFloat[m_breakpoints.size()];
//        ElVisFloat* RedBreakpoints = new ElVisFloat[m_breakpoints.size()];
//        ElVisFloat* GreenBreakpoints = new ElVisFloat[m_breakpoints.size()];
//        ElVisFloat* BlueBreakpoints = new ElVisFloat[m_breakpoints.size()];

//        ElVisFloat* DensityValues = new ElVisFloat[m_breakpoints.size()];
//        ElVisFloat* RedValues = new ElVisFloat[m_breakpoints.size()];
//        ElVisFloat* GreenValues = new ElVisFloat[m_breakpoints.size()];
//        ElVisFloat* BlueValues = new ElVisFloat[m_breakpoints.size()];

//        int index = 0;
//        for(std::map<double, Breakpoint>::iterator iter = m_breakpoints.begin(); iter != m_breakpoints.end(); ++iter)
//        {
//            DensityBreakpoints[index] = static_cast<ElVisFloat>((*iter).first);
//            DensityValues[index] = (*iter).second.Density;
//            ++index;
//        }

//        index = 0;
//        for(std::map<double, Breakpoint>::iterator iter = m_breakpoints.begin(); iter != m_breakpoints.end(); ++iter)
//        {
//            RedBreakpoints[index] = static_cast<ElVisFloat>((*iter).first);
//            RedValues[index] = (*iter).second.Col.Red();
//            ++index;
//        }

//        index = 0;
//        for(std::map<double, Breakpoint>::iterator iter = m_breakpoints.begin(); iter != m_breakpoints.end(); ++iter)
//        {
//            GreenBreakpoints[index] = static_cast<ElVisFloat>((*iter).first);
//            GreenValues[index] = (*iter).second.Col.Green();
//            ++index;
//        }

//        index = 0;
//        for(std::map<double, Breakpoint>::iterator iter = m_breakpoints.begin(); iter != m_breakpoints.end(); ++iter)
//        {
//            BlueBreakpoints[index] = static_cast<ElVisFloat>((*iter).first);
//            BlueValues[index] = (*iter).second.Col.Blue();
//            ++index;
//        }
           
//        checkedCudaCall(cuMemcpyHtoD(m_deviceDensityBreakpoints, DensityBreakpoints, m_breakpoints.size()*sizeof(ElVisFloat)));
//        checkedCudaCall(cuMemcpyHtoD(m_deviceRedBreakpoints, RedBreakpoints, m_breakpoints.size()*sizeof(ElVisFloat)));
//        checkedCudaCall(cuMemcpyHtoD(m_deviceGreenBreakpoints, GreenBreakpoints, m_breakpoints.size()*sizeof(ElVisFloat)));
//        checkedCudaCall(cuMemcpyHtoD(m_deviceBlueBreakpoints, BlueBreakpoints, m_breakpoints.size()*sizeof(ElVisFloat)));

//        checkedCudaCall(cuMemcpyHtoD(m_deviceDensityValues, DensityValues, m_breakpoints.size()*sizeof(ElVisFloat)));
//        checkedCudaCall(cuMemcpyHtoD(m_deviceRedValues, RedValues, m_breakpoints.size()*sizeof(ElVisFloat)));
//        checkedCudaCall(cuMemcpyHtoD(m_deviceGreenValues, GreenValues, m_breakpoints.size()*sizeof(ElVisFloat)));
//        checkedCudaCall(cuMemcpyHtoD(m_deviceBlueValues, BlueValues, m_breakpoints.size()*sizeof(ElVisFloat)));

//        m_localDeviceTransferFunction.RedBreakpoints() = (ElVisFloat*)(m_deviceRedBreakpoints);
//        m_localDeviceTransferFunction.GreenBreakpoints() = (ElVisFloat*)(m_deviceGreenBreakpoints);
//        m_localDeviceTransferFunction.BlueBreakpoints() = (ElVisFloat*)(m_deviceBlueBreakpoints);
//        m_localDeviceTransferFunction.DensityBreakpoints() = (ElVisFloat*)(m_deviceDensityBreakpoints);

//        m_localDeviceTransferFunction.RedValues() = (ElVisFloat*)(m_deviceRedValues);
//        m_localDeviceTransferFunction.GreenValues() = (ElVisFloat*)(m_deviceGreenValues);
//        m_localDeviceTransferFunction.BlueValues() = (ElVisFloat*)(m_deviceBlueValues);
//        m_localDeviceTransferFunction.DensityValues() = (ElVisFloat*)(m_deviceDensityValues);

//        m_localDeviceTransferFunction.NumDensityBreakpoints() = static_cast<int>(m_breakpoints.size());
//        m_localDeviceTransferFunction.NumRedBreakpoints() = static_cast<int>(m_breakpoints.size());
//        m_localDeviceTransferFunction.NumGreenBreakpoints() = static_cast<int>(m_breakpoints.size());
//        m_localDeviceTransferFunction.NumBlueBreakpoints() = static_cast<int>(m_breakpoints.size());

//        checkedCudaCall(cuMemcpyHtoD(m_deviceObject, &m_localDeviceTransferFunction, sizeof(TransferFunction)));

//        delete [] DensityBreakpoints;
//        delete [] RedBreakpoints;
//        delete [] GreenBreakpoints;
//        delete [] BlueBreakpoints;

//        delete [] DensityValues;
//        delete [] RedValues;
//        delete [] GreenValues;
//        delete [] BlueValues;

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

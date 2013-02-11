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


#include <fstream>

#include <ElVis/Core/ColorMap.h>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/SceneView.h>
#include <ElVis/Core/OptiXExtensions.hpp>

#include <boost/filesystem.hpp>

namespace ElVis
{
    ColorMap::ColorMap() :
        m_min(0.0f),
        m_max(1.0f)
    {
    }

    void ColorMap::SetMin(float value)
    {
        if( value != m_min )
        {
            m_min = value;
            OnMinChanged(value);
        }
    }

    void ColorMap::SetMax(float value)
    {
        if( value != m_max )
        {
            m_max = value;
            OnMaxChanged(value);
        }
    }


    void TextureColorMap::DoPopulateTexture(optixu::Buffer& buffer)
    {
        RTsize bufSize;
        buffer->getSize(bufSize);
        if( bufSize*4 != m_localData.size() )
        {
            std::cout << "Local data size " << m_localData.size() << std::endl;
            buffer->setSize(m_localData.size()/4);
        }

        float* colorMapData = static_cast<float*>(buffer->map());
        std::copy(m_localData.begin(), m_localData.end(), colorMapData);
        buffer->unmap();

    }

    TextureColorMap::TextureColorMap(const std::string& fileName) :
        m_localData()
    {
        Read(fileName);
    }
            
    void TextureColorMap::Read(const std::string& fileName)
    {
        if( !boost::filesystem::exists(fileName) )
        {
            std::cout << "Unable to read " << fileName << std::endl;
            return;
        }
        std::ifstream inFile(fileName.c_str());
        float min;
        inFile >> min;
        float max;
        inFile >> max;
        
        SetMin(min);
        SetMax(max);
        unsigned int size;
        inFile >> size;
        
        std::cout << "Reading color map information from " << fileName << std::endl;
        std::cout << "Min = " << min << std::endl;
        std::cout << "Max = " << max << std::endl;
        
        m_localData.resize(size*4);
        
        float t;
        for(unsigned int i = 0; i < size*4; ++i)
        {
            inFile >> t;
            m_localData[i] = t;
        }
        inFile.close();
    }
    
//    void TextureColorMap::Write(const std::string& fileName)
//    {
//        std::ofstream outFile(fileName.c_str());
        
//        outFile << GetMin() << std::endl;
//        outFile << GetMax() << std::endl;
//        outFile << m_size << std::endl;
        
//        for(unsigned int i = 0; i < m_localData.size(); ++i)
//        {
//            outFile << m_localData[i] << std::endl;
//        }
        
//        outFile.close();
//    }

//    void TextureColorMap::WriteVTK(const std::string& fileName)
//    {
//        std::ofstream outFile(fileName.c_str());
//        outFile << "<ColorMap name=\"Exported\" space=\"RGB\">" << std::endl;

//        unsigned int wrote = 0;
//        float prev[3];
//        for(unsigned int i = 0; i < m_size; ++i)
//        {
//            if( i != 0 && i != m_size-1)
//            {
//                if( (int)(prev[0]*255.0) == (int)(m_localData[4*i]*255.0) &&
//                    (int)(prev[1]*255.0) == (int)(m_localData[4*i+1]*255.0) &&
//                    (int)(prev[2]*255.0) == (int)(m_localData[4*i+2]*255.0) )
//                {
//                    continue;
//                }
//            }

//            prev[0] = m_localData[4*i];
//            prev[1] = m_localData[4*i+1];
//            prev[2] = m_localData[4*i+2];

//            ++wrote;
//            outFile << "<Point x=\"" << (double)i/(m_size-1) << "\" o=\"1\" ";
//            outFile <<  "r=\"" << m_localData[4*i] << "\" ";
//            outFile <<  "g=\"" << m_localData[4*i+1] << "\" ";
//            outFile <<  "b=\"" << m_localData[4*i+2] << "\" ";
//            outFile << "/>" << std::endl;
//        }

//        outFile << "</ColorMap>" << std::endl;
//        outFile.close();

//        std::cout << "Wrote " << wrote << " samples out of " << m_size << std::endl;
//    }
    
    PiecewiseLinearColorMap::PiecewiseLinearColorMap() :
        ColorMap(),
        m_breakpoints()
    {
    }

    void PiecewiseLinearColorMap::SetBreakpoint(ElVisFloat value, const Color& c)
    {
        std::map<ElVisFloat, ColorMapBreakpoint>::iterator found = m_breakpoints.find(value);
        if( found == m_breakpoints.end() )
        {
            ColorMapBreakpoint breakpoint;
            breakpoint.Col = c;
            breakpoint.Scalar = value;
            m_breakpoints[value] = breakpoint;
            OnColorMapChanged(*this);
        }
    }

    void PiecewiseLinearColorMap::SetBreakpoint(const std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator& iter, const Color& c)
    {
        if( iter == m_breakpoints.end() ) return;

        if( (*iter).second.Col == c ) return;

        std::map<ElVisFloat, ColorMapBreakpoint>::iterator found = m_breakpoints.find((*iter).first);
        if( found != m_breakpoints.end() )
        {
            (*found).second.Col = c;
            OnColorMapChanged(*this);
        }
    }

    std::map<ElVisFloat, ColorMapBreakpoint>::iterator PiecewiseLinearColorMap::InsertBreakpoint(ElVisFloat value, const Color& c)
    {
        ColorMapBreakpoint point;
        point.Col = c;
        point.Scalar = value;

        std::map<ElVisFloat, ColorMapBreakpoint>::value_type v(value, point);
        std::pair<std::map<ElVisFloat, ColorMapBreakpoint>::iterator, bool> result = 
            m_breakpoints.insert(v);

        OnColorMapChanged(*this);
        return result.first;
    }

    void PiecewiseLinearColorMap::RemoveBreakpoint(const std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator iter)
    {
        if( iter != m_breakpoints.end() )
        {
            m_breakpoints.erase((*iter).first);
            OnColorMapChanged(*this);
        }
    }

    Color PiecewiseLinearColorMap::Sample(const ElVisFloat& value) const
    {
        std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator iter = m_breakpoints.lower_bound(value);

        if( iter == m_breakpoints.end() )
        {

            return (*m_breakpoints.rbegin()).second.Col;
        }
        else if( iter == m_breakpoints.begin() )
        {
            return (*m_breakpoints.begin()).second.Col;
        }
        else
        {
            std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator prev = iter;
            --prev;
            double percent = (value-(*prev).first)/((*iter).first - (*prev).first);
            Color c = (*prev).second.Col + ((*iter).second.Col - (*prev).second.Col)*percent;
            return c;
        }

    }

    void PiecewiseLinearColorMap::DoPopulateTexture(optixu::Buffer& buffer)
    {
        RTsize bufSize;
        buffer->getSize(bufSize);
        std::cout << "Piecwise size " << bufSize << std::endl;
        int entries = bufSize;

        // Since the buffer is an ElVisFloat4, the actual memory size is 4*bufSize;
        float* colorMapData = static_cast<float*>(buffer->map());
        for(int i = 0; i < entries; ++i)
        {
            double p = static_cast<double>(i)/(static_cast<double>(entries-1));
            Color c = Sample(p);
            colorMapData[i*4] = c.Red();
            colorMapData[i*4+1] = c.Green();
            colorMapData[i*4+2] = c.Blue();
            colorMapData[i*4+3] = c.Alpha();
        }
        buffer->unmap();
    }
}

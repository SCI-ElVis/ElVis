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
  ColorMap::ColorMap() : m_min(0.0f), m_max(1.0f) , m_breakpoints(){}

  void ColorMap::SetMin(float value)
  {
    if (value != m_min)
    {
      m_min = value;
      OnMinChanged(value);
    }
  }

  void ColorMap::SetMax(float value)
  {
    if (value != m_max)
    {
      m_max = value;
      OnMaxChanged(value);
    }
  }

  void ColorMap::SetBreakpoint(ElVisFloat value, const Color& c)
  {
    std::map<ElVisFloat, ColorMapBreakpoint>::iterator found =
      m_breakpoints.find(value);
    if (found == m_breakpoints.end())
    {
      ColorMapBreakpoint breakpoint;
      breakpoint.Col = c;
      breakpoint.Scalar = value;
      m_breakpoints[value] = breakpoint;
      OnColorMapChanged(*this);
    }
  }

  void ColorMap::SetBreakpoint(
    const std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator& iter,
    const Color& c)
  {
    if (iter == m_breakpoints.end()) return;

    if ((*iter).second.Col == c) return;

    std::map<ElVisFloat, ColorMapBreakpoint>::iterator found =
      m_breakpoints.find((*iter).first);
    if (found != m_breakpoints.end())
    {
      (*found).second.Col = c;
      OnColorMapChanged(*this);
    }
  }

  std::map<ElVisFloat, ColorMapBreakpoint>::iterator ColorMap::
    InsertBreakpoint(ElVisFloat value, const Color& c)
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

  void ColorMap::RemoveBreakpoint(
    const std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator iter)
  {
    if (iter != m_breakpoints.end())
    {
      m_breakpoints.erase((*iter).first);
      OnColorMapChanged(*this);
    }
  }

  Color ColorMap::Sample(const ElVisFloat& value) const
  {
    std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator iter =
      m_breakpoints.lower_bound(value);

    if (iter == m_breakpoints.end())
    {

      return (*m_breakpoints.rbegin()).second.Col;
    }
    else if (iter == m_breakpoints.begin())
    {
      return (*m_breakpoints.begin()).second.Col;
    }
    else
    {
      std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator prev = iter;
      --prev;
      double percent =
        (value - (*prev).first) / ((*iter).first - (*prev).first);
      Color c = (*prev).second.Col +
                ((*iter).second.Col - (*prev).second.Col) * percent;
      return c;
    }
  }

  void ColorMap::PopulateTexture(optixu::Buffer& buffer)
  {
    RTsize bufSize;
    buffer->getSize(bufSize);
    std::cout << "Piecwise size " << bufSize << std::endl;
    int entries = bufSize;

    // Since the buffer is an ElVisFloat4, the actual memory size is 4*bufSize;
    float* colorMapData = static_cast<float*>(buffer->map());
    for (int i = 0; i < entries; ++i)
    {
      double p = static_cast<double>(i) / (static_cast<double>(entries - 1));
      Color c = Sample(p);
      colorMapData[i * 4] = c.Red();
      colorMapData[i * 4 + 1] = c.Green();
      colorMapData[i * 4 + 2] = c.Blue();
      colorMapData[i * 4 + 3] = c.Alpha();
    }
    buffer->unmap();
  }
}

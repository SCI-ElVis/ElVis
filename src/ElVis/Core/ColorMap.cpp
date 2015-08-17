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

// Serialization keys
namespace
{
  const std::string MIN_KEY_NAME("Min");
  const std::string MAX_KEY_NAME("Max");
  const std::string BREAKPOINTS_KEY_NAME("Breakpoints");
}

namespace ElVis
{
  ColorMap::ColorMap() : m_min(0.0f), m_max(1.0f) , m_breakpoints(){}

  void ColorMap::SetMin(float value)
  {
    if (value != m_min && value < m_max)
    {
      m_min = value;
      OnMinChanged(value);
    }
  }

  void ColorMap::SetMax(float value)
  {
    if (value != m_max && value > m_min)
    {
      m_max = value;
      OnMaxChanged(value);
    }
  }

  std::map<ElVisFloat, ColorMapBreakpoint>::iterator
  ColorMap::SetBreakpoint(ElVisFloat value, const Color& c)
  {
    if( value < 0.0f || value > 1.0f ) return m_breakpoints.end();

    std::map<ElVisFloat, ColorMapBreakpoint>::iterator found =
      m_breakpoints.find(value);
    if (found == m_breakpoints.end())
    {
      ColorMapBreakpoint breakpoint;
      breakpoint.Col = c;
      breakpoint.Scalar = value;
      auto iter = m_breakpoints.insert(std::make_pair(value, breakpoint));
      OnColorMapChanged(*this);
      return iter.first;
    }
    else if(found->second.Col != c)
    {
      found->second.Col = c;
      OnColorMapChanged(*this);
      return found;
    }

    return m_breakpoints.end();
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

  /// \brief Serializes a camera to an archive.
  /// \param ar The serialization destination.
  template <typename Archive>
  void ColorMap::save(Archive& ar, const unsigned int /*version*/) const
  {
    ar & boost::serialization::make_nvp(MIN_KEY_NAME.c_str(), m_min);
    ar & boost::serialization::make_nvp(MAX_KEY_NAME.c_str(), m_max);
    ar & boost::serialization::make_nvp(BREAKPOINTS_KEY_NAME.c_str(), m_breakpoints);
  }

  /// \brief Deserializes a camera from an archive.
  /// \param ar The serialization source.
  template <typename Archive>
  void ColorMap::load(Archive& ar, const unsigned int /*version*/)
  {
    ar & boost::serialization::make_nvp(MIN_KEY_NAME.c_str(), m_min);
    ar & boost::serialization::make_nvp(MAX_KEY_NAME.c_str(), m_max);
    ar & boost::serialization::make_nvp(BREAKPOINTS_KEY_NAME.c_str(), m_breakpoints);
  }

  template void ColorMap::save(boost::archive::xml_oarchive&,
                                const unsigned int) const;

  template void ColorMap::load(boost::archive::xml_iarchive&,
                                const unsigned int);
}




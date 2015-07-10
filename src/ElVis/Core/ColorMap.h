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

#ifndef COLOR_MAP_H
#define COLOR_MAP_H

#include <optixu/optixpp.h>
#include <optix_math.h>

#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/Color.h>
#include <ElVis/Core/RayGeneratorProgram.h>
#include <ElVis/Core/PtxManager.h>
#include <ElVis/Core/Float.h>

#include <iostream>

#include <boost/signals2.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/map.hpp>

namespace ElVis
{
  class SceneView;

  /// \brief A node in the piecewise-linear color map defined by the ColorMap
  /// class.
  struct ColorMapBreakpoint
  {
    friend class boost::serialization::access;
    ElVis::Color Col;
    ElVisFloat Scalar;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
      ar& BOOST_SERIALIZATION_NVP(Col);
      ar& BOOST_SERIALIZATION_NVP(Scalar);
    }
  };

  /// \brief A piecewise-linear color map consisting of breakpoints on [0,1]
  /// with an associated color.  Samples falling between breakpoints are
  /// linearly interpolated, while samples falling outside the range are
  /// clamped.
  ///
  /// Breakpoints are specified on [0,1], but the color map can be scaled
  /// to any scalar range using SetMin and SetMax.
  class ColorMap
  {
  public:
    friend class boost::serialization::access;
    ELVIS_EXPORT ColorMap();
    ELVIS_EXPORT ~ColorMap() {}

    /// \brief Sets the minimum scalar value for the color map.
    ELVIS_EXPORT void SetMin(float value);

    /// \brief Sets the maximum scalar value for the color map.
    ELVIS_EXPORT void SetMax(float value);

    /// \brief Returns the minimum scalar value.
    ELVIS_EXPORT float GetMin() const { return m_min; }

    /// \brief Returns the maximum scalar value.
    ELVIS_EXPORT float GetMax() const { return m_max; }

    /// \brief Sets a breakpoint.
    /// \param[in] value The scalar value of the breakpoint on [0,1]
    /// \param[in] c The breakpoint's color.
    /// \returns An iterator to the breakpoint.  If value is outside the
    ///          range of [0,1], then this will be an invalid iterator.
    /// TODO - Why are we exposing internals?  The gui uses it, but why?
    ELVIS_EXPORT std::map<ElVisFloat, ColorMapBreakpoint>::iterator
      SetBreakpoint(ElVisFloat value, const Color& c);

    /// Returns the breakpoint collection.
    ELVIS_EXPORT const std::map<ElVisFloat, ColorMapBreakpoint>&
    GetBreakpoints() const
    {
      return m_breakpoints;
    }

    /// Removes the given breakpoint from the color map.
    ELVIS_EXPORT void RemoveBreakpoint(
      const std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator iter);
    ELVIS_EXPORT Color Sample(const ElVisFloat& value) const;

    ELVIS_EXPORT bool IsValid() const { return m_breakpoints.size() >= 2; }


    boost::signals2::signal<void(float)> OnMinChanged;
    boost::signals2::signal<void(float)> OnMaxChanged;
    boost::signals2::signal<void(const ColorMap&)> OnColorMapChanged;

    ELVIS_EXPORT void PopulateTexture(optixu::Buffer& buffer);

  private:
    ColorMap(const ColorMap& rhs);
    ColorMap& operator=(const ColorMap& rhs);


    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
      ar& BOOST_SERIALIZATION_NVP(m_min);
      ar& BOOST_SERIALIZATION_NVP(m_max);
      ar& BOOST_SERIALIZATION_NVP(m_breakpoints);
    }
    float m_min;
    float m_max;

    std::map<ElVisFloat, ColorMapBreakpoint> m_breakpoints;
  };
}

#endif // COLOR_MAP_H

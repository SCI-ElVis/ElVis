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

#ifndef ELVIS_ELVIS_NATIVE_LIGHT_H
#define ELVIS_ELVIS_NATIVE_LIGHT_H

#include <ElVis/Core/Color.h>
#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/Point.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/split_member.hpp>

namespace ElVis
{
  class Light
  {
  public:
    ELVIS_EXPORT Light();
//    ELVIS_EXPORT Light(const Light& rhs);
    ELVIS_EXPORT virtual ~Light() {}

    ELVIS_EXPORT const Color& GetColor() const { return m_color; }
    ELVIS_EXPORT void SetColor(const Color& rhs) { m_color = rhs; }

    ELVIS_EXPORT const WorldPoint& Position() const { return m_position; }
    ELVIS_EXPORT void SetPosition(const WorldPoint& value)
    {
      m_position = value;
    }

//    ELVIS_EXPORT bool IsDirectional() const { return m_isDirectionalLight; }
//    ELVIS_EXPORT void SetIsDirectional(bool value)
//    {
//      m_isDirectionalLight = value;
//    }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
      ar& BOOST_SERIALIZATION_NVP(m_color);
      ar& BOOST_SERIALIZATION_NVP(m_position);
      ar& BOOST_SERIALIZATION_NVP(m_isDirectionalLight);
    }

  private:
    Light& operator=(const Light& rhs);

    Color m_color;
    WorldPoint m_position;
    bool m_isDirectionalLight;
  };
}

#endif // ELVIS_ELVIS_NATIVE_LIGHT_H

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

#ifndef ELVIS_ELVIS_CORE_COLOR_H
#define ELVIS_ELVIS_CORE_COLOR_H

#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/Float.h>

#include <boost/signals2.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/split_member.hpp>


namespace ElVis
{
    /// \brief A color in RGBA space.
    class Color
    {
        public:
            boost::signals2::signal<void (const Color&)> OnColorChanged;

        public:
            ELVIS_EXPORT Color();

            template<typename T>
            Color(T r, T g, T b, T alpha) :
                OnColorChanged(),
                m_red(static_cast<float>(r)),
                m_green(static_cast<float>(g)),
                m_blue(static_cast<float>(b)),
                m_alpha(static_cast<float>(alpha))
            {
            }

            template<typename T>
            Color(T r, T g, T b) :
                OnColorChanged(),
                m_red(static_cast<float>(r)),
                m_green(static_cast<float>(g)),
                m_blue(static_cast<float>(b)),
                m_alpha(1.0f)
            {
            }

            ELVIS_EXPORT Color(const Color& rhs);
            ELVIS_EXPORT Color& operator=(const Color& rhs);

            ELVIS_EXPORT float Red() const;
            ELVIS_EXPORT float Green() const;
            ELVIS_EXPORT float Blue() const;
            ELVIS_EXPORT float Alpha() const;

            ELVIS_EXPORT unsigned int RedAsInt() const;
            ELVIS_EXPORT unsigned int GreenAsInt() const;
            ELVIS_EXPORT unsigned int BlueAsInt() const;
            ELVIS_EXPORT unsigned int AlphaAsInt() const;

            ELVIS_EXPORT void SetRed(double value);
            ELVIS_EXPORT void SetGreen(double value);
            ELVIS_EXPORT void SetBlue(double value);
            ELVIS_EXPORT void SetAlpha(double value);

            ELVIS_EXPORT void SetRed(int value);
            ELVIS_EXPORT void SetGreen(int value);
            ELVIS_EXPORT void SetBlue(int value);
            ELVIS_EXPORT void SetAlpha(int value);

            template<typename Archive>
            void NotifyLoad(Archive& ar, const unsigned int version, 
                typename boost::enable_if<typename Archive::is_saving>::type* p = 0)
            {
            }

            template<typename Archive>
            void NotifyLoad(Archive& ar, const unsigned int version, 
                typename boost::enable_if<typename Archive::is_loading>::type* p = 0)
            {
                OnColorChanged(*this);
            }

            template<typename Archive>
            void serialize(Archive& ar, const unsigned int version)
            {
                ar & BOOST_SERIALIZATION_NVP(m_red);
                ar & BOOST_SERIALIZATION_NVP(m_green);
                ar & BOOST_SERIALIZATION_NVP(m_blue);
                ar & BOOST_SERIALIZATION_NVP(m_alpha);
                NotifyLoad(ar, version);
            }

        private:
            float m_red;
            float m_green;
            float m_blue;
            float m_alpha;
    };

    ELVIS_EXPORT bool operator==(const Color& lhs, const Color& rhs);
    ELVIS_EXPORT bool operator!=(const Color& lhs, const Color& rhs);

    ELVIS_EXPORT Color operator+(const Color& lhs, const Color& rhs);
    ELVIS_EXPORT Color operator-(const Color& lhs, const Color& rhs);
    ELVIS_EXPORT Color operator*(const Color& lhs, const ElVisFloat& s);
    ELVIS_EXPORT Color operator*(const ElVisFloat& s, const Color& rhs);

}

#endif //ELVIS_ELVIS_CORE_COLOR_H

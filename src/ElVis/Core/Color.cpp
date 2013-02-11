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


#include <ElVis/Core/Color.h>

namespace ElVis
{
    Color::Color() :
      OnColorChanged(),
      m_red(0.0f),
      m_green(0.0f),
      m_blue(0.0f),
      m_alpha(0.0f)
    {
    }


    Color::Color(const Color& rhs) :
        OnColorChanged(),
        m_red(rhs.m_red),
        m_green(rhs.m_green),
        m_blue(rhs.m_blue),
        m_alpha(rhs.m_alpha)
    {

    }

    Color& Color::operator=(const Color& rhs)
    {
        m_red = rhs.m_red;
        m_green = rhs.m_green;
        m_blue = rhs.m_blue;
        m_alpha = rhs.m_alpha;
        OnColorChanged(*this);
        return *this;
    }

    float Color::Red() const { return m_red; }
    float Color::Green() const { return m_green; }
    float Color::Blue() const { return m_blue; }
    float Color::Alpha() const { return m_alpha; }

    unsigned int Color::RedAsInt() const { return static_cast<unsigned int>(m_red*255); }
    unsigned int Color::GreenAsInt() const { return static_cast<unsigned int>(m_green*255); }
    unsigned int Color::BlueAsInt() const { return static_cast<unsigned int>(m_blue*255); }
    unsigned int Color::AlphaAsInt() const { return static_cast<unsigned int>(m_alpha*255); }

    void Color::SetRed(double value) { m_red = static_cast<float>(value); OnColorChanged(*this); }
    void Color::SetGreen(double value) { m_green = static_cast<float>(value); OnColorChanged(*this); }
    void Color::SetBlue(double value) { m_blue = static_cast<float>(value); OnColorChanged(*this); }
    void Color::SetAlpha(double value) { m_alpha = static_cast<float>(value); OnColorChanged(*this);}

    void Color::SetRed(int value) { m_red = (float)value/255.0; OnColorChanged(*this);}
    void Color::SetGreen(int value) { m_green = (float)value/255.0; OnColorChanged(*this);}
    void Color::SetBlue(int value) { m_blue = (float)value/255.0; OnColorChanged(*this);}
    void Color::SetAlpha(int value) { m_alpha = (float)value/255.0; OnColorChanged(*this);}

    Color operator+(const Color& lhs, const Color& rhs)
    {
        Color result(lhs.Red() + rhs.Red(),
                     lhs.Green() + rhs.Green(),
                     lhs.Blue() + rhs.Blue(),
                     lhs.Alpha() + rhs.Alpha());
        return result;
    }

    Color operator-(const Color& lhs, const Color& rhs)
    {
        Color result(lhs.Red() - rhs.Red(),
                     lhs.Green() - rhs.Green(),
                     lhs.Blue() - rhs.Blue(),
                     lhs.Alpha() - rhs.Alpha());
        return result;
    }

    Color operator*(const Color& lhs, const ElVisFloat& s)
    {
        Color result(lhs.Red() *s ,
                     lhs.Green() *s,
                     lhs.Blue() *s,
                     lhs.Alpha() *s);
        return result;
    }

    Color operator*(const ElVisFloat& s, const Color& rhs)
    {
        return rhs*s;
    }

    bool operator==(const Color& lhs, const Color& rhs)
    {
        return lhs.Red() == rhs.Red() &&
                lhs.Green() == rhs.Green() &&
                lhs.Blue() == rhs.Blue() &&
                lhs.Alpha() == rhs.Alpha();
    }

    bool operator!=(const Color& lhs, const Color& rhs)
    {
        return !(lhs == rhs);
    }
}

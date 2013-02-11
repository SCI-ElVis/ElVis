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

#ifndef ELVIS_INTERVAL_POINT_CU
#define ELVIS_INTERVAL_POINT_CU

#include <ElVis/Core/Cuda.h>
#include <ElVis/Core/Float.cu>
#include <ElVis/Core/Interval.hpp>

struct IntervalPoint
{
    ELVIS_DEVICE IntervalPoint() {}

    ELVIS_DEVICE IntervalPoint(const ElVis::Interval<ElVisFloat>& xParam,
                               const ElVis::Interval<ElVisFloat>& yParam,
                               const ElVis::Interval<ElVisFloat>& zParam) :
        x(xParam),
        y(yParam),
        z(zParam)
    {
    }

    ELVIS_DEVICE
    IntervalPoint(const ElVisFloat3& p0, const ElVisFloat3& p1) 
    :
        x(fminf(p0.x, p1.x), fmaxf(p0.x, p1.x)),
        y(fminf(p0.y, p1.y), fmaxf(p0.y, p1.y)),
        z(fminf(p0.z, p1.z), fmaxf(p0.z, p1.z))
    {
    }

    ELVIS_DEVICE
    IntervalPoint(const IntervalPoint& rhs) :
        x(rhs.x),
        y(rhs.y),
        z(rhs.z)
    {
    }

    ELVIS_DEVICE
    IntervalPoint& operator=(const IntervalPoint& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        return *this;
    }

    ELVIS_DEVICE bool IsEmpty() const
    {
        bool result = x.IsEmpty();
        result |= y.IsEmpty();
        result |= z.IsEmpty();
        return result;
    }

    ELVIS_DEVICE ElVisFloat3 GetMidpoint() const
    {
        ElVisFloat3 result;
        result.x = x.GetMidpoint();
        result.y = y.GetMidpoint();
        result.z = z.GetMidpoint();
        return result;
    }

    ELVIS_DEVICE double GetWidth()
    {
        if( IsEmpty() )
        {
            return 0.0;
        }
        else
        {
            return fmaxf(x.GetWidth(), fmaxf(y.GetWidth(), z.GetWidth()));
        }
    }


    ELVIS_DEVICE const ElVis::Interval<ElVisFloat>& operator[](int index) const
    {
        if( index == 0 ) return x;
        if( index == 1 ) return y;
        return z;
    }

    ELVIS_DEVICE ElVis::Interval<ElVisFloat>& operator[](int index)
    {
        if( index == 0 ) return x;
        if( index == 1 ) return y;
        return z;
    }

    ElVis::Interval<ElVisFloat> x;
    ElVis::Interval<ElVisFloat> y;
    ElVis::Interval<ElVisFloat> z;
};

ELVIS_DEVICE bool Subset(const IntervalPoint& a, const IntervalPoint& b)
{
    return Subset(a.x, b.x) && Subset(a.y, b.y) && Subset(a.z, b.z);
}

ELVIS_DEVICE IntervalPoint operator+(const IntervalPoint& lhs, const IntervalPoint& rhs)
{
    IntervalPoint result(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z+rhs.z);
    return result;
}

ELVIS_DEVICE IntervalPoint operator+(const ElVisFloat3& lhs, const IntervalPoint& rhs)
{
    IntervalPoint result(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z+rhs.z);
    return result;
}

ELVIS_DEVICE IntervalPoint operator+(const IntervalPoint& lhs, const ElVisFloat3& rhs)
{
    IntervalPoint result(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z+rhs.z);
    return result;
}

ELVIS_DEVICE IntervalPoint operator-(const IntervalPoint& lhs, const IntervalPoint& rhs)
{
    IntervalPoint result(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
    return result;
}

ELVIS_DEVICE IntervalPoint operator-(const IntervalPoint& lhs, const ElVisFloat3& rhs)
{
    IntervalPoint result(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
    return result;
}

ELVIS_DEVICE IntervalPoint Intersection(const IntervalPoint& lhs, const IntervalPoint& rhs)
{
    IntervalPoint result(Intersection(lhs.x, rhs.x), Intersection(lhs.y, rhs.y), Intersection(lhs.z, rhs.z));
    return result;
}

#endif

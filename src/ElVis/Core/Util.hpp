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

#ifndef ELVIS_UTIL_HPP
#define ELVIS_UTIL_HPP

#include <optix_math.h>
#include <optixu/optixu_matrix.h>

#include <ElVis/Core/Point.hpp>
#include <ElVis/Core/Vector.hpp>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/ElVisDeclspec.h>

namespace ElVis
{
    ELVIS_EXPORT void Min(WorldPoint& lhs, const WorldPoint& rhs);
    ELVIS_EXPORT void Max(WorldPoint& lhs, const WorldPoint& rhs);

    ELVIS_EXPORT ElVisFloat3 MakeFloat3(const WorldPoint& p);
    ELVIS_EXPORT ElVisFloat3 MakeFloat3(const WorldVector& v);
    ELVIS_EXPORT ElVisFloat4 MakeFloat4(const WorldPoint& p);
    ELVIS_EXPORT ElVisFloat4 MakeFloat4(const WorldVector& v);

    inline void noDelete(void* ptr) {}
}


#endif //UTIL_HPP



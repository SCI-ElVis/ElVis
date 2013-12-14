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

#ifndef ELVIS_CORE_JACOBI_FACE_H
#define ELVIS_CORE_JACOBI_FACE_H

#include <ElVis/Core/Point.hpp>
#include <ElVis/Core/Vector.hpp>
#include <ElVis/Core/FaceInfo.h>
#include <algorithm>

namespace ElVis
{
    namespace JacobiExtension
    {
        bool closePointLessThan(const WorldPoint& lhs, const WorldPoint& rhs);
        bool closePointEqual(const WorldPoint& lhs, const WorldPoint& rhs);
        struct JacobiFaceKey
        {
            JacobiFaceKey(const WorldPoint& point0, const WorldPoint& point1, const WorldPoint& point2, const WorldPoint& point3) 
            {
                p[0] = point0;
                p[1] = point1;
                p[2] = point2;
                p[3] = point3;

                for(int i = 0; i < 4; ++i)
                {
                    sorted[i] = p[i];
                }

                std::sort(sorted, sorted+4, closePointLessThan);
            }

            JacobiFaceKey(const JacobiFaceKey& rhs)
            {
                for(int i = 0; i < 4; ++i)
                {
                    p[i] = rhs.p[i];
                }

                for(int i = 0; i < 4; ++i)
                {
                    sorted[i] = rhs.sorted[i];
                }
            }

            JacobiFaceKey& operator=(const JacobiFaceKey& rhs)
            {
                for(int i = 0; i < 4; ++i)
                {
                    p[i] = rhs.p[i];
                    sorted[i] = rhs.sorted[i];
                }
                return *this;
            }

            WorldPoint MinExtent() const;
            WorldPoint MaxExtent() const;

            int NumVertices() const;

            WorldPoint p[4];
            WorldPoint sorted[4];
        };

        // Temporary face structure to find unique faces among all elements.
        struct JacobiFace
        {
            explicit JacobiFace(const WorldVector& n) :
                normal(n),
                info(),
                planarInfo()
            {

            }

            JacobiFace(const JacobiFace& rhs) :
              normal(rhs.normal),
              info(rhs.info),
              planarInfo(rhs.planarInfo)
            {
            }

            JacobiFace& operator=(const JacobiFace& rhs)
            {
                normal = rhs.normal;
                info = rhs.info;
                planarInfo = rhs.planarInfo;
                return *this;
            }

            //WorldPoint MinExtent() const;
            //WorldPoint MaxExtent() const;

            //int NumVertices() const;

            WorldVector normal;
            FaceInfo info;
            PlanarFaceInfo planarInfo;
        };

        bool operator<(const JacobiFaceKey& lhs, const JacobiFaceKey& rhs);
    }
}


#endif

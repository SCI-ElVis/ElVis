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

#ifndef ELVIS_CORE_FACEINFO_H
#define ELVIS_CORE_FACEINFO_H

#include <ElVis/Core/ElementId.h>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/Point.hpp>
#include <ElVis/Core/Float.h>

namespace ElVis
{
    enum FaceType
    {
        eCurved,
        ePlanar
    };

    struct FaceInfo
    {
        // CommonElements[0] - The element on the opposite side as the face's normal.
        // CommonElements[1] - The element on the same side of the face's normal.
        ElementId CommonElements[2];
        FaceType Type;
        ElVisFloat3 MinExtent;
        ElVisFloat3 MaxExtent;

        void widenExtents()
        {
          if( MinExtent.x == MaxExtent.x )
          {
            MinExtent.x = MinExtent.x - static_cast<ElVisFloat>(.0001);
            MaxExtent.x = MaxExtent.x + static_cast<ElVisFloat>(.0001);
          }

          if( MinExtent.y == MaxExtent.y )
          {
            MinExtent.y = MinExtent.y - static_cast<ElVisFloat>(.0001);
            MaxExtent.y = MaxExtent.y + static_cast<ElVisFloat>(.0001);
          }

          if( MinExtent.z == MaxExtent.z )
          {
            MinExtent.z = MinExtent.z - static_cast<ElVisFloat>(.0001);
            MaxExtent.z = MaxExtent.z + static_cast<ElVisFloat>(.0001);
          }
        }
    };

    inline bool isPlanarFace(const FaceInfo& info)
    {
      return info.Type == ePlanar;
    }

    inline bool isCurvedFace(const FaceInfo& info)
    {
      return info.Type == eCurved;
    }

    enum TwoDElementType
    {
      eTriangle,
      eQuad
    };

    struct PlanarFaceInfo
    {
      TwoDElementType Type;
      unsigned int vertexIdx[4];
    };
}

#endif

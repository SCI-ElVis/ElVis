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

#include <ElVis/Extensions/JacobiExtension/JacobiFace.h>

namespace ElVis
{
    namespace JacobiExtension
    {
        bool closePointEqual(const WorldPoint& lhs, const WorldPoint& rhs)
        {
          return !closePointLessThan(lhs, rhs) &&
            !closePointLessThan(rhs, lhs);
        }

        WorldPoint JacobiFaceKey::MinExtent() const
        {
            return CalcMin(p[0], CalcMin(p[1], CalcMin(p[2], p[3])));
        }

        WorldPoint JacobiFaceKey::MaxExtent() const
        {
            return CalcMax(p[0], CalcMax(p[1], CalcMax(p[2], p[3])));
        }

        int JacobiFaceKey::NumVertices() const
        {
          if( !closePointLessThan(sorted[2], sorted[3]) &&
              !closePointLessThan(sorted[3], sorted[2]) )
          {
            return 3;
          }
          return 4;
        }

        bool operator<(const JacobiFaceKey& lhs, const JacobiFaceKey& rhs)
        {
            if( lhs.NumVertices() != rhs.NumVertices() )
            {
                return lhs.NumVertices() < rhs.NumVertices();
            }

            for(int i = 0; i < 4; ++i)
            {
                if( closePointLessThan(lhs.sorted[i], rhs.sorted[i]) ) return true;
                if( closePointLessThan(rhs.sorted[i], lhs.sorted[i]) ) return false;
            }
            return false;

        }
    }
}
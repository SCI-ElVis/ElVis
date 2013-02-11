////////////////////////////////////////////////////////////////////////////////
//
//  File: hoEdge.cpp
//
//
//  The MIT License
//
//  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
//  Department of Aeronautics, Imperial College London (UK), and Scientific
//  Computing and Imaging Institute, University of Utah (USA).
//
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
//  Description:
//
//
////////////////////////////////////////////////////////////////////////////////

#include <ElVis/Extensions/JacobiExtension/Isosurface.h>
#include <ElVis/Extensions/JacobiExtension/Edge.h>

namespace ElVis
{
    namespace JacobiExtension
    {
        bool operator<(const Edge& lhs, const Edge& rhs)
        {
            unsigned int lhsVertices[] = { lhs.GetVertex0(), lhs.GetVertex1() };
            unsigned int rhsVertices[] = { rhs.GetVertex0(), rhs.GetVertex1() };

            if( lhsVertices[1] < lhsVertices[0] ) std::swap(lhsVertices[0], lhsVertices[1]);
            if( rhsVertices[1] < rhsVertices[0] ) std::swap(rhsVertices[0], rhsVertices[1]);

            if( lhsVertices[0] < rhsVertices[0]) return true;
            if( lhsVertices[0] > rhsVertices[0]) return false;

            return lhsVertices[1] < rhsVertices[1];
        }
    }
}



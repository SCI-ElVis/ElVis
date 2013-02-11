////////////////////////////////////////////////////////////////////////////////
//
//  File: hoEdge.h
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
//  Defines an edge of a finite element after it has
//  been projected onto an arbitrary plane.
//
//  Edge is the line segment between the two end WorldPoints,
//  and the coordinates of the two end WorldPoints are in
//  (u,v), where (u,v) are orthonormal and define the plane.
//
////////////////////////////////////////////////////////////////////////////////


#ifndef ELVIS_JACOBI_EXTENSION__EDGE_H_
#define ELVIS_JACOBI_EXTENSION__EDGE_H_

#include <ElVis/Core/Point.hpp>
#include <ElVis/Extensions/JacobiExtension/Declspec.h>

namespace ElVis
{
    namespace JacobiExtension
    {
        class Edge
        {
        public:
            JACOBI_EXTENSION_EXPORT Edge() :
              m_vertexIndex0(0),
                  m_vertexIndex1(0)
              {
              }

              JACOBI_EXTENSION_EXPORT Edge(unsigned int v0, unsigned int v1)
                  : m_vertexIndex0(v0),
                  m_vertexIndex1(v1)
              {
              }

              ~Edge() {}

              JACOBI_EXTENSION_EXPORT unsigned int GetVertex0() const { return m_vertexIndex0; }
              JACOBI_EXTENSION_EXPORT unsigned int GetVertex1() const { return m_vertexIndex1; }

        private:
            unsigned int m_vertexIndex0;
            unsigned int m_vertexIndex1;
        };

        JACOBI_EXTENSION_EXPORT bool operator<(const Edge& lhs, const Edge& rhs);

    }
}

#endif

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

#ifndef ELVIS_JACOBI_EXTENSION_CUDA_HEXAHEDRON_CU
#define ELVIS_JACOBI_EXTENSION_CUDA_HEXAHEDRON_CU

#include <ElVis/Core/matrix.cu>
#include <ElVis/Core/util.cu>
#include <ElVis/Core/jacobi.cu>
#include <ElVis/Extensions/JacobiExtension/HexahedronCommon.cu>

//// The vertices associated with this hex.
__device__ ElVisFloat4* HexVertexBuffer;

//
//// Hexvertex_face_index[i] gives the index for the four 
//// vertices associated with face i.
__device__ uint4* Hexvertex_face_index;
//
//// Defines the planes for each hex side.
__device__ ElVisFloat4* HexPlaneBuffer;
//
__device__ ElVisFloat* HexCoefficients;
__device__ int* HexCoefficientIndices;

__device__ uint3* HexDegrees;

__device__ unsigned int HexNumElements;




//__device__ bool IntersectsFace(int hexId, unsigned int faceNumber,
//                               ElVisFloat4* p, const ElVisFloat3& origin, const ElVisFloat3& direction,
//                               ElVisFloat& t)
//{
//    uint4 index = Hexvertex_face_index[faceNumber];
//    bool result = false;
//    if( ContainsOrigin(p[index.x], p[index.y], p[index.z], p[index.w]) )
//    {
//        result = FindPlaneIntersection(origin, direction, GetPlane(HexPlaneBuffer, hexId, faceNumber), t);
//    }
     
//    return result;
//}






#endif

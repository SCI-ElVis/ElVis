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

#ifndef ELVIS_EXTENSIONS_JACOBI_EXTENSION_CUDA_PRISM_CU
#define ELVIS_EXTENSIONS_JACOBI_EXTENSION_CUDA_PRISM_CU

#include <ElVis/Core/Float.cu>
#include <ElVis/Core/Interval.hpp>
#include <ElVis/Core/IntervalPoint.cu>
#include <ElVis/Extensions/JacobiExtension/PrismCommon.cu>

__device__ ElVisFloat4* PrismVertexBuffer;


__device__ uint4* Prismvertex_face_index;
//
//// Defines the planes for each hex side.
__device__ ElVisFloat4* PrismPlaneBuffer;
//
__device__ ElVisFloat* PrismCoefficients;
__device__ int* PrismCoefficientIndices;

__device__ uint3* PrismDegrees;

__device__ unsigned int PrismNumElements;






#endif

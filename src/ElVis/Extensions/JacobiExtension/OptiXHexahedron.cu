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

#ifndef ELVIS_EXTENSIONS_JACOBI_EXTENSION_OPTIX_HEXAHEDRON_CU
#define ELVIS_EXTENSIONS_JACOBI_EXTENSION_OPTIX_HEXAHEDRON_CU

#include <optix_cuda.h>
#include <optix_math.h>
#include <ElVis/Core/matrix.cu>
#include <optixu/optixu_aabb.h>
//#include "matrix.cu"
#include <ElVis/Core/CutSurfacePayloads.cu>
#include <ElVis/Core/VolumeRenderingPayload.cu>
#include <ElVis/Core/typedefs.cu>
#include <ElVis/Core/jacobi.cu>
#include <ElVis/Core/util.cu>
#include <ElVis/Core/OptixVariables.cu>
#include <ElVis/Core/Interval.hpp>
#include <ElVis/Core/IntervalPoint.cu>

#include <ElVis/Extensions/JacobiExtension/HexahedronCommon.cu>

// The vertices associated with this hex.
rtBuffer<ElVisFloat4> HexVertexBuffer;

// Hexvertex_face_index[i] gives the index for the four 
// vertices associated with face i.
rtBuffer<uint4> Hexvertex_face_index;

// Defines the planes for each hex side.
rtBuffer<ElVisFloat4> HexPlaneBuffer;

rtBuffer<ElVisFloat> HexCoefficients;
rtBuffer<uint> HexCoefficientIndices;

rtBuffer<uint3> HexDegrees;


rtDeclareVariable(int, intersectedHexId, attribute IntersectedHex, );


#endif

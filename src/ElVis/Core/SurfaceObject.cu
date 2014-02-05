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

#ifndef ELVIS_SURFACE_OBJECT_CU
#define ELVIS_SURFACE_OBJECT_CU

#include <optix_cuda.h>
#include <optix_math.h>
#include "CutSurfacePayloads.cu"
#include "ConvertToColor.cu"
#include <ElVis/Core/DiffuseLighting.cu>
#include <ElVis/Core/Float.cu>
#include <ElVis/Core/OptixVariables.cu>


RT_PROGRAM void SurfaceObjectClosestHit()
{
    payload.isValid = true;

    ElVisFloat3 color = MakeFloat3(MAKE_FLOAT(1.0), MAKE_FLOAT(1.0), MAKE_FLOAT(1.0));
    payload.result = color;
    payload.Normal = normal;
    payload.Color = color;
    payload.IntersectionPoint = MakeFloat3(ray.origin) + closest_t * MakeFloat3(ray.direction);
    payload.IntersectionT = closest_t;

}


#endif

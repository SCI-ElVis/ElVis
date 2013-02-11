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

#ifndef ELVIS_OPENGL_LIGHTING_CU
#define ELVIS_OPENGL_LIGHTING_CU

#include <optix_cuda.h>
#include <optix_math.h>
#include <optixu/optixu_matrix.h>
#include <optixu/optixu_aabb.h>
#include "CutSurfacePayloads.cu"
#include "ConvertToColor.cu"
#include <ElVis/Core/DiffuseLighting.cu>
#include <ElVis/Core/Float.cu>
#include <ElVis/Core/OptixVariables.cu>

RT_PROGRAM void OpenGLLighting()
{
    ElVisFloat3 normal = normal_buffer[launch_index];
    
    if( normal.x != MAKE_FLOAT(0.0) ||
        normal.y != MAKE_FLOAT(0.0) ||
        normal.z != MAKE_FLOAT(0.0) )
    {
        
        const ElVisFloat3 intersectionPoint = intersection_buffer[launch_index];
        const ElVisFloat3 color = raw_color_buffer[launch_index];
		
        normal = normalize(normal);    
        ElVisFloat3 resultColor = DiffuseLighting(color, normal, intersectionPoint);

        ElVisFloat3 clamped = clamp(resultColor, MAKE_FLOAT(0.0), MAKE_FLOAT(1.0));
        color_buffer[launch_index] = ConvertToColor(clamped);
    }
}

#endif

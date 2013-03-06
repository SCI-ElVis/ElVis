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

#ifndef ELVIS_PRIMARY_RAY_GENERATOR_CU
#define ELVIS_PRIMARY_RAY_GENERATOR_CU

#include <optix_cuda.h>
#include <optix_math.h>
#include <optixu/optixu_matrix.h>
#include <optixu/optixu_aabb.h>
#include <ElVis/Core/Float.cu>
#include <ElVis/Core/matrix.cu>
#include <ElVis/Core/OptixVariables.cu>
#include <ElVis/Core/SceneViewProjection.h>

rtDeclareVariable(ElVis::SceneViewProjection , ProjectionType, , );

rtDeclareVariable(ElVisFloat3, eye, , );
rtDeclareVariable(ElVisFloat3, U, , );
rtDeclareVariable(ElVisFloat3, V, , );
rtDeclareVariable(ElVisFloat3, W, , );

// Rays generated here have origin in the lower left corner.
__device__ __forceinline__ optix::Ray GeneratePrimaryRay(const optix::size_t2& screen, unsigned int rayTypeId, float tolerance)
{
    ElVisFloat2 d = MakeFloat2(launch_index) / MakeFloat2(screen) * MAKE_FLOAT(2.0) - MAKE_FLOAT(1.0);

    if( ProjectionType == ElVis::ePerspective )
    {
        ELVIS_PRINTF("GeneratePrimaryRay: Perspective projection.\n");
        ElVisFloat3 ray_origin = eye;
        ElVisFloat3 ray_direction = normalize(d.x*U + d.y*V + W);

        optix::Ray ray = optix::make_Ray( ConvertToFloat3(ray_origin), ConvertToFloat3(ray_direction), rayTypeId, tolerance, RT_DEFAULT_MAX);

        //ELVIS_PRINTF("Screen (%d, %d)\n", screen.x, screen.y);
        //ELVIS_PRINTF("d (%f, %f)\n", d.x, d.y);
        //ELVIS_PRINTF("U (%f, %f, %f)\n", U.x, U.y, U.z);
        //ELVIS_PRINTF("V (%f, %f, %f)\n", V.x, V.y, V.z);
        //ELVIS_PRINTF("W (%f, %f, %f)\n", W.x, W.y, W.z);

        return ray;
    }
    else
    {
        ELVIS_PRINTF("GeneratePrimaryRay: Orthographic projection.\n");
        ElVisFloat3 newEye = eye + near*W + d.x*U + d.y*V;
        ElVisFloat3 dir = normalize(W);
        optix::Ray ray = optix::make_Ray(ConvertToFloat3(newEye), ConvertToFloat3(dir), rayTypeId, tolerance, RT_DEFAULT_MAX);
        return ray;
    }

}

#endif

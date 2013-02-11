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

#ifndef ELVIS_DIFFUSE_LIGHTING_CU
#define ELVIS_DIFFUSE_LIGHTING_CU

#include <optix_cuda.h>
#include <optix_math.h>
#include <optixu/optixu_matrix.h>
#include <optixu/optixu_aabb.h>
#include "CutSurfacePayloads.cu"
#include "ConvertToColor.cu"
#include <ElVis/Core/Float.cu>

rtDeclareVariable(float3, ambientColor, , );
rtBuffer<float, 1> lightPosition;
rtBuffer<float, 1> lightColor;

__device__ __forceinline__ ElVisFloat3 CalculateContribution(const ElVisFloat3& lightPos, const ElVisFloat3& intersectionPoint, const ElVisFloat3& lcolor, const ElVisFloat3& normal, const ElVisFloat3& color)
{

    ELVIS_PRINTF("LightPos: (%2.15f, %2.15f, %2.15f), Intersection Point (%2.15f, %2.15f, %2.15f), Light Color (%2.15f, %2.15f, %2.15f), Normal (%2.15f, %2.15f, %2.15f)\n",
                 lightPos.x, lightPos.y, lightPos.z,
                 intersectionPoint.x, intersectionPoint.y, intersectionPoint.z,
                 lcolor.x, lcolor.y, lcolor.z,
                 normal.x, normal.y, normal.z
                 );
    \
    ElVisFloat3 vectorToLight = lightPos - intersectionPoint;
    vectorToLight = normalize(vectorToLight);
    ELVIS_PRINTF("Vector to light: (%2.15f, %2.15f, %2.15f)\n", vectorToLight.x, vectorToLight.y, vectorToLight.z);

    ElVisFloat d = dot(vectorToLight, normal);

    //d = max(MAKE_FLOAT(0.0), d);
    if( d < MAKE_FLOAT(0.0) )
    {
        d = -d;
    }
    ELVIS_PRINTF("d %2.15f\n", d);

    //ElVisFloat3 lcolor = MakeFloat3(lightColor[i], lightColor[i+1], lightColor[i+2]);
    ElVisFloat3 contribution = d*lcolor*color;
    return contribution;
}

// color - the color at the intersection point.
// normal - the unit normal at the intersection point.
__device__ __forceinline__ ElVisFloat3 DiffuseLighting(const ElVisFloat3& color, const ElVisFloat3& normal, const ElVisFloat3& intersectionPoint)
{
    // Depends on lighting information in the Scene.
    ElVisFloat3 resultColor = MakeFloat3(ambientColor)*color;
    ElVisFloat3 normalizedNormal = normalize(normal);

    int numLights = lightPosition.size()/3;
    ELVIS_PRINTF("Num Lights %d", numLights);
    for(int i = 0; i < numLights; i+=3)
    {
        // Just simple diffuse.
        ElVisFloat3 lightPos = MakeFloat3(lightPosition[i], lightPosition[i+1], lightPosition[i+2]);
        ElVisFloat3 lcolor = MakeFloat3(lightColor[i], lightColor[i+1], lightColor[i+2]);
        resultColor += CalculateContribution(lightPos, intersectionPoint, lcolor, normal, color);

//        ElVisFloat3 vectorToLight = lightPos - intersectionPoint;
//        vectorToLight = normalize(vectorToLight);
//        ElVisFloat d = dot(vectorToLight, normal);

//        d = max(MAKE_FLOAT(0.0), d);
//        ElVisFloat3 lcolor = MakeFloat3(lightColor[i], lightColor[i+1], lightColor[i+2]);
//        ElVisFloat3 contribution = d*lcolor*color;
//        resultColor += contribution;
    }

    ELVIS_PRINTF("Color eye (%f, %f, %f), normal (%f, %f, %f)\n", eye.x, eye.y, eye.z, normal.x, normal.y, normal.z);
    resultColor += CalculateContribution(eye, intersectionPoint, HeadlightColor, normalizedNormal, color);
    ELVIS_PRINTF("Result color (%f, %f, %f)\n", resultColor.x, resultColor.y, resultColor.z);
    return resultColor;
}

#endif //ELVIS_DIFFUSE_LIGHTING_CU

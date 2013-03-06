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

#include <optix_cuda.h>
#include <optix_math.h>
#include <optixu/optixu_matrix.h>
#include <optixu/optixu_aabb.h>
#include <ElVis/Core/DiffuseLighting.cu>
#include <ElVis/Core/OptixVariables.cu>


rtDeclareVariable(ElVisFloat4, PlaneNormal, , );
rtDeclareVariable(ElVisFloat3, PlanePoint, , );


RT_PROGRAM void Plane_intersect( int primIdx )
{
    ElVisFloat t;
    FindPlaneIntersection(ray, PlaneNormal, t);

    ELVIS_PRINTF("Eye (%f, %f, %f), Dir (%f, %f, %f)\n",
        ray.origin.x, ray.origin.y, ray.origin.z,
        ray.direction.x, ray.direction.y, ray.direction.z);
    ELVIS_PRINTF("Plane intersection with normal (%f, %f, %f, %f) and point (%f, %f, %f) is %f\n",
        PlaneNormal.x,
        PlaneNormal.y,
        PlaneNormal.z,
        PlaneNormal.w,
        PlanePoint.x,
        PlanePoint.y,
        PlanePoint.z,
        t);

    if(  rtPotentialIntersection( t ) )
    {
        ELVIS_PRINTF("Plane_intersect: Closest intersection found at %f\n", t);
        normal.x = PlaneNormal.x;
        normal.y = PlaneNormal.y;
        normal.z = PlaneNormal.z;
        rtReportIntersection(0);
    }
}

RT_PROGRAM void Plane_bounding (int, float result[6])
{
    ELVIS_PRINTF("Plane Bounding.\n");
    optix::Aabb* aabb = (optix::Aabb*)result;
//    ElVisFloat3 u, v;
//    ElVisFloat3 w = MakeFloat3(Plane);
//    GenerateUVWCoordinateSystem(w, u, v);
    //aabb->m_min = fminf( fminf( ConvertToFloat3(p0), ConvertToFloat3(p1)), ConvertToFloat3(p2) );
    //aabb->m_max = fmaxf( fmaxf( ConvertToFloat3(p0), ConvertToFloat3(p1)), ConvertToFloat3(p2) );
    aabb->m_min = make_float3(-100000.0f, -100000.0f, -100000.0f);
    aabb->m_max = make_float3(100000.0f, 100000.0f, 100000.0f);
}


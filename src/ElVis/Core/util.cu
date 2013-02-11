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

#ifndef ELVISNATIVE_UTIL_CU
#define ELVISNATIVE_UTIL_CU

#include <optix_cuda.h>
#include <optix_math.h>
#include <optixu/optixu_matrix.h>
#include <ElVis/Core/Float.cu>
#include <ElVis/Core/typedefs.cu>
#include <ElVis/Core/Printf.cu>

ELVIS_DEVICE bool PointInBox(const ElVisFloat3& p, const ElVisFloat3& p0, const ElVisFloat3& p1)
{
    bool result = (p.x >= p0.x);
    result &= (p.y >= p0.y);
    result &= (p.z >= p0.z);
    result &= (p.x <= p1.x);
    result &= (p.y <= p1.y);
    result &= (p.z <= p1.z);
    return result;
}

ELVIS_DEVICE ElVisFloat4 CalculatePlaneEquation(const ElVisFloat3& v0, const ElVisFloat3& v1, const ElVisFloat3& v2)
{
    ElVisFloat4 result;
    ElVisFloat3 normal = cross((v2-v1), (v0-v1));
    normal = normalize(normal);
    result.w = -(normal.x*v1.x + normal.y*v1.y + normal.z*v1.z);
    result.x = normal.x;
    result.y = normal.y;
    result.z = normal.z;
    return result;
}

__device__ __forceinline__ bool FindPlaneIntersection(const ElVisFloat3& origin, const ElVisFloat3& direction, const ElVisFloat4& plane, ElVisFloat& t)
{
    ElVisFloat3 normal = MakeFloat3(plane);
    ElVisFloat d = plane.w;

    ElVisFloat vd = dot(direction, normal);

    if( vd == MAKE_FLOAT(0.0) ) return false;

    ElVisFloat v0 = -(dot(origin, normal) + d);
    t = v0/vd;

    return (t > MAKE_FLOAT(0.0));
}

__device__ __forceinline__ bool FindPlaneIntersection(optix::Ray ray, const ElVisFloat4& plane, ElVisFloat& t)
{
    return FindPlaneIntersection(MakeFloat3(ray.origin), MakeFloat3(ray.direction), plane, t);
}

__device__ __forceinline__ bool FindPlaneIntersection1(const optix::Ray& ray, const float4& plane, float& t)
{
    float3 normal = make_float3(plane);
    float d = plane.w;

    float vd = dot(ray.direction, normal);

    if( vd == 0.0f ) return false;

    float v0 = -(dot(ray.origin, normal) + d);
    t = v0/vd;

    return (t > 0.0f);
}


template<typename T>
__device__ __forceinline__ int NumberOfTimesCrossesXAxis(const T& p0, const T& p1)
{
    ElVisFloat epsilon = MAKE_FLOAT(-.001);
    if( (p0.y < MAKE_FLOAT(0.0) && p1.y >= MAKE_FLOAT(0.0)) ||
       (p0.y >= MAKE_FLOAT(0.0) && p1.y < MAKE_FLOAT(0.0) ) )
    {
        // Changed from algrithm to include points at x=0.  This means
        // two adjacent polygons could both claim intersection, but then
        // we won't miss edges (hopefully).
        if( p0.x >= MAKE_FLOAT(0.0) && p1.x >= MAKE_FLOAT(0.0) )
            return 1;

        if( p0.x >= MAKE_FLOAT(0.0) || p1.x >= MAKE_FLOAT(0.0) )
        {
            // Kludge for numerical issues.
            //if( p0.y - p0.x*( (p1.y-p0.y)/(p1.x-p0.x) ) < epsilon )
            // Old version that didn't work.
            if( (p0.x - p0.y*(p1.x-p0.x)/(p1.y - p0.y)) > epsilon )

            //if( p0.x - p0.y*(p1.x-p0.x)/(p1.y - p0.y) > MAKE_FLOAT(0.0) )
            {
                return 1;
            }
        }
    }

    return 0;
}

template<typename T>
__device__ __forceinline__ bool ContainsOrigin(const T& p0, const T& p1, const T& p2)
{   
    int numCrossings = NumberOfTimesCrossesXAxis(p0, p1) +
        NumberOfTimesCrossesXAxis(p1, p2) +
        NumberOfTimesCrossesXAxis(p2, p0);
    
    return numCrossings & 0x01;

}


template<typename T>
__device__ __forceinline__ bool ContainsOrigin(const T& p0, const T& p1, const T& p2, const T& p3)
{   
    int numCrossings = NumberOfTimesCrossesXAxis(p0, p1) +
        NumberOfTimesCrossesXAxis(p1, p2) +
        NumberOfTimesCrossesXAxis(p2, p3) +
        NumberOfTimesCrossesXAxis(p3, p0);
    
    return numCrossings & 0x01;

}


__device__ __forceinline__ void GenerateUVWCoordinateSystem(const ElVisFloat3& W, ElVisFloat3& u, ElVisFloat3& v)
{
    ElVisFloat3 w = W;
    normalize(w);

    // From Pete Shirley's book.
    ElVisFloat3 z = w;
    ElVisFloat fabx = fabs(z.x);
    ElVisFloat faby = fabs(z.y);
    ElVisFloat fabz = fabs(z.z);
    if( fabx < fabz && faby < fabz )
    {
        z.x = MAKE_FLOAT(1.0);
    }
    else if( faby < fabx && fabz < fabx )
    {
        z.y = MAKE_FLOAT(1.0);
    }
    else
    {
        z.z = MAKE_FLOAT(1.0);
    }

    u = cross(z, w);
    normalize(u);

    v = cross(w, u);
}

__device__ __forceinline__ ElVisFloat EvaluatePlane(const ElVisFloat4& plane, const WorldPoint& p)
{
    return p.x*plane.x + p.y*plane.y + p.z*plane.z + plane.w;
}

__device__ __forceinline__ bool FindBoxEntranceAndExit(const float3& origin, const float3& direction, const ElVisFloat3& p0, const ElVisFloat3& p1, const ElVisFloat t0, const ElVisFloat t1, ElVisFloat& tmin, ElVisFloat& tmax)
{

    ElVisFloat tymin, tymax, tzmin, tzmax;

    ElVisFloat3 parameters[] = {p0, p1};
    int sign[] = { direction.x < 0, direction.y < 0, direction.z < 0 };

    ElVisFloat xinv = MAKE_FLOAT(1.0)/direction.x;
    ElVisFloat yinv = MAKE_FLOAT(1.0)/direction.y;
    ElVisFloat zinv = MAKE_FLOAT(1.0)/direction.z;

    tmin = (parameters[sign[0]].x - origin.x) * xinv;
    tmax = (parameters[1-sign[0]].x - origin.x) * xinv;
    tymin = (parameters[sign[1]].y - origin.y) * yinv;
    tymax = (parameters[1-sign[1]].y - origin.y) * yinv;
    if ( (tmin > tymax) || (tymin > tmax) )
      return false;
    if (tymin > tmin)
      tmin = tymin;
    if (tymax < tmax)
      tmax = tymax;
    tzmin = (parameters[sign[2]].z - origin.z) * zinv;
    tzmax = (parameters[1-sign[2]].z - origin.z) * zinv;
    if ( (tmin > tzmax) || (tzmin > tmax) )
      return false;
    if (tzmin > tmin)
      tmin = tzmin;
    if (tzmax < tmax)
      tmax = tzmax;
    return ( (tmin < t1) && (tmax > t0) );
}

#endif
